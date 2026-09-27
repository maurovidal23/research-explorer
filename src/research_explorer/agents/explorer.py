"""Explorer agent — an ACO "ant" that traverses the citation graph.

Each agent keeps its OWN incomplete, private view of the citation graph
(local_refs / local_cits in AgentState) and its OWN private pheromone trail.
Edges and pheromone are not shared across agents.

Per turn:
  1. If the current paper's neighbors haven't been discovered yet, "read" it:
     - use the provider's native references/citations when available (S2,
       OpenAlex, PubMed), OR
     - read the full text and have the LLM extract the references from the
       bibliography (arXiv, which has no citation API).
     The discovered neighbors go into the agent's private graph + frontier.
  2. Choose a direction (ref/cites) based on caste/phase weights.
  3. Choose a next paper via ACO transition (tau^alpha * eta^beta) over the
     agent's private pheromone and frontier.
  4. Fetch the paper (through the provider registry, by normalized ID prefix).
  5. Integrate the paper into the narrative (LLM). For full-text papers this
     same call also extracts the references.
  6. Update state (visited, frontier, budget) and discover the new paper's
     neighbors.

The heuristic eta(v) is computed via embeddings (cached in the shared
GraphStore, which now serves only as a metadata/embedding cache).
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import re
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import numpy as np

from research_explorer.aco.frontier import SharedFrontier
from research_explorer.aco.transition import (
    Candidate,
    EtaComponents,
    Selection,
    caste_direction_weights,
    combine_eta,
    mode_direction_modifier,
    select_candidate,
)
from research_explorer.agents.llm_client import LLMClient
from research_explorer.agents.prompts import (
    evaluate_references,
    integrate,
    integrate_and_extract,
)
from research_explorer.agents.state import AgentState, normalize_narrative
from research_explorer.config import Config
from research_explorer.graph.embeddings import EmbeddingService
from research_explorer.graph.models import Paper, PaperSummary, normalize_id, parse_normalized_id
from research_explorer.graph.store import GraphStore
from research_explorer.logging_setup import get_logger
from research_explorer.providers.base import ResilientProvider, TransientProviderError
from research_explorer.replay.trace import RunTracer

if TYPE_CHECKING:
    from research_explorer.resolution.traversal import ExpansionResult, NeighborExpander

log = get_logger("agent")

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


_STOP_WORDS = frozenset({"a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "by", "from"})


def _tokenize_title(title: str) -> set[str]:
    return {w for w in re.split(r"\W+", title.lower()) if w and w not in _STOP_WORDS}


def _titles_match(expected: str, actual: str, threshold: float = 0.3) -> bool:
    if not expected or not actual:
        return True
    exp_tokens = _tokenize_title(expected)
    act_tokens = _tokenize_title(actual)
    if not exp_tokens or not act_tokens:
        return True
    overlap = exp_tokens & act_tokens
    union = exp_tokens | act_tokens
    return len(overlap) / len(union) >= threshold


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(np.dot(va, vb) / norm)


@contextlib.asynccontextmanager
async def _strict_scope(provider: object) -> AsyncIterator[None]:
    """Ask a provider to raise transient failures, tolerating duck-typed fakes."""
    scope = getattr(provider, "strict_outcomes", None)
    if scope is None:
        yield
        return
    async with scope():
        yield


def _parse_json_response(text: str) -> dict | None:
    """Best-effort parse of an LLM JSON response (strips code fences).

    Only a top-level JSON object is accepted. Handles truncated JSON by
    extracting the narrative and any complete reference objects that were
    returned before truncation.
    """
    cleaned = _JSON_FENCE.sub("", text.strip())
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    # Try extracting the JSON object span
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # Last resort: recover narrative + complete ref objects from truncated JSON
    return _recover_partial_json(cleaned)


def _recover_partial_json(text: str) -> dict | None:
    """Recover narrative, paper_analysis, and complete reference objects from truncated JSON."""
    import re

    result: dict = {}

    m = re.search(r'"narrative"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        try:
            result["narrative"] = m.group(1).encode().decode("unicode_escape")
        except Exception:
            result["narrative"] = m.group(1)

    analysis_match = re.search(r'"paper_analysis"\s*:\s*\{', text)
    if analysis_match:
        start = analysis_match.end() - 1
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if depth == 0:
            with contextlib.suppress(json.JSONDecodeError):
                result["paper_analysis"] = json.loads(text[start:end])

    refs: list[dict] = []
    for m in re.finditer(r'\{[^{}]*?"title"\s*:\s*"[^"]*"[^{}]*?\}', text):
        try:
            obj = json.loads(m.group(0))
            if "title" in obj:
                refs.append(obj)
        except json.JSONDecodeError:
            continue
    if refs:
        result["references"] = refs

    if not result:
        return None
    return result


_PROVIDER_CONFIDENCE = {
    "openalex": 0.95,
    "semantic_scholar": 0.9,
    "pubmed": 0.85,
    "arxiv": 0.8,
}


def _provider_confidence_base(provider: str) -> float:
    return _PROVIDER_CONFIDENCE.get(provider, 0.4)


def _mode_dir_modifier(mode: str, ref_weight: float, cites_weight: float) -> float:
    return mode_direction_modifier(mode, ref_weight, cites_weight)


class ExplorerAgent:
    """An ACO agent that explores the citation graph and builds a narrative."""

    def __init__(
        self,
        state: AgentState,
        graph: GraphStore,
        llm: LLMClient,
        embedding: EmbeddingService,
        provider: ResilientProvider,
        config: Config,
        seed_query: str,
        seed_embedding: list[float] | None = None,
        providers: dict[str, ResilientProvider] | None = None,
        shared_visited: set[str] | None = None,
        shared_frontier: SharedFrontier | None = None,
        expander: NeighborExpander | None = None,
        rng: random.Random | None = None,
    ):
        self.state = state
        self.graph = graph
        self.llm = llm
        self.embedding = embedding
        self.provider = provider
        self.providers = providers or {provider.name: provider}
        self.cfg = config
        self.seed_query = seed_query
        self.seed_embedding = seed_embedding
        self._shared_visited = shared_visited if shared_visited is not None else set()
        self._frontier = shared_frontier if shared_frontier is not None else SharedFrontier()
        self.expander = expander
        self.tracer: RunTracer | None = None
        self.rng = rng if rng is not None else random.Random()
        self._eta_cache: dict[str, EtaComponents] = {}
        self._llm_priority: dict[str, float] = {}

    def _emit(self, type: str, **payload) -> None:
        if self.tracer is not None:
            self.tracer.emit(type, **payload)

    async def take_turn(self, k: int) -> list[tuple[str, str, str]]:
        """Execute one turn: fetch k papers, integrate narratives.

        Best-first search: at each step, pick the highest-scored paper from
        the shared frontier (regardless of which paper referenced it),
        claim it in the shared visited set, fetch it, integrate it, discover
        its neighbors, and evaluate the new references with the LLM.

        Returns the list of (src, dst, mode) edges traversed for pheromone deposit.
        """
        self.state.start_turn()
        edges: list[tuple[str, str, str]] = []

        # Ensure the seed has discovered neighbors and they're evaluated
        if not self.state.is_discovered(self.state.pos):
            await self._discover_neighbors(self.state.pos)
            await self._evaluate_new_refs()

        inflight: str | None = None
        try:
            for _ in range(k):
                if self.state.budget <= 0:
                    break

                selection = await self._select_candidate()
                if selection is None:
                    log.debug("agent_no_candidates", agent=self.state.id, pos=self.state.pos)
                    break

                next_id = selection.chosen
                src, mode = selection.src, selection.mode

                expected_summary = self.graph.get_paper_summary(next_id)
                expected_title = expected_summary.title if expected_summary else ""
                self._shared_visited.add(next_id)
                inflight = next_id
                provider_name, _ = parse_normalized_id(next_id)
                self._emit(
                    "paper_fetch_started",
                    agent_id=self.state.id,
                    paper_id=next_id,
                    mode=mode,
                    provider=provider_name,
                    src=src,
                    turn=self.state.turn_count,
                )

                try:
                    paper = await self._fetch_paper(next_id)
                except TransientProviderError as exc:
                    self._shared_visited.discard(next_id)
                    inflight = None
                    self._frontier.record_transient_failure(
                        next_id, self.state.id, self.state.turn_count
                    )
                    log.warning(
                        "provider_transient",
                        paper_id=next_id,
                        provider=exc.provider,
                        reason=exc.reason,
                    )
                    self._emit(
                        "paper_fetch_failed",
                        agent_id=self.state.id,
                        paper_id=next_id,
                        provider=exc.provider,
                        reason=exc.reason,
                        classification="transient",
                        attempts=self._frontier.attempt_count(next_id),
                    )
                    self._emit(
                        "provider_failure",
                        paper_id=next_id,
                        provider=exc.provider,
                        reason=exc.reason,
                        classification="transient",
                        attempts=self._frontier.attempt_count(next_id),
                    )
                    continue

                owned = self._frontier.claim_for(next_id, self.state.id)
                if not owned:
                    self._shared_visited.discard(next_id)
                    inflight = None
                    self._frontier.release(next_id, self.state.id)
                    continue

                if paper is None:
                    if await self._metadata_transit(next_id, src, mode):
                        inflight = None
                        self._frontier.release(next_id, self.state.id)
                        continue
                    self._shared_visited.discard(next_id)
                    inflight = None
                    self._frontier.record_absence(next_id)
                    continue

                if not _titles_match(expected_title, paper.title):
                    log.warning(
                        "id_title_mismatch",
                        paper_id=next_id,
                        expected_title=expected_title,
                        actual_title=paper.title,
                    )
                    if self.tracer is not None:
                        self.tracer.emit(
                            "id_title_mismatch",
                            paper_id=next_id,
                            expected_title=expected_title,
                            actual_title=paper.title,
                            reason="id_title_mismatch",
                        )
                    self._shared_visited.discard(next_id)
                    inflight = None
                    self._frontier.record_absence(next_id)
                    continue

                if getattr(self, "expander", None) is not None and not (
                    paper.fulltext or paper.abstract
                ):
                    if await self._metadata_transit(next_id, src, mode, paper=paper):
                        inflight = None
                        self._frontier.release(next_id, self.state.id)
                        continue
                    self._shared_visited.discard(next_id)
                    inflight = None
                    self._frontier.record_absence(next_id)
                    continue

                self.graph.cache_paper(paper)
                self._emit(
                    "paper_fetch_completed",
                    agent_id=self.state.id,
                    paper_id=next_id,
                    title=paper.title,
                    year=paper.year,
                    authors=list(paper.authors),
                    provider=paper.provider,
                    mode=mode,
                    src=src,
                    turn=self.state.turn_count,
                )
                self._emit(
                    "paper_integration_started",
                    agent_id=self.state.id,
                    paper_id=next_id,
                    title=paper.title,
                    mode=mode,
                    src=src,
                    turn=self.state.turn_count,
                )
                narrative, extracted = await self._integrate(paper)
                self.state.narrative = narrative
                if paper.fulltext or paper.abstract:
                    self.graph.mark_integrated(next_id)

                self._emit(
                    "paper_integration_completed",
                    agent_id=self.state.id,
                    paper_id=next_id,
                    title=paper.title,
                    year=paper.year,
                    authors=list(paper.authors),
                    provider=paper.provider,
                    mode=mode,
                    src=src,
                    turn=self.state.turn_count,
                    analysis=self.state.paper_analyses.get(next_id),
                )

                self.state.visit(next_id, mode)
                self._frontier.remove(next_id)
                inflight = None
                edges.append((src, next_id, mode))

                self._emit(
                    "neighbor_discovery_started",
                    agent_id=self.state.id,
                    paper_id=next_id,
                    turn=self.state.turn_count,
                )
                await self._discover_neighbors(next_id, paper, extracted)
                self._emit(
                    "neighbor_discovery_completed",
                    agent_id=self.state.id,
                    paper_id=next_id,
                    turn=self.state.turn_count,
                    refs=len(self.state.local_references(next_id)),
                    cits=len(self.state.local_citants(next_id)),
                )

                await self._evaluate_new_refs()

                log.info(
                    "agent_step",
                    agent=self.state.id,
                    mode=mode,
                    title=paper.title[:60],
                    year=paper.year,
                    budget=self.state.budget,
                    frontier=len(self._frontier),
                )
                if self.tracer is not None:
                    self.tracer.emit(
                        "agent_step",
                        agent=self.state.id,
                        mode=mode,
                        paper_id=next_id,
                        title=paper.title,
                        year=paper.year,
                        budget=self.state.budget,
                        frontier=len(self._frontier),
                    )
        finally:
            # Exception paths must never strand a claim in the shared frontier
            # or leave an unfinalized id in the shared visited set (STAB-3).
            if inflight is not None:
                self._shared_visited.discard(inflight)
            self._frontier.release_all(self.state.id)

        self.state.turn_count += 1
        return edges

    def _direction_weights(self) -> tuple[float, float]:
        """Caste-adjusted (ref_weight, cites_weight) for this agent."""
        return caste_direction_weights(
            self.state.caste,
            self.cfg.direction.ref_weight,
            self.cfg.direction.cites_weight,
        )

    async def _select_candidate(self) -> Selection | None:
        """Pick the next candidate via tau^alpha * eta^beta transition weights.

        Builds a ``Candidate`` per eligible frontier node (not visited, not
        claimed), computes tau from the agent's private pheromone and the
        caste-adjusted direction modifier, then draws via ``select_candidate``.
        The chosen node is claimed so no concurrent agent duplicates the work.
        """
        eligible = self._frontier.eligible(
            exclude=self._shared_visited, turn=self.state.turn_count
        )
        if not eligible:
            return None

        ref_w, cites_w = self._direction_weights()
        candidates_list: list[Candidate] = []
        for pid in eligible:
            src, mode = self._frontier.sources.get(pid, (self.state.pos, "ref"))
            components = await self._eta_components(pid)
            tau = self.state.get_pheromone(src, pid, mode)
            dm = _mode_dir_modifier(mode, ref_w, cites_w)
            candidates_list.append(
                Candidate(pid=pid, src=src, mode=mode, tau=tau, eta=components.value, dir_modifier=dm)
            )

        guard = 0
        max_guard = max(1, len(candidates_list) + 1)
        while candidates_list and guard < max_guard:
            guard += 1
            selection = select_candidate(
                candidates_list,
                self.cfg.aco.alpha,
                self.cfg.aco.beta,
                self.cfg.aco.epsilon,
                self.rng,
            )
            if selection is None:
                return None
            if self._frontier.claim_for(selection.chosen, self.state.id):
                self._emit_candidate_selected(selection)
                return selection
            candidates_list = [c for c in candidates_list if c.pid != selection.chosen]

        return None

    def _emit_candidate_selected(self, selection: Selection) -> None:
        if self.tracer is None:
            return
        record = getattr(self.tracer, "record_candidate_selected", None)
        if record is None:
            return
        from research_explorer.replay.models import CandidateSelection

        component = self._eta_cache.get(selection.chosen)
        record(
            CandidateSelection(
                agent_id=self.state.id,
                paper_id=selection.chosen,
                src=selection.src,
                mode=selection.mode,
                caste=self.state.caste,
                dir_modifier=selection.dir_modifier,
                tau=selection.tau,
                alpha=selection.alpha,
                beta=selection.beta,
                eta=component.value if component else selection.eta,
                final_weight=selection.final_weight,
                probability=selection.chosen_probability,
                epsilon_branch=selection.epsilon_branch,
                chosen=True,
                rationale=self._selection_rationale(selection),
            )
        )

    def _selection_rationale(self, selection: Selection) -> str:
        branch = "epsilon" if selection.epsilon_branch else "weighted"
        tau_term = max(1e-12, selection.tau) ** selection.alpha if selection.tau > 0 else 0.0
        eta_term = max(0.0, selection.eta) ** selection.beta
        return (
            f"caste={self.state.caste} mode={selection.mode} "
            f"tau^alpha={tau_term:.4f} eta^beta={eta_term:.4f} "
            f"dir={selection.dir_modifier:.4f} branch={branch}"
        )

    def _provider_confidence(self, summary: PaperSummary | None, provider: str | None = None) -> float:
        if summary is None:
            prov = provider or parse_normalized_id(self.state.pos)[0]
            return max(0.0, min(1.0, _provider_confidence_base(prov) - 0.2))
        prov = provider or summary.provider
        conf = _provider_confidence_base(prov)
        if summary.doi or summary.arxiv_id:
            conf += 0.05
        if summary.abstract:
            conf += 0.05
        if not summary.title:
            conf -= 0.2
        return max(0.0, min(1.0, conf))

    async def _eta_components(
        self, paper_id: str, embedding: list[float] | None = None,
        llm_priority: float | None = None,
    ) -> EtaComponents:
        """Compute (and cache) the eta decomposition for a frontier candidate."""
        cached = self._eta_cache.get(paper_id)
        if cached is not None:
            return cached

        summary = self.graph.get_paper_summary(paper_id)
        h = self.cfg.heuristica
        weights = {
            "w_sim": h.w_sim,
            "w_citas": h.w_citas,
            "w_recencia": h.w_recencia,
            "w_confidence": h.w_confidence,
            "w_llm": h.w_llm,
        }

        if summary is None:
            prov = parse_normalized_id(paper_id)[0]
            value = combine_eta(0.5, 0.5, 0.5, self._provider_confidence(None, prov), 0.0,
                                h.w_sim, h.w_citas, h.w_recencia, h.w_confidence, h.w_llm, h.eta_llm)
            eta = EtaComponents(sim=0.5, citations=0.5, recency=0.5,
                                confidence=self._provider_confidence(None, prov), llm=0.0,
                                weights=weights, value=value)
            self._cache_eta(paper_id, eta)
            return eta

        sim = 0.5
        if self.seed_embedding is not None:
            emb = embedding
            if emb is None:
                emb = self.graph.get_paper_embedding(paper_id)
            if emb is None:
                text = f"{summary.title} {summary.abstract or ''}"
                try:
                    emb = await self.embedding.embed(text)
                    self.graph.set_paper_embedding(paper_id, emb)
                except Exception:
                    emb = None
            if emb is not None:
                sim = (_cosine_sim(self.seed_embedding, emb) + 1.0) / 2.0
                sim = max(0.0, min(1.0, sim))

        citations_val = _sigmoid(math.log1p(summary.citation_count or 0) / 5.0)
        year = summary.year or 2000
        recency = max(0.0, min(1.0, (year - 1950) / 75.0))
        conf = self._provider_confidence(summary)
        llm_val = (
            llm_priority if llm_priority is not None
            else self._llm_priority.get(paper_id, 0.0)
        )
        value = combine_eta(sim, citations_val, recency, conf, llm_val,
                            h.w_sim, h.w_citas, h.w_recencia, h.w_confidence, h.w_llm, h.eta_llm)
        eta = EtaComponents(sim=sim, citations=citations_val, recency=recency,
                            confidence=conf, llm=llm_val, weights=weights, value=value)
        self._cache_eta(paper_id, eta)
        return eta

    def _cache_eta(self, paper_id: str, eta: EtaComponents) -> None:
        self._eta_cache[paper_id] = eta
        if self.tracer is None:
            return
        record = getattr(self.tracer, "record_candidate_score", None)
        if record is None:
            return
        from research_explorer.replay.models import CandidateScore

        src, mode = self._frontier.sources.get(paper_id, ("", "ref"))
        prov = parse_normalized_id(paper_id)[0]
        record(
            CandidateScore(
                paper_id=paper_id,
                agent_id=self.state.id,
                provider=prov,
                components=eta.components(),
                weights=dict(eta.weights),
                eta=eta.value,
                llm_used=bool(getattr(self.cfg.heuristica, "eta_llm", False)),
                source=src,
                mode=mode,
            )
        )

    async def _evaluate_new_refs(self) -> None:
        """Evaluate unevaluated frontier papers with a batch LLM call.

        The LLM scores each candidate (0.0-1.0) for exploration priority
        based on the research line and current narrative context. Only
        new (unevaluated) refs are scored; old refs keep their scores.
        """
        new_refs = self._frontier.unevaluated()
        if not new_refs:
            return
        self._emit(
            "frontier_reference_evaluation_started",
            agent_id=self.state.id,
            count=len(new_refs),
            turn=self.state.turn_count,
        )

        # Build PaperSummary list for the prompt (from shared store)
        candidates: list[PaperSummary] = []
        id_map: dict[str, str] = {}
        for pid in new_refs:
            summary = self.graph.get_paper_summary(pid)
            if summary is not None:
                candidates.append(summary)
                id_map[summary.id] = pid

        if not candidates:
            for pid in new_refs:
                eta = await self._heuristic(pid)
                self._frontier.set_score(pid, eta)
            self._emit(
                "frontier_reference_evaluation_completed",
                agent_id=self.state.id,
                count=len(new_refs),
                turn=self.state.turn_count,
            )
            return

        MAX_BATCH = 30
        if len(candidates) > MAX_BATCH:
            candidates = candidates[:MAX_BATCH]

        messages = evaluate_references(
            self.seed_query, self.state.narrative, candidates
        )
        try:
            raw = await self.llm.chat(
                messages,
                model=self.cfg.llm.explorer_model,
                temperature=0.3,
                max_tokens=min(self.cfg.llm.max_tokens, 2000),
                purpose="frontier_reference_evaluation",
            )
        except Exception as e:
            log.warning("eval_refs_failed", agent=self.state.id, error=str(e))
            for pid in new_refs:
                eta = await self._heuristic(pid)
                self._frontier.set_score(pid, eta)
            self._emit(
                "frontier_reference_evaluation_completed",
                agent_id=self.state.id,
                count=len(new_refs),
                turn=self.state.turn_count,
                degraded=True,
            )
            return

        scores = self._parse_eval_response(raw)
        for s in scores:
            raw_id = s.get("id", "")
            score = s.get("score", 0.0)
            pid = id_map.get(raw_id, raw_id)
            if pid and isinstance(score, (int, float)):
                self._llm_priority[pid] = float(score)
                self._eta_cache.pop(pid, None)

        for pid in new_refs:
            if pid not in self._frontier.scores:
                eta = await self._heuristic(pid)
                self._frontier.set_score(pid, eta)
        self._emit(
            "frontier_reference_evaluation_completed",
            agent_id=self.state.id,
            count=len(new_refs),
            turn=self.state.turn_count,
        )

    def _parse_eval_response(self, raw: str) -> list[dict]:
        """Parse the LLM's reference evaluation response.

        Validates that the payload is an object with a list of objects under
        ``scores``; malformed individual entries are ignored rather than
        crashing the turn.
        """
        cleaned = _JSON_FENCE.sub("", raw.strip())
        obj: object = None
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                with contextlib.suppress(json.JSONDecodeError):
                    obj = json.loads(cleaned[start : end + 1])
        if not isinstance(obj, dict):
            return []
        scores = obj.get("scores")
        if not isinstance(scores, list):
            return []
        return [entry for entry in scores if isinstance(entry, dict)]

    async def _batch_embeddings(
        self, paper_ids: list[str]
    ) -> dict[str, list[float]]:
        """Pre-compute embeddings for multiple papers in a single batch API call."""
        need_embed: list[str] = []
        texts: list[str] = []
        result: dict[str, list[float]] = {}

        for pid in paper_ids:
            emb = self.graph.get_paper_embedding(pid)
            if emb is not None:
                result[pid] = emb
                continue
            summary = self.graph.get_paper_summary(pid)
            if summary is None:
                continue
            text = f"{summary.title} {summary.abstract or ''}"
            need_embed.append(pid)
            texts.append(text)

        if texts:
            try:
                vecs = await self.embedding.embed_batch(texts)
                for pid, vec in zip(need_embed, vecs, strict=False):
                    if vec is not None:
                        self.graph.set_paper_embedding(pid, vec)
                        result[pid] = vec
            except Exception as e:
                log.warning("batch_embed_failed", error=str(e))

        return result

    async def _heuristic(
        self, paper_id: str, embedding: list[float] | None = None,
        llm_priority: float | None = None,
    ) -> float:
        """Compute eta(v) -- the combined relevance heuristic, cached per node."""
        components = await self._eta_components(paper_id, embedding=embedding, llm_priority=llm_priority)
        return components.value

    def _provider_for(self, paper_id: str) -> ResilientProvider:
        """Pick the provider for a normalized ID by its prefix; fall back to default."""
        provider_name, _ = parse_normalized_id(paper_id)
        return self.providers.get(provider_name, self.provider)

    async def _fetch_paper(self, paper_id: str) -> Paper | None:
        """Fetch a paper via the right provider; attach full text when needed.

        The shared GraphStore does not persist references or full text, so we
        always go through the provider (which has its own disk cache) to obtain
        a Paper with its native neighbors / full text.

        Definitive absence returns ``None``; a transient provider failure raises
        :class:`TransientProviderError` so the caller can retain the candidate.
        Unexpected provider/transport errors are translated to the same typed
        transient error rather than degraded to ``None``, which would otherwise
        masquerade as definitive absence and evict a live candidate.
        """
        provider = self._provider_for(paper_id)
        provider_name, native_id = parse_normalized_id(paper_id)
        native = native_id if provider.name == provider_name else (
            paper_id.split(":", 1)[1] if ":" in paper_id else paper_id
        )

        try:
            async with _strict_scope(provider):
                paper = await provider.get_paper(native)
        except TransientProviderError:
            raise
        except Exception as e:
            log.warning("fetch_failed", paper_id=paper_id, error=str(e))
            raise TransientProviderError(provider.name, "unexpected_error") from e

        if paper is None:
            return None

        # Attach full text + bibliography for providers without a citation API.
        # The provider tries HTML first, then falls back to PDF; returns None
        # only if both are unavailable (rare), in which case we skip the paper.
        if not paper.references and not paper.citations and getattr(provider, "supports_fulltext", False):
            try:
                async with _strict_scope(provider):
                    ft = await provider.get_fulltext_and_refs(
                        native, max_chars=self.cfg.llm.fulltext_max_chars, ref_limit=50
                    )
            except TransientProviderError:
                raise
            except Exception as e:
                log.warning("fulltext_failed", paper_id=paper_id, error=str(e))
                raise TransientProviderError(
                    provider.name, "fulltext_unexpected_error"
                ) from e
            if ft is None:
                log.info("fulltext_unavailable_skip", paper_id=paper_id, provider=provider.name)
                return None
            paper.fulltext, paper.ref_entries = ft

        return paper

    async def _integrate(self, paper: Paper) -> tuple[str, list[PaperSummary]]:
        """Integrate a paper into the narrative; extract refs for full-text papers.

        Returns (narrative, extracted_references). For providers with native
        references, extracted is empty and the caller uses paper.references.
        Also stores the per-paper analysis in state.paper_analyses.
        """
        if paper.ref_entries:
            messages = integrate_and_extract(self.state.narrative, paper, self.seed_query)
            try:
                raw = await self.llm.chat(
                    messages,
                    model=self.cfg.llm.explorer_model,
                    temperature=self.cfg.llm.temperature,
                    max_tokens=self.cfg.llm.max_tokens,
                    purpose="paper_integration",
                )
            except Exception as e:
                log.warning("integrate_extract_failed", agent=self.state.id, error=str(e))
                return self.state.narrative, []
            parsed = _parse_json_response(raw)
            if parsed is None:
                return self.state.narrative, []
            narrative = normalize_narrative(parsed.get("narrative"), self.state.narrative)
            extracted = self._parse_extracted_refs(parsed.get("references") or [])

            analysis = parsed.get("paper_analysis")
            if isinstance(analysis, dict):
                paper_nid = normalize_id(paper.provider, paper.id)
                self.state.paper_analyses[paper_nid] = analysis

            return narrative, extracted

        messages = integrate(self.state.narrative, paper, self.seed_query)
        try:
            narrative = await self.llm.chat(
                messages,
                model=self.cfg.llm.explorer_model,
                temperature=self.cfg.llm.temperature,
                max_tokens=self.cfg.llm.max_tokens,
                purpose="paper_integration",
            )
        except Exception as e:
            log.warning("integrate_failed", agent=self.state.id, error=str(e))
            return self.state.narrative, []
        return normalize_narrative(narrative, self.state.narrative), []

    def _parse_extracted_refs(self, refs: list) -> list[PaperSummary]:
        """Turn the LLM's extracted reference list into PaperSummary candidates.

        Only entries with an arXiv id or DOI are traversable and go into the
        frontier. Entries with neither are still returned (graph structure) but
        the caller keeps them out of the frontier.
        """
        summaries: list[PaperSummary] = []
        for r in refs:
            if not isinstance(r, dict):
                continue
            title = (r.get("title") or "").strip()
            if not title:
                continue
            authors = r.get("authors") or []
            if isinstance(authors, str):
                authors = [a.strip() for a in authors.split(",") if a.strip()]
            year = r.get("year")
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            if not isinstance(year, int):
                year = None
            arxiv_id = (r.get("arxiv_id") or "").strip() or None
            doi = (r.get("doi") or "").strip() or None
            if arxiv_id:
                summaries.append(PaperSummary(
                    id=arxiv_id, doi=doi, title=title, year=year,
                    authors=list(authors), provider="arxiv",
                ))
            elif doi:
                provider = "semantic_scholar" if "semantic_scholar" in self.providers else (
                    "openalex" if "openalex" in self.providers else self.provider.name
                )
                summaries.append(PaperSummary(
                    id=doi, doi=doi, title=title, year=year,
                    authors=list(authors), provider=provider,
                ))
            else:
                summaries.append(PaperSummary(
                    id=title, title=title, year=year,
                    authors=list(authors), provider="unknown",
                ))
        return summaries

    async def _metadata_transit(
        self, paper_id: str, src: str, mode: str, paper: Paper | None = None
    ) -> bool:
        """Traverse a metadata-only node: expand it without integrating content.

        Metadata-only nodes stay expandable (neighbors are discovered through
        the provider expander) but receive no narrative/evaluation evidence
        credit: the fetch slot is consumed, no visit is recorded, no traversal
        edge is returned, and the node is not marked integrated.
        """
        expander = getattr(self, "expander", None)
        if expander is None:
            return False
        summary = self.graph.get_paper_summary(paper_id)
        if summary is None and paper is None:
            return False
        if self.tracer is not None:
            provider = (
                summary.provider
                if summary is not None
                else (paper.provider if paper is not None else "unknown")
            )
            self.tracer.emit(
                "metadata_transit",
                paper_id=paper_id,
                src=src,
                mode=mode,
                provider=provider,
                reason="full_text_unavailable" if paper is None else "metadata_only",
            )
        result = await expander.expand(paper_id, paper=paper, tracer=self.tracer)
        self._add_expanded_to_frontier(paper_id, result)
        self.state.set_local_neighbors(
            paper_id, result.outgoing.node_ids, result.incoming.node_ids
        )
        self.state.record_transit(paper_id)
        self._frontier.remove(paper_id)
        return True

    def _add_expanded_to_frontier(self, paper_id: str, result: ExpansionResult) -> None:
        if result.outgoing.node_ids:
            self._frontier.add(
                result.outgoing.node_ids, source=paper_id, mode="ref",
                exclude=self._shared_visited,
            )
        if result.incoming.node_ids:
            self._frontier.add(
                result.incoming.node_ids, source=paper_id, mode="cites",
                exclude=self._shared_visited,
            )

    async def _discover_neighbors(
        self,
        paper_id: str,
        paper: Paper | None = None,
        extracted: list[PaperSummary] | None = None,
    ) -> None:
        """Read a paper and record its neighbors in the agent's private graph.

        With an expander attached, both directions come from provider-verified
        canonical expansion (OpenAlex primary, Semantic Scholar fallback) plus
        LLM-extracted bibliography entries. Without one, the legacy native-list
        path is used.
        """
        if self.state.is_discovered(paper_id):
            return

        expander = getattr(self, "expander", None)
        if expander is not None:
            await self._discover_via_expander(paper_id, paper, extracted, expander)
            return

        if paper is None:
            paper = await self._fetch_paper(paper_id)
            if paper is None:
                self.state.set_local_neighbors(paper_id, [], [])
                return
            self.graph.cache_paper(paper)

        # Full-text papers (arXiv): extract references from the bibliography.
        # _fetch_paper already attached fulltext + ref_entries. The seed is read
        # here (its narrative is integrated as a side effect); visited papers
        # pass `extracted` in so we don't integrate twice.
        if extracted is None and not paper.references and not paper.citations and paper.ref_entries:
            narrative, extracted = await self._integrate(paper)
            self.state.narrative = narrative

        if extracted is None:
            extracted = []

        ref_summaries, cit_summaries = self._neighbor_summaries(paper, extracted)

        ref_ids: list[str] = []
        cit_ids: list[str] = []
        ref_frontier: list[str] = []
        cit_frontier: list[str] = []
        for s in ref_summaries:
            nid = normalize_id(s.provider, s.id)
            ref_ids.append(nid)
            self.graph.cache_summary(s)
            if s.provider != "unknown":
                ref_frontier.append(nid)
        for s in cit_summaries:
            nid = normalize_id(s.provider, s.id)
            cit_ids.append(nid)
            self.graph.cache_summary(s)
            if s.provider != "unknown":
                cit_frontier.append(nid)

        self.state.set_local_neighbors(paper_id, ref_ids, cit_ids)
        if ref_frontier:
            self._frontier.add(ref_frontier, source=paper_id, mode="ref", exclude=self._shared_visited)
        if cit_frontier:
            self._frontier.add(cit_frontier, source=paper_id, mode="cites", exclude=self._shared_visited)

        log.info(
            "discovered_neighbors",
            agent=self.state.id,
            paper=paper_id,
            refs=len(ref_ids),
            cits=len(cit_ids),
            frontier_added=len(ref_frontier) + len(cit_frontier),
        )

    async def _discover_via_expander(
        self,
        paper_id: str,
        paper: Paper | None,
        extracted: list[PaperSummary] | None,
        expander: NeighborExpander,
    ) -> None:
        """Canonical neighbor discovery via the resolution expander."""
        if paper is None:
            paper = await self._fetch_paper(paper_id)
            if paper is not None:
                self.graph.cache_paper(paper)
        if (
            extracted is None
            and paper is not None
            and not paper.references
            and not paper.citations
            and paper.ref_entries
        ):
            narrative, extracted = await self._integrate(paper)
            self.state.narrative = narrative
        if extracted is None:
            extracted = []

        result = await expander.expand(
            paper_id, paper=paper, extracted=extracted, tracer=self.tracer
        )
        self._add_expanded_to_frontier(paper_id, result)
        self.state.set_local_neighbors(
            paper_id, result.outgoing.node_ids, result.incoming.node_ids
        )
        log.info(
            "discovered_neighbors",
            agent=self.state.id,
            paper=paper_id,
            refs=len(result.outgoing.node_ids),
            cits=len(result.incoming.node_ids),
            frontier_added=len(result.outgoing.node_ids) + len(result.incoming.node_ids),
        )

    def _neighbor_summaries(
        self, paper: Paper, extracted: list[PaperSummary]
    ) -> tuple[list[PaperSummary], list[PaperSummary]]:
        """Return (references, citations) summaries for a paper.

        Prefers the provider's native lists; falls back to LLM-extracted refs
        (outgoing only — arXiv has no incoming-citation source).
        """
        if paper.references or paper.citations:
            return list(paper.references), list(paper.citations)
        return extracted, []
