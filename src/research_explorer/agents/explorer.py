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
import re

import numpy as np

from research_explorer.aco.frontier import SharedFrontier
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
from research_explorer.providers.base import ResilientProvider
from research_explorer.replay.trace import RunTracer

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


def _parse_json_response(text: str) -> dict | None:
    """Best-effort parse of an LLM JSON response (strips code fences).

    Handles truncated JSON by extracting the narrative and any complete
    reference objects that were returned before truncation.
    """
    cleaned = _JSON_FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try extracting the JSON object span
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
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
        self.tracer: RunTracer | None = None

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

        for _ in range(k):
            if self.state.budget <= 0:
                break

            next_id = self._frontier.best(exclude=self._shared_visited)
            if next_id is None:
                log.debug("agent_no_candidates", agent=self.state.id, pos=self.state.pos)
                break

            src, mode = self._frontier.sources.get(next_id, (self.state.pos, "ref"))

            expected_summary = self.graph.get_paper_summary(next_id)
            expected_title = expected_summary.title if expected_summary else ""
            self._shared_visited.add(next_id)

            paper = await self._fetch_paper(next_id)
            if paper is None:
                self._shared_visited.discard(next_id)
                self._frontier.remove(next_id)
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
                self._frontier.remove(next_id)
                continue

            self.graph.cache_paper(paper)
            narrative, extracted = await self._integrate(paper)
            self.state.narrative = narrative

            self.state.visit(next_id, mode)
            self._frontier.remove(next_id)
            edges.append((src, next_id, mode))

            await self._discover_neighbors(next_id, paper, extracted)

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

        self.state.turn_count += 1
        return edges

    async def _evaluate_new_refs(self) -> None:
        """Evaluate unevaluated frontier papers with a batch LLM call.

        The LLM scores each candidate (0.0-1.0) for exploration priority
        based on the research line and current narrative context. Only
        new (unevaluated) refs are scored; old refs keep their scores.
        """
        new_refs = self._frontier.unevaluated()
        if not new_refs:
            return

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
            )
        except Exception as e:
            log.warning("eval_refs_failed", agent=self.state.id, error=str(e))
            for pid in new_refs:
                eta = await self._heuristic(pid)
                self._frontier.set_score(pid, eta)
            return

        scores = self._parse_eval_response(raw)
        for s in scores:
            raw_id = s.get("id", "")
            score = s.get("score", 0.0)
            pid = id_map.get(raw_id, raw_id)
            if pid and isinstance(score, (int, float)):
                self._frontier.set_score(pid, float(score))

        still_unevaluated = self._frontier.unevaluated()
        for pid in still_unevaluated:
            eta = await self._heuristic(pid)
            self._frontier.set_score(pid, eta)

    def _parse_eval_response(self, raw: str) -> list[dict]:
        """Parse the LLM's reference evaluation response."""
        cleaned = _JSON_FENCE.sub("", raw.strip())
        try:
            obj = json.loads(cleaned)
            return obj.get("scores", []) if isinstance(obj, dict) else []
        except json.JSONDecodeError:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                obj = json.loads(cleaned[start : end + 1])
                return obj.get("scores", []) if isinstance(obj, dict) else []
            except json.JSONDecodeError:
                pass
        return []

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
        self, paper_id: str, embedding: list[float] | None = None
    ) -> float:
        """Compute eta(v) -- relevance heuristic, cached in the graph store."""
        summary = self.graph.get_paper_summary(paper_id)
        if summary is None:
            return 0.5

        sim = 0.5
        if self.seed_embedding is not None:
            if embedding is not None:
                sim = _cosine_sim(self.seed_embedding, embedding)
            else:
                emb = self.graph.get_paper_embedding(paper_id)
                if emb is None:
                    text = f"{summary.title} {summary.abstract or ''}"
                    try:
                        emb = await self.embedding.embed(text)
                        self.graph.set_paper_embedding(paper_id, emb)
                    except Exception:
                        emb = None
                if emb is not None:
                    sim = _cosine_sim(self.seed_embedding, emb)

        citations = summary.citation_count or 0
        norm_citas = _sigmoid(math.log1p(citations) / 5.0)

        year = summary.year or 2000
        recency = max(0.0, min(1.0, (year - 1950) / 75.0))

        h = self.cfg.heuristica
        value = h.w_sim * sim + h.w_citas * norm_citas + h.w_recencia * recency
        return _sigmoid(value)

    def _provider_for(self, paper_id: str) -> ResilientProvider:
        """Pick the provider for a normalized ID by its prefix; fall back to default."""
        provider_name, _ = parse_normalized_id(paper_id)
        return self.providers.get(provider_name, self.provider)

    async def _fetch_paper(self, paper_id: str) -> Paper | None:
        """Fetch a paper via the right provider; attach full text when needed.

        The shared GraphStore does not persist references or full text, so we
        always go through the provider (which has its own disk cache) to obtain
        a Paper with its native neighbors / full text.
        """
        provider = self._provider_for(paper_id)
        provider_name, native_id = parse_normalized_id(paper_id)
        native = native_id if provider.name == provider_name else (
            paper_id.split(":", 1)[1] if ":" in paper_id else paper_id
        )

        try:
            paper = await provider.get_paper(native)
        except Exception as e:
            log.warning("fetch_failed", paper_id=paper_id, error=str(e))
            return None

        if paper is None:
            return None

        # Attach full text + bibliography for providers without a citation API.
        # The provider tries HTML first, then falls back to PDF; returns None
        # only if both are unavailable (rare), in which case we skip the paper.
        if not paper.references and not paper.citations and getattr(provider, "supports_fulltext", False):
            try:
                ft = await provider.get_fulltext_and_refs(
                native, max_chars=self.cfg.llm.fulltext_max_chars, ref_limit=50
            )
            except Exception as e:
                log.warning("fulltext_failed", paper_id=paper_id, error=str(e))
                return None
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

    async def _discover_neighbors(
        self,
        paper_id: str,
        paper: Paper | None = None,
        extracted: list[PaperSummary] | None = None,
    ) -> None:
        """Read a paper and record its neighbors in the agent's private graph.

        - Native refs (S2/OpenAlex/PubMed): use paper.references / paper.citations.
        - Full-text (arXiv): use the LLM-extracted references (outgoing only).
        Fetchable neighbors (arxiv_id/doi) enter the frontier; the rest are kept
        only in the private graph for structure. Neighbor summaries are cached in
        the shared store so embeddings/heuristics work.
        """
        if self.state.is_discovered(paper_id):
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
