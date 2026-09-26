"""SQLite-backed graph store: papers, edges, embeddings, and pheromone.

This is the shared memory of the colony — all agents read and write here.
Metadata is cached globally (not duplicated per agent), keeping agent state light.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research_explorer.graph.models import Paper, PaperSummary, normalize_id
from research_explorer.logging_setup import get_logger

log = get_logger("graph")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    doi TEXT,
    arxiv_id TEXT,
    title TEXT,
    year INTEGER,
    authors TEXT,
    citation_count INTEGER,
    abstract TEXT,
    embedding BLOB,
    fetched_at TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    PRIMARY KEY (src, dst)
);

CREATE TABLE IF NOT EXISTS pheromone (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    mode TEXT NOT NULL,
    tau REAL DEFAULT 1.0,
    PRIMARY KEY (src, dst, mode)
);

CREATE TABLE IF NOT EXISTS embeddings (
    text_hash TEXT PRIMARY KEY,
    embedding BLOB,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_pheromone_src ON pheromone(src);

CREATE TABLE IF NOT EXISTS paper_aliases (
    alias TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    provider TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_paper_aliases_canonical ON paper_aliases(canonical_id);

CREATE TABLE IF NOT EXISTS edge_provenance (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    direction TEXT NOT NULL,
    provider TEXT NOT NULL,
    created_at TEXT,
    PRIMARY KEY (src, dst, direction)
);
CREATE INDEX IF NOT EXISTS idx_edge_provenance_src ON edge_provenance(src);
CREATE INDEX IF NOT EXISTS idx_edge_provenance_dst ON edge_provenance(dst);
"""


def _pack_embedding(vec: list[float]) -> bytes:
    arr = np.array(vec, dtype=np.float32)
    return arr.tobytes()


def _unpack_embedding(blob: bytes) -> list[float]:
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr.tolist()


class GraphStore:
    """SQLite store for the explored citation subgraph, embeddings, and pheromone."""

    def __init__(self, db_path: str = "data/explorer.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Idempotent additive migrations; preserves pre-existing databases."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(papers)")}
        if "arxiv_id" not in cols:
            self._conn.execute("ALTER TABLE papers ADD COLUMN arxiv_id TEXT")
        if "integrated" not in cols:
            self._conn.execute(
                "ALTER TABLE papers ADD COLUMN integrated INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        self._conn.close()

    # ---- Papers ----------------------------------------------------------

    def cache_paper(self, paper: Paper) -> None:
        """Insert or replace a paper in the store.

        Re-caching provider metadata without an embedding must not erase an
        embedding already computed and stored for the node; an explicit
        non-null embedding may replace it (STAB-5).
        """
        nid = normalize_id(paper.provider, paper.id)
        existing = self._conn.execute(
            "SELECT integrated, embedding FROM papers WHERE id = ?", (nid,)
        ).fetchone()
        integrated = int(bool(existing and existing["integrated"]))
        stored_embedding = (
            existing["embedding"]
            if existing is not None and existing["embedding"] is not None
            else None
        )
        embedding_blob = (
            _pack_embedding(paper.embedding) if paper.embedding else stored_embedding
        )
        self._conn.execute(
            """INSERT OR REPLACE INTO papers
               (id, provider, doi, arxiv_id, title, year, authors, citation_count,
                abstract, embedding, fetched_at, metadata_json, integrated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                nid,
                paper.provider,
                paper.doi,
                paper.arxiv_id,
                paper.title,
                paper.year,
                json.dumps(paper.authors),
                paper.citation_count,
                paper.abstract,
                embedding_blob,
                None,
                json.dumps(
                    {
                        "external_ids": paper.external_ids,
                        "fields_of_study": paper.fields_of_study,
                        "tldr": paper.tldr,
                    }
                ),
                integrated,
            ),
        )
        # Store edges for references
        for ref in paper.references:
            ref_nid = normalize_id(ref.provider, ref.id)
            self.record_edge(nid, ref_nid, paper.provider, "references")
            # Also cache the reference summary if not present
            self._cache_summary_if_missing(ref)
        for cit in paper.citations:
            cit_nid = normalize_id(cit.provider, cit.id)
            self.record_edge(cit_nid, nid, paper.provider, "cited_by")
            self._cache_summary_if_missing(cit)
        self._conn.commit()

    def _cache_summary_if_missing(self, summary: PaperSummary) -> None:
        nid = normalize_id(summary.provider, summary.id)
        row = self._conn.execute("SELECT id FROM papers WHERE id = ?", (nid,)).fetchone()
        if row is None:
            self._insert_summary_row(nid, summary)

    def _insert_summary_row(self, nid: str, summary: PaperSummary) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO papers
               (id, provider, doi, arxiv_id, title, year, authors, citation_count,
                abstract, embedding, fetched_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                nid,
                summary.provider,
                summary.doi,
                summary.arxiv_id,
                summary.title,
                summary.year,
                json.dumps(summary.authors),
                summary.citation_count,
                summary.abstract,
                None,
                None,
                "{}",
            ),
        )

    def cache_summary(self, summary: PaperSummary) -> None:
        """Cache a paper summary (node metadata) if not already present."""
        self._cache_summary_if_missing(summary)
        self._conn.commit()

    def cache_summary_for_nid(self, nid: str, summary: PaperSummary) -> None:
        """Cache a summary under an exact canonical node id (insert-if-missing)."""
        row = self._conn.execute("SELECT id FROM papers WHERE id = ?", (nid,)).fetchone()
        if row is not None:
            return
        self._insert_summary_row(nid, summary)
        self._conn.commit()

    # ---- Aliases / canonical identity ------------------------------------

    def add_alias(self, alias: str, canonical_id: str, provider: str | None = None) -> None:
        """Map a provider-specific alias to a canonical node id (first mapping wins)."""
        if not alias or not canonical_id or alias == canonical_id:
            return
        existing = self.get_canonical_id(alias)
        if existing is not None:
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO paper_aliases (alias, canonical_id, provider, created_at)"
            " VALUES (?, ?, ?, ?)",
            (alias, canonical_id, provider, _utcnow()),
        )
        self._conn.commit()

    def get_canonical_id(self, alias: str) -> str | None:
        row = self._conn.execute(
            "SELECT canonical_id FROM paper_aliases WHERE alias = ?", (alias,)
        ).fetchone()
        return row["canonical_id"] if row else None

    def get_aliases(self, canonical_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT alias FROM paper_aliases WHERE canonical_id = ? ORDER BY alias",
            (canonical_id,),
        ).fetchall()
        return [r["alias"] for r in rows]

    def canonical_id(self, nid: str) -> str:
        """Resolve a node id through the alias table (identity if unmapped)."""
        current = nid
        seen: set[str] = set()
        while current not in seen:
            seen.add(current)
            nxt = self.get_canonical_id(current)
            if nxt is None or nxt == current:
                break
            current = nxt
        return current

    # ---- Edge provenance ---------------------------------------------------

    def record_edge(self, src: str, dst: str, provider: str, direction: str) -> None:
        """Insert an edge (src cites dst) with provider provenance.

        direction: 'references' (learned from src's bibliography) or
        'cited_by' (learned from dst's incoming-citation list).
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO edges (src, dst) VALUES (?, ?)", (src, dst)
        )
        self._conn.execute(
            """INSERT OR IGNORE INTO edge_provenance (src, dst, direction, provider, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (src, dst, direction, provider, _utcnow()),
        )

    def get_edge_provenance(self, src: str | None = None, dst: str | None = None) -> list[dict]:
        query = "SELECT src, dst, direction, provider, created_at FROM edge_provenance"
        conditions, params = [], []
        if src is not None:
            conditions.append("src = ?")
            params.append(src)
        if dst is not None:
            conditions.append("dst = ?")
            params.append(dst)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY src, dst, direction"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ---- Integration status ------------------------------------------------

    def mark_integrated(self, nid: str) -> None:
        """Flag a node whose readable content was actually integrated (evidence credit)."""
        self._conn.execute(
            "INSERT INTO papers (id, provider, integrated) VALUES (?, 'unknown', 1)"
            " ON CONFLICT(id) DO UPDATE SET integrated = 1",
            (nid,),
        )
        self._conn.commit()

    def is_integrated(self, nid: str) -> bool:
        row = self._conn.execute(
            "SELECT integrated FROM papers WHERE id = ?", (nid,)
        ).fetchone()
        return bool(row and row["integrated"])

    def get_paper(self, nid: str) -> Paper | None:
        """Retrieve a cached paper by normalized ID."""
        row = self._conn.execute("SELECT * FROM papers WHERE id = ?", (nid,)).fetchone()
        if row is None:
            return None
        return self._row_to_paper(row)

    def get_paper_summary(self, nid: str) -> PaperSummary | None:
        """Retrieve a cached paper summary by normalized ID."""
        row = self._conn.execute("SELECT * FROM papers WHERE id = ?", (nid,)).fetchone()
        if row is None:
            return None
        return self._row_to_summary(row)

    def get_all_paper_summaries(self) -> list[PaperSummary]:
        """Retrieve all cached paper summaries (every node in the shared store)."""
        rows = self._conn.execute("SELECT * FROM papers ORDER BY title").fetchall()
        return [self._row_to_summary(r) for r in rows]

    def has_paper(self, nid: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM papers WHERE id = ? AND abstract IS NOT NULL", (nid,)
        ).fetchone()
        return row is not None

    def _row_to_summary(self, row: sqlite3.Row) -> PaperSummary:
        return PaperSummary(
            id=row["id"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            title=row["title"] or "",
            year=row["year"],
            authors=json.loads(row["authors"]) if row["authors"] else [],
            citation_count=row["citation_count"],
            abstract=row["abstract"],
            provider=row["provider"],
        )

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        embedding = _unpack_embedding(row["embedding"]) if row["embedding"] else None
        return Paper(
            id=row["id"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            title=row["title"] or "",
            year=row["year"],
            authors=json.loads(row["authors"]) if row["authors"] else [],
            citation_count=row["citation_count"],
            abstract=row["abstract"],
            provider=row["provider"],
            external_ids=meta.get("external_ids", {}),
            fields_of_study=meta.get("fields_of_study", []),
            tldr=meta.get("tldr"),
            embedding=embedding,
        )

    # ---- Edges / Neighbors -----------------------------------------------

    def get_references(self, nid: str) -> list[str]:
        """Get normalized IDs of papers referenced by nid (outgoing)."""
        rows = self._conn.execute(
            "SELECT dst FROM edges WHERE src = ?", (nid,)
        ).fetchall()
        return [r["dst"] for r in rows]

    def get_citants(self, nid: str) -> list[str]:
        """Get normalized IDs of papers that cite nid (incoming)."""
        rows = self._conn.execute(
            "SELECT src FROM edges WHERE dst = ?", (nid,)
        ).fetchall()
        return [r["src"] for r in rows]

    def get_neighbors(self, nid: str) -> tuple[list[str], list[str]]:
        """Return (references, citants) as normalized IDs."""
        return self.get_references(nid), self.get_citants(nid)

    # ---- Embeddings ------------------------------------------------------

    def get_cached_embedding(self, text: str) -> list[float] | None:
        """Get a cached embedding for a text string (by hash)."""
        import hashlib

        h = hashlib.sha256(text.encode()).hexdigest()
        row = self._conn.execute(
            "SELECT embedding FROM embeddings WHERE text_hash = ?", (h,)
        ).fetchone()
        if row is None:
            return None
        return _unpack_embedding(row["embedding"])

    def cache_embedding(self, text: str, embedding: list[float]) -> None:
        import hashlib

        h = hashlib.sha256(text.encode()).hexdigest()
        self._conn.execute(
            """INSERT OR REPLACE INTO embeddings (text_hash, embedding, created_at)
               VALUES (?, ?, ?)""",
            (h, _pack_embedding(embedding), _utcnow()),
        )
        self._conn.commit()

    def set_paper_embedding(self, nid: str, embedding: list[float]) -> None:
        self._conn.execute(
            "UPDATE papers SET embedding = ? WHERE id = ?",
            (_pack_embedding(embedding), nid),
        )
        self._conn.commit()

    def get_paper_embedding(self, nid: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT embedding FROM papers WHERE id = ?", (nid,)
        ).fetchone()
        if row is None or row["embedding"] is None:
            return None
        return _unpack_embedding(row["embedding"])

    # ---- Pheromone -------------------------------------------------------

    def get_pheromone(self, src: str, dst: str, mode: str) -> float:
        row = self._conn.execute(
            "SELECT tau FROM pheromone WHERE src = ? AND dst = ? AND mode = ?",
            (src, dst, mode),
        ).fetchone()
        if row is None:
            return 1.0  # default tau_init
        return row["tau"]

    def set_pheromone(self, src: str, dst: str, mode: str, tau: float) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO pheromone (src, dst, mode, tau)
               VALUES (?, ?, ?, ?)""",
            (src, dst, mode, tau),
        )

    def evaporate_all(self, rho: float, tau_min: float = 0.1) -> None:
        """Evaporate all pheromone: tau <- (1-rho)*tau, clipped to tau_min."""
        self._conn.execute(
            "UPDATE pheromone SET tau = MAX(?, (1 - ?) * tau)",
            (tau_min, rho),
        )
        self._conn.commit()

    def clip_pheromone(self, tau_min: float, tau_max: float) -> None:
        """Clip all pheromone values to [tau_min, tau_max] (MMAS bounds)."""
        self._conn.execute(
            "UPDATE pheromone SET tau = MIN(?, MAX(?, tau))",
            (tau_max, tau_min),
        )
        self._conn.commit()

    def pheromone_concentration(self) -> float:
        """Ratio of max tau to mean tau — high values indicate convergence."""
        row = self._conn.execute(
            "SELECT MAX(tau) as mx, AVG(tau) as avg FROM pheromone"
        ).fetchone()
        if row is None or row["avg"] is None or row["avg"] == 0:
            return 0.0
        return row["mx"] / row["avg"]

    def commit(self) -> None:
        self._conn.commit()
