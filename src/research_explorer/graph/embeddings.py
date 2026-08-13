"""Embedding service — wraps an embedder with SQLite caching.

The embedder is pluggable: by default it uses the NaN qwen3-embedding endpoint
(via LLMClient), but any callable (texts) -> list[list[float]] works.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from research_explorer.graph.store import GraphStore

EmbedderFn = Callable[[list[str]], Awaitable[list[list[float]]]]


class EmbeddingService:
    """Caching wrapper around an embedding function."""

    def __init__(self, store: GraphStore, embedder: EmbedderFn | None = None):
        self.store = store
        self._embedder = embedder

    async def embed(self, text: str) -> list[float]:
        """Get embedding for a single text, using cache when available."""
        cached = self.store.get_cached_embedding(text)
        if cached is not None:
            return cached
        if self._embedder is None:
            raise RuntimeError("No embedder configured")
        vecs = await self._embedder([text])
        self.store.cache_embedding(text, vecs[0])
        return vecs[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed multiple texts, using cache for hits and batching misses.

        Returns a list of the same length as texts, with None for failures.
        """
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[str] = []
        miss_indices: list[int] = []
        for i, t in enumerate(texts):
            cached = self.store.get_cached_embedding(t)
            if cached is not None:
                results[i] = cached
            else:
                misses.append(t)
                miss_indices.append(i)
        if misses and self._embedder is not None:
            try:
                vecs = await self._embedder(misses)
                for idx, text, vec in zip(miss_indices, misses, vecs, strict=False):
                    results[idx] = vec
                    self.store.cache_embedding(text, vec)
            except Exception:
                pass
        return results
