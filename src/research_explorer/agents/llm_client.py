"""LLM client for NaN (OpenAI-compatible) with concurrency and rate limiting.

All agents and evaluators share a single LLMClient instance, which enforces:
  - asyncio.Semaphore(max_concurrent) — caps in-flight LLM requests
  - aiolimiter(rpm/60, 1) — paces requests per second
  - tenacity retry with exponential backoff

NaN base URL: https://api.nan.builders/v1
NaN limits: 60 rpm, 5 concurrent, 1.5M tpm per model.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from research_explorer.logging_setup import get_logger

log = get_logger("llm")


class LLMClient:
    """Async LLM client with concurrency control, rate limiting, and retry.

    Wraps the OpenAI-compatible NaN API. Also provides embedding and rerank
    endpoints (qwen3-embedding and rerank models).
    """

    def __init__(
        self,
        base_url: str = "https://api.nan.builders/v1",
        api_key: str | None = None,
        max_concurrent: int = 5,
        rpm: int = 60,
    ):
        resolved_key = api_key or os.environ.get("NAN_API_KEY", "")
        if not resolved_key:
            raise RuntimeError(
                "NAN_API_KEY not found. Set it as an environment variable or "
                "create a .env file with NAN_API_KEY=sk-... in the project root."
            )
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=resolved_key,
        )
        self.sem = asyncio.Semaphore(max_concurrent)
        self.limiter = AsyncLimiter(max(rpm / 60, 0.1), 1)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type(Exception),
    )
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "qwen3.6",
        response_format: dict | None = None,
        temperature: float = 0.6,
        max_tokens: int = 2000,
        extra_body: dict | None = None,
    ) -> str:
        """Send a chat completion request. Returns the assistant message content."""
        async with self.sem, self.limiter:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = await self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "qwen3.6",
        schema: dict | None = None,
        temperature: float = 0.6,
        max_tokens: int = 2000,
    ) -> dict:
        """Chat with structured JSON output (json_schema strict)."""
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            }
        else:
            response_format = {"type": "json_object"}
        content = await self.chat(
            messages,
            model=model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        import json

        return json.loads(content)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=30),
    )
    async def embed(self, texts: list[str], model: str = "qwen3-embedding") -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        async with self.sem, self.limiter:
            resp = await self.client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]

    async def embed_one(self, text: str, model: str = "qwen3-embedding") -> list[float]:
        """Embed a single text."""
        vecs = await self.embed([text], model=model)
        return vecs[0]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=30),
    )
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        model: str = "rerank",
    ) -> list[dict]:
        """Rerank documents by relevance to a query (Qwen3-Reranker).

        Returns list of {index, relevance_score, document} sorted by score desc.
        """
        async with self.sem, self.limiter:
            resp = await self.client.post(
                "/rerank",
                body={
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
                cast_to=dict,
            )
            return resp.get("results", [])

    async def aclose(self) -> None:
        await self.client.close()
