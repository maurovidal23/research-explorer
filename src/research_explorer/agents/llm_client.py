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
import json
import os
import time
from typing import TYPE_CHECKING, Any

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from research_explorer.logging_setup import get_logger
from research_explorer.redaction import redact_secrets

if TYPE_CHECKING:
    from research_explorer.replay.trace import RunTracer

log = get_logger("llm")


class EmptyCompletionError(ValueError):
    pass


class TruncatedCompletionError(ValueError):
    pass


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
        self.tracer: RunTracer | None = None

    def _emit_operation(self, type: str, **payload: Any) -> None:
        tracer = self.tracer
        if tracer is None:
            return
        tracer.emit(type, **payload)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "qwen3.6",
        response_format: dict | None = None,
        temperature: float = 0.6,
        max_tokens: int | None = 2000,
        extra_body: dict | None = None,
        purpose: str = "chat",
    ) -> str:
        """Send a chat completion request. Returns the assistant message content.

        Emits ``llm_operation_started``/``completed``/``failed`` telemetry when a
        :class:`~research_explorer.replay.trace.RunTracer` is attached; the
        payload carries purpose, model, elapsed time, and token usage only.
        """
        started = time.monotonic()
        self._emit_operation("llm_operation_started", purpose=purpose, model=model)
        try:
            content, usage = await self._chat_impl(
                messages,
                model=model,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        except Exception as exc:
            self._emit_operation(
                "llm_operation_failed",
                purpose=purpose,
                model=model,
                elapsed=round(time.monotonic() - started, 4),
                error=redact_secrets(str(exc)),
            )
            raise
        self._emit_operation(
            "llm_operation_completed",
            purpose=purpose,
            model=model,
            elapsed=round(time.monotonic() - started, 4),
            **usage,
        )
        return content

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type(Exception),
    )
    async def _chat_impl(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        response_format: dict | None,
        temperature: float,
        max_tokens: int | None,
        extra_body: dict | None,
    ) -> tuple[str, dict[str, Any]]:
        async with self.sem, self.limiter:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None and max_tokens > 0:
                kwargs["max_tokens"] = max_tokens
            if response_format:
                kwargs["response_format"] = response_format
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = await self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content or ""
            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            if finish_reason == "length":
                raise TruncatedCompletionError(
                    f"completion truncated at configured token limit; response_chars={len(content)}"
                )
            if not content.strip():
                raise EmptyCompletionError(
                    f"completion returned no visible content; finish_reason={finish_reason or 'unknown'}"
                )
            usage: dict[str, Any] = {
                "finish_reason": finish_reason or "unknown",
                "response_chars": len(content),
            }
            reported = getattr(resp, "usage", None)
            if reported is not None:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = getattr(reported, key, None)
                    if value is not None:
                        usage[key] = int(value)
            if "total_tokens" not in usage:
                total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                if total:
                    usage["total_tokens"] = total
            return content, usage

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str = "qwen3.6",
        schema: dict | None = None,
        temperature: float = 0.6,
        max_tokens: int | None = 2000,
        purpose: str = "chat_json",
        attempts: int = 2,
    ) -> dict:
        """Chat with structured JSON output (json_schema strict)."""
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            }
        else:
            response_format = {"type": "json_object"}
        last_error: json.JSONDecodeError | None = None
        for attempt in range(max(1, attempts)):
            content = await self.chat(
                messages,
                model=model,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                purpose=purpose,
            )
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                self._emit_operation(
                    "llm_structured_output_invalid",
                    purpose=purpose,
                    model=model,
                    attempt=attempt + 1,
                    response_chars=len(content),
                    error=redact_secrets(str(exc)),
                )
        if last_error is None:
            raise RuntimeError("structured output failed without an error")
        raise last_error

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
