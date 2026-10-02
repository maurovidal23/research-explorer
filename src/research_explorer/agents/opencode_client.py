"""Structured LLM access through an authenticated OpenCode subscription."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from research_explorer.redaction import redact_secrets


def _parse_stream(output: str) -> tuple[str, dict[str, Any]]:
    texts: list[str] = []
    usage: dict[str, Any] = {"finish_reason": "unknown"}
    for line in output.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        part = event.get("part", {})
        if event.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
        if event.get("type") == "step_finish":
            usage["finish_reason"] = str(part.get("reason") or "unknown")
            tokens = part.get("tokens", {})
            for source, target in (
                ("input", "prompt_tokens"),
                ("output", "completion_tokens"),
                ("total", "total_tokens"),
            ):
                value = tokens.get(source)
                if isinstance(value, int):
                    usage[target] = value
    content = "".join(texts).strip()
    usage["response_chars"] = len(content)
    return content, usage


class OpenCodeLLMClient:
    """Async JSON client backed by the local OpenCode CLI."""

    def __init__(self, executable: str = "opencode") -> None:
        self.executable = executable
        self.last_usage: dict[str, Any] | None = None
        self.tracer: Any | None = None

    def _emit_operation(self, event_type: str, **payload: Any) -> None:
        if self.tracer is not None:
            self.tracer.emit(event_type, **payload)

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        purpose: str = "chat_json",
        attempts: int = 2,
        extra_body: dict | None = None,
    ) -> dict:
        del temperature
        model_id = model if "/" in model else f"openai/{model}"
        effort = str((extra_body or {}).get("reasoning_effort") or "medium")
        prompt = self._prompt(messages, schema, max_tokens)
        last_error: Exception | None = None
        for attempt in range(max(1, attempts)):
            started = time.monotonic()
            self._emit_operation("llm_operation_started", purpose=purpose, model=model_id)
            try:
                content, usage = await self._run(prompt, model_id, effort)
                payload = json.loads(content)
                if not isinstance(payload, dict):
                    raise ValueError("OpenCode response must be a JSON object")
            except Exception as exc:
                last_error = exc
                self._emit_operation(
                    "llm_operation_failed",
                    purpose=purpose,
                    model=model_id,
                    attempt=attempt + 1,
                    elapsed=round(time.monotonic() - started, 4),
                    error=redact_secrets(str(exc)),
                )
                continue
            self.last_usage = usage
            self._emit_operation(
                "llm_operation_completed",
                purpose=purpose,
                model=model_id,
                elapsed=round(time.monotonic() - started, 4),
                **usage,
            )
            return payload
        if last_error is None:
            raise RuntimeError("OpenCode structured output failed without an error")
        raise last_error

    def _prompt(
        self,
        messages: list[dict[str, str]],
        schema: dict | None,
        max_tokens: int | None,
    ) -> str:
        sections = [
            "Do not use tools. Return only one valid JSON object with no markdown fences.",
        ]
        if schema is not None:
            sections.append(f"The JSON must conform to this schema:\n{json.dumps(schema)}")
        if max_tokens:
            sections.append(f"Keep the response within {max_tokens} output tokens.")
        sections.extend(
            f"{message['role'].upper()}:\n{message['content']}" for message in messages
        )
        return "\n\n".join(sections)

    async def _run(
        self, prompt: str, model: str, reasoning_effort: str
    ) -> tuple[str, dict[str, Any]]:
        process = await asyncio.create_subprocess_exec(
            self.executable,
            "run",
            "--pure",
            "--model",
            model,
            "--variant",
            reasoning_effort,
            "--format",
            "json",
            cwd=Path(tempfile.gettempdir()),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(prompt.encode("utf-8"))
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            if not detail:
                detail = stdout.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"OpenCode exited {process.returncode}: {detail}")
        content, usage = _parse_stream(stdout.decode("utf-8", errors="replace"))
        if not content:
            raise RuntimeError("OpenCode returned no final text")
        return content, usage

    async def aclose(self) -> None:
        return None
