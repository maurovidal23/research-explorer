"""Secret redaction for logs, exceptions, trace events, and artifacts.

Credentials can leak into the system in several places: URL query strings
carried by HTTP exceptions, structlog event dictionaries, and trace payloads.
Every one of those paths funnels through :func:`redact_secrets` (strings) or
:func:`redact_obj` (recursive JSON-like structures).
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "***"

_NAME = (
    r"(?:x-api-key|api[_-]?key|apikey|access[_-]?token|auth[_-]?token|"
    r"authorization|token|email|password|secret|key)"
)
_NAME_RE = re.compile(rf"(?i)^{_NAME}$")

_SECRET_RE = re.compile(
    rf"(?i)(?P<pre>[?&;\"'\s=:,{{]|^)(?P<name>{_NAME})(?P<mid>\s*[\"']?\s*[:=]\s*[\"']?)"
    rf"(?P<val>[^&\s,;\"'}}\]]+)"
)


def _replace(match: re.Match[str]) -> str:
    return f"{match.group('pre')}{match.group('name')}{match.group('mid')}{REDACTED}"


def redact_secrets(text: str) -> str:
    """Redact values that follow sensitive parameter names in arbitrary text."""
    if not text:
        return text
    return _SECRET_RE.sub(_replace, text)


def redact_obj(value: Any) -> Any:
    """Recursively redact every string in a JSON-like structure."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _NAME_RE.match(key.strip()):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_obj(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_obj(v) for v in value]
    return value
