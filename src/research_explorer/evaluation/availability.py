"""Explicit component-availability reasons for the quality function.

A component may be unavailable for a stable, replayable reason (empty input,
model failure, parse failure, timeout, or missing peers). These reasons keep an
unavailable component distinguishable from a genuine numeric zero.
"""

from __future__ import annotations

import asyncio
import json

REASON_EMPTY_NARRATIVE = "empty_narrative"
REASON_MODEL_FAILED = "model_failed"
REASON_PARSE_FAILED = "parse_failed"
REASON_TIMEOUT = "timeout"
REASON_NO_PEERS = "no_peers"
REASON_NOT_CONFIGURED = "not_configured"


def classify_failure(exc: BaseException) -> str:
    """Map a raised component exception to a stable availability reason."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return REASON_TIMEOUT
    if isinstance(exc, (json.JSONDecodeError, ValueError, TypeError, KeyError)):
        return REASON_PARSE_FAILED
    return REASON_MODEL_FAILED
