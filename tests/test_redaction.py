"""STAB-7: secret redaction in logs, exceptions, events, and artifacts."""

from __future__ import annotations

import tempfile

from research_explorer.logging_setup import _redact_processor
from research_explorer.redaction import REDACTED, redact_obj, redact_secrets
from research_explorer.replay.trace import RunTraceStore

SENTINEL = "SENTINEL-SECRET-9f8e7d"


def test_redacts_api_key_query_param() -> None:
    url = f"https://api.example.test/v1/paper?api_key={SENTINEL}&limit=5"
    redacted = redact_secrets(url)
    assert SENTINEL not in redacted
    assert "limit=5" in redacted
    assert REDACTED in redacted


def test_redacts_key_token_and_email_params() -> None:
    text = f"key={SENTINEL} token={SENTINEL} email={SENTINEL}@example.com"
    redacted = redact_secrets(text)
    assert SENTINEL not in redacted


def test_redacts_dict_style_secrets() -> None:
    text = f"{{'x-api-key': '{SENTINEL}', 'token': '{SENTINEL}'}}"
    redacted = redact_secrets(text)
    assert SENTINEL not in redacted


def test_redact_obj_recurses_into_lists_and_dicts() -> None:
    payload = {
        "url": f"https://x.test?token={SENTINEL}",
        "nested": [f"email={SENTINEL}@a.com", {"key": SENTINEL}],
    }
    redacted = redact_obj(payload)
    assert SENTINEL not in str(redacted)


def test_redacts_authorization_bearer_credential() -> None:
    for text in (
        f"Authorization: Bearer {SENTINEL}",
        f"headers={{'authorization': 'Basic {SENTINEL}'}}",
    ):
        redacted = redact_secrets(text)
        assert SENTINEL not in redacted
        assert REDACTED in redacted


def test_logging_processor_redacts_event_values() -> None:
    event = {"event": "http.request", "url": f"https://x.test?api_key={SENTINEL}"}
    processed = _redact_processor(None, "info", event)
    assert SENTINEL not in processed["url"]
    assert REDACTED in processed["url"]


def test_logging_processor_redacts_nested_event_values() -> None:
    event = {
        "event": "provider_failure",
        "context": {"url": f"https://x.test?token={SENTINEL}"},
        "items": [{"api_key": SENTINEL}, f"authorization: Bearer {SENTINEL}"],
    }
    processed = _redact_processor(None, "warning", event)
    assert SENTINEL not in str(processed)


def test_trace_events_and_artifacts_never_contain_secret() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = RunTraceStore(f"{tmp}/replay.db")
        try:
            run_id = store.create_run("10.1/x", "q")
            store.append_event(
                run_id,
                "provider_failure",
                {"error": f"429 https://api.x.test?api_key={SENTINEL}"},
            )
            store.save_artifact(
                run_id, "err.txt", "error", f"boom token={SENTINEL}"
            )
            too = store.append_event(run_id, "seed_routed", {"api_key": SENTINEL})
            assert too == 2
            events = store.list_events(run_id)
            for event in events:
                assert SENTINEL not in str(event["payload"])
            artifacts = store.list_artifacts(run_id)
            assert artifacts
            for meta in artifacts:
                content = store.get_artifact(meta["artifact_id"])["content"]
                assert SENTINEL not in content
        finally:
            store.close()
