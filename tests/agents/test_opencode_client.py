from __future__ import annotations

import json

from research_explorer.agents.opencode_client import OpenCodeLLMClient, _parse_stream


def test_parse_stream_extracts_final_text_and_usage() -> None:
    output = "\n".join(
        [
            json.dumps({"type": "text", "part": {"text": '{"ok":true}'}}),
            json.dumps(
                {
                    "type": "step_finish",
                    "part": {
                        "reason": "stop",
                        "tokens": {"input": 10, "output": 4, "total": 14},
                    },
                }
            ),
        ]
    )

    content, usage = _parse_stream(output)

    assert content == '{"ok":true}'
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 4
    assert usage["total_tokens"] == 14


def test_prompt_keeps_schema_and_messages_out_of_process_arguments() -> None:
    client = OpenCodeLLMClient()

    prompt = client._prompt(
        [{"role": "user", "content": "private evidence"}],
        {"type": "object"},
        100,
    )

    assert "private evidence" in prompt
    assert '"type": "object"' in prompt
    assert "100 output tokens" in prompt
