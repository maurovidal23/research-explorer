"""Subprocess memory regression for the high-volume live projection path.

The child applies a frozen synthetic incident workload (large candidate-score
bursts with mixed telemetry) through the same :class:`RunProjection` the live
TUI consumes, and reports peak RSS. The ceiling is deliberately generous and
machine-independent: the assertion that matters is that retained windows stay
at their configured bounds while the total-seen counters stay exact.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from research_explorer.events.limits import LIVE_CANDIDATE_WINDOW, LIVE_EVENT_WINDOW

_CHILD = r"""
import json
import resource
from research_explorer.events.models import RunEvent
from research_explorer.events.projection import RunProjection

projection = RunProjection()
total = 50000
for seq in range(1, total + 1):
    if seq % 2:
        event = RunEvent(
            seq=seq,
            type="candidate_score",
            payload={
                "paper_id": "paper:%d" % seq,
                "agent_id": "a%d" % (seq % 15),
                "eta": 0.5,
                "components": {"sim": 0.5, "citations": 0.4},
                "mode": "ref",
            },
        )
    else:
        event = RunEvent(
            seq=seq,
            type="paper_integration_completed",
            payload={"agent_id": "a%d" % (seq % 15), "paper_id": "paper:%d" % seq},
        )
    projection.apply(event)
state = projection.state
print(json.dumps({
    "rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "events": len(state.events),
    "scores": len(state.candidate_scores),
    "events_seen": state.events_seen_total,
    "scores_seen": state.candidate_scores_seen_total,
}))
"""


def test_high_volume_projection_stays_within_bounds_and_rss_ceiling() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
        timeout=180,
        check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["events"] == LIVE_EVENT_WINDOW
    assert payload["scores"] == LIVE_CANDIDATE_WINDOW
    assert payload["events_seen"] == 50_000
    assert payload["scores_seen"] == 25_000
    assert payload["rss_kb"] < 400_000
