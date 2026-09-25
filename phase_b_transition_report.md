 config/default.toml                         |  21 +-
 config/profiles/experiment.toml             |   3 -
 config/profiles/nan_free.toml               |   3 -
 config/profiles/nan_full.toml               |   3 -
 config/profiles/quick.toml                  |   3 -
 docs/model.md                               | 177 +++++++-----
 opencode.json                               |  16 ++
 src/research_explorer/aco/colony.py         |  13 +
 src/research_explorer/aco/frontier.py       |  59 +++-
 src/research_explorer/aco/scheduler.py      |   2 +-
 src/research_explorer/agents/explorer.py    | 406 +++++++++++++++++++++++++---
 src/research_explorer/agents/state.py       |  22 ++
 src/research_explorer/config.py             |  46 +++-
 src/research_explorer/graph/models.py       |   1 +
 src/research_explorer/graph/store.py        | 203 +++++++++++---
 src/research_explorer/providers/openalex.py |  20 ++
 src/research_explorer/replay/__init__.py    |   8 +-
 src/research_explorer/replay/models.py      |  60 ++++
 src/research_explorer/replay/trace.py       |  12 +-
 tests/agents/test_explorer.py               | 136 +++++++++-
 tests/graph/test_store.py                   | 211 +++++++++++++++
 21 files changed, 1245 insertions(+), 180 deletions(-)

EVidence:
- Full tests: 305 passed (was 264 baseline) + 1 warning
- Focused Phase A (tests/resolution + tests/agents/test_explorer + tests/graph/test_store): 144 passed (unchanged)
- New tests: tests/aco/test_transition.py (23), test_frontier.py (9), test_selection_integration.py (11)
- Ruff src/ tests/: All checks passed
- mypy src/: blocked by known numpy env incompatibility (numpy __init__.pyi uses 3.12 'type' syntax; pyproject sets 3.10) - unrelated pre-existing; my changed files pass under --follow-imports=skip --ignore-missing-imports --python-version 3.12
