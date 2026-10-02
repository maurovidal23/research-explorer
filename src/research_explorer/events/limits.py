"""Centralized memory bounds and cadences for the live run projection.

SQLite remains the authoritative, complete history. These limits bound only the
in-memory projection windows, the number of rows the Events tab renders per
page, the candidate alternatives shown per decision, and the live render
cadence, so a multi-hour run stays within a predictable memory envelope. Live
and replay projection share these defaults so equivalent inputs compact
deterministically.
"""

LIVE_EVENT_WINDOW = 1000
LIVE_CANDIDATE_WINDOW = 1000
EVENT_PAGE_SIZE = 200
CANDIDATE_TOP_ALTERNATIVES = 12
TUI_REFRESH_INTERVAL_SECONDS = 0.1
