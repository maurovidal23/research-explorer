"""UI-independent run event contract, publishing sinks, and projection.

This package deliberately has no Textual (or other UI) imports: orchestration,
replay, and the TUI all depend on these typed events. The Textual application
lives in :mod:`research_explorer.tui` and only consumes projected state.
"""

from research_explorer.events.models import (
    AgentSummary,
    EventType,
    RunEvent,
    RunViewState,
    TimelineEntry,
)
from research_explorer.events.projection import RunProjection
from research_explorer.events.sink import (
    CallbackSink,
    ChannelSink,
    CompositeSink,
    EventSink,
    NullSink,
)

__all__ = [
    "AgentSummary",
    "CallbackSink",
    "ChannelSink",
    "CompositeSink",
    "EventSink",
    "EventType",
    "NullSink",
    "RunEvent",
    "RunProjection",
    "RunViewState",
    "TimelineEntry",
]
