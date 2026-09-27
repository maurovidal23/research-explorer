"""TUI controller: bundles the projection and the bounded live event channel."""

from __future__ import annotations

import asyncio

from research_explorer.events.models import RunEvent
from research_explorer.events.projection import RunProjection
from research_explorer.events.sink import ChannelSink, QueueMarker


class TUIController:
    """Owns the shared projection and the TUI's bounded event channel."""

    def __init__(self, maxsize: int = 2048) -> None:
        self.projection = RunProjection()
        self.channel = ChannelSink(maxsize=maxsize)

    @property
    def queue(self) -> asyncio.Queue[RunEvent | QueueMarker]:
        return self.channel.queue

    @property
    def event_sink(self) -> ChannelSink:
        return self.channel

    def drain(self) -> None:
        """Apply any queued events synchronously (used after shutdown/tests)."""
        while not self.channel.empty():
            item = self.channel.get_nowait()
            if isinstance(item, QueueMarker):
                item.resolve()
                continue
            self.projection.apply(item)
