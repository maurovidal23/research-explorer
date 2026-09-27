"""UI-independent event publishing contracts.

Orchestration publishes through :class:`EventSink`. A :class:`CompositeSink`
fans out to the durable trace store and a bounded, async-safe in-memory channel.
The publish boundary is synchronous and must never block on terminal rendering;
the channel drops oldest entries on overflow (the durable store remains
authoritative) rather than stalling research.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

from research_explorer.events.models import RunEvent


@runtime_checkable
class EventSink(Protocol):
    """Publishes normalized run events to one consumer."""

    def publish(self, event: RunEvent) -> None: ...


class NullSink:
    """Sink that discards every event (non-TUI default)."""

    def publish(self, event: RunEvent) -> None:
        return None


class CallbackSink:
    """Sink that forwards events to a plain synchronous callback."""

    def __init__(self, callback: Callable[[RunEvent], None]) -> None:
        self._callback = callback

    def publish(self, event: RunEvent) -> None:
        self._callback(event)


class ChannelSink:
    """Bounded, async-safe in-memory channel consumed by the TUI.

    ``publish`` never blocks: when the queue is full the oldest event is dropped
    and ``dropped`` is incremented. Durable semantic events are still written by
    the store; display-only gaps can be reconciled from the durable store.
    """

    def __init__(self, maxsize: int = 1024, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=maxsize)
        self._loop = loop
        self.dropped = 0

    @property
    def queue(self) -> asyncio.Queue[RunEvent]:
        return self._queue

    def publish(self, event: RunEvent) -> None:
        self._put(event)

    def _put(self, event: RunEvent) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._put_nowait, event)
                return
            except RuntimeError:
                pass
        self._put_nowait(event)

    def _put_nowait(self, event: RunEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def get(self) -> RunEvent:
        return await self._queue.get()

    def get_nowait(self) -> RunEvent:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()


class CompositeSink:
    """Fan out each event to every child sink in order."""

    def __init__(self, sinks: Iterable[EventSink]) -> None:
        self._sinks = list(sinks)

    def add(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def publish(self, event: RunEvent) -> None:
        for sink in self._sinks:
            sink.publish(event)
