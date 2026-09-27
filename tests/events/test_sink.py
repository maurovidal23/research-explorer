"""Event sink behavior: bounded channel, composite fan-out, null sink."""

from __future__ import annotations

import asyncio

from research_explorer.events.models import RunEvent
from research_explorer.events.sink import (
    CallbackSink,
    ChannelSink,
    CompositeSink,
    NullSink,
)


def _event(seq: int) -> RunEvent:
    return RunEvent(seq=seq, type="wave_started", payload={"oleada": seq})


def test_channel_sink_is_bounded_and_never_blocks() -> None:
    sink = ChannelSink(maxsize=2)
    for seq in range(1, 6):
        sink.publish(_event(seq))
    assert sink.queue.qsize() <= 2
    assert sink.dropped >= 3
    assert [sink.get_nowait().seq, sink.get_nowait().seq] == [4, 5]


def test_channel_sink_preserves_order_under_capacity() -> None:
    sink = ChannelSink(maxsize=10)
    for seq in range(1, 4):
        sink.publish(_event(seq))
    assert [sink.get_nowait().seq for _ in range(3)] == [1, 2, 3]


async def test_channel_sink_overflow_keeps_join_consistent() -> None:
    """Evicted events are acknowledged so ``queue.join()`` cannot hang.

    The runner awaits ``queue.join()`` as its terminal-state barrier, so an
    overflow must leave the unfinished-task count consistent with what the
    consumer actually dequeues.
    """
    sink = ChannelSink(maxsize=2)
    for seq in range(1, 6):
        sink.publish(_event(seq))
    assert sink.dropped == 3

    while not sink.empty():
        sink.get_nowait()
        sink.queue.task_done()
    await asyncio.wait_for(sink.queue.join(), timeout=1.0)


def test_composite_sink_fans_out() -> None:
    first: list[RunEvent] = []
    second: list[RunEvent] = []
    composite = CompositeSink([CallbackSink(first.append), CallbackSink(second.append)])
    composite.publish(_event(1))
    assert first == second
    assert first[0].seq == 1


def test_null_sink_discards() -> None:
    NullSink().publish(_event(1))
