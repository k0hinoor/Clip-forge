"""In-process event bus for live UI updates (Server-Sent Events).

The worker publishes progress; the API streams it to the browser. A small ring
buffer keeps recent events so a page refresh (or an SSE reconnect) does not lose
the current state.

Two kinds of subscribers are supported:

* :meth:`EventBus.subscribe` returns a thread-safe :class:`queue.Queue` for
  synchronous consumers (tests, scripts, worker processes);
* :meth:`EventBus.subscribe_async` returns an :class:`AsyncSubscription` bound to
  the running event loop. The SSE endpoint uses it, so a connected browser never
  occupies a worker thread and a disconnect is just a cancelled ``await``.

Publishing never blocks: a slow or vanished subscriber only loses its own events.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

MAX_BUFFER = 200
REPLAY_EVENTS = 40
MAILBOX_SIZE = 1000


@dataclass
class Event:
    type: str
    data: dict[str, Any]
    project_id: str = ""
    job_id: str = ""
    at: float = field(default_factory=time.time)

    def to_payload(self) -> dict[str, Any]:
        payload = {"type": self.type, "at": self.at, **self.data}
        if self.project_id:
            payload.setdefault("project_id", self.project_id)
        if self.job_id:
            payload.setdefault("job_id", self.job_id)
        return payload


@dataclass(eq=False)
class _Subscriber:
    scope: str
    deliver: Callable[[Event], None]
    handle: object

    def wants(self, event: Event) -> bool:
        return not self.scope or not event.project_id or event.project_id == self.scope


class AsyncSubscription:
    """An asyncio mailbox fed by :meth:`EventBus.publish` from any thread."""

    def __init__(self, bus: "EventBus", project_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self._bus = bus
        self._loop = loop
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=MAILBOX_SIZE)
        self.project_id = project_id
        self.closed = False

    # Called on the publishing thread.
    def _deliver(self, event: Event) -> None:
        if self.closed:
            return
        try:
            self._loop.call_soon_threadsafe(self._put, event)
        except RuntimeError:  # the loop is closed: the client is long gone
            self.closed = True

    # Runs on the event loop.
    def _put(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # a stalled client must never block the pipeline

    async def next(self, timeout: float | None = None) -> Event | None:
        """The next event, or ``None`` when ``timeout`` seconds pass without one."""
        try:
            if timeout is None:
                return await self._queue.get()
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    def close(self) -> None:
        if not self.closed:
            self.closed = True
        self._bus._unregister(self)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[_Subscriber] = []
        self._lock = threading.Lock()
        self._buffer: deque[Event] = deque(maxlen=MAX_BUFFER)

    # -------------------------------------------------------------- publish
    def publish(self, event_type: str, data: dict[str, Any], *, project_id: str = "", job_id: str = "") -> Event:
        event = Event(type=event_type, data=data, project_id=project_id, job_id=job_id)
        with self._lock:
            self._buffer.append(event)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            if not subscriber.wants(event):
                continue
            try:
                subscriber.deliver(event)
            except Exception:  # noqa: BLE001 - one broken subscriber must not affect the others
                continue
        return event

    def recent(self, project_id: str = "", limit: int = REPLAY_EVENTS) -> list[Event]:
        """The latest buffered events visible to a subscriber of ``project_id``."""
        with self._lock:
            events = [event for event in self._buffer if not project_id or not event.project_id or event.project_id == project_id]
        return events[-limit:] if limit else []

    # ------------------------------------------------------------ subscribe
    def subscribe(self, project_id: str = "", *, replay: bool = True) -> queue.Queue[Event]:
        """Thread-safe mailbox for synchronous consumers."""
        channel: queue.Queue[Event] = queue.Queue(maxsize=MAILBOX_SIZE)

        def deliver(event: Event) -> None:
            try:
                channel.put_nowait(event)
            except queue.Full:
                pass  # a stalled client must never block the pipeline

        backlog = self._register(_Subscriber(scope=project_id, deliver=deliver, handle=channel), replay=replay)
        for event in backlog:
            deliver(event)
        return channel

    def subscribe_async(self, project_id: str = "", *, replay: bool = True) -> AsyncSubscription:
        """Mailbox for coroutines; must be called from inside the event loop."""
        subscription = AsyncSubscription(self, project_id, asyncio.get_running_loop())
        backlog = self._register(_Subscriber(scope=project_id, deliver=subscription._deliver, handle=subscription), replay=replay)
        for event in backlog:  # already on the loop: enqueue directly, in order
            subscription._put(event)
        return subscription

    def unsubscribe(self, channel: queue.Queue[Event] | AsyncSubscription) -> None:
        if isinstance(channel, AsyncSubscription):
            channel.close()
            return
        self._unregister(channel)

    def stream(self, project_id: str = "", *, heartbeat: float = 15.0) -> Iterator[dict[str, Any]]:
        """Blocking iterator for synchronous consumers (yields heartbeats when idle)."""
        channel = self.subscribe(project_id)
        try:
            while True:
                try:
                    event = channel.get(timeout=heartbeat)
                except queue.Empty:
                    yield {"type": "heartbeat", "at": time.time()}
                    continue
                yield event.to_payload()
        finally:
            self.unsubscribe(channel)

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # -------------------------------------------------------------- internals
    def _register(self, subscriber: _Subscriber, *, replay: bool) -> list[Event]:
        with self._lock:
            self._subscribers.append(subscriber)
            if not replay:
                return []
            backlog = [event for event in self._buffer if subscriber.wants(event)]
        return backlog[-REPLAY_EVENTS:]

    def _unregister(self, handle: object) -> None:
        with self._lock:
            self._subscribers = [item for item in self._subscribers if item.handle is not handle]


BUS = EventBus()


def publish(event_type: str, data: dict[str, Any], *, project_id: str = "", job_id: str = "") -> Event:
    return BUS.publish(event_type, data, project_id=project_id, job_id=job_id)


__all__ = ["AsyncSubscription", "BUS", "Event", "EventBus", "publish"]
