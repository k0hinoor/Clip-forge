"""In-process event bus for live UI updates (Server-Sent Events).

The worker publishes progress; the API streams it to the browser. A small ring
buffer keeps recent events per project so a page refresh (or an SSE reconnect)
does not lose the current state.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

MAX_BUFFER = 200


@dataclass
class Event:
    type: str
    data: dict[str, Any]
    project_id: str = ""
    job_id: str = ""
    at: float = field(default_factory=time.time)

    def to_payload(self) -> dict[str, Any]:
        return {"type": self.type, "at": self.at, **self.data}


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[tuple[str, queue.Queue[Event]]] = []
        self._lock = threading.Lock()
        self._buffer: deque[Event] = deque(maxlen=MAX_BUFFER)

    # -------------------------------------------------------------- publish
    def publish(self, event_type: str, data: dict[str, Any], *, project_id: str = "", job_id: str = "") -> Event:
        event = Event(type=event_type, data=data, project_id=project_id, job_id=job_id)
        with self._lock:
            self._buffer.append(event)
            subscribers = list(self._subscribers)
        for scope, channel in subscribers:
            if scope and project_id and scope != project_id:
                continue
            try:
                channel.put_nowait(event)
            except queue.Full:
                pass  # a stalled client must never block the pipeline
        return event

    # ------------------------------------------------------------ subscribe
    def subscribe(self, project_id: str = "") -> queue.Queue[Event]:
        channel: queue.Queue[Event] = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append((project_id, channel))
            recent = [event for event in self._buffer if not project_id or not event.project_id or event.project_id == project_id][-40:]
        for event in recent:
            try:
                channel.put_nowait(event)
            except queue.Full:
                break
        return channel

    def unsubscribe(self, channel: queue.Queue[Event]) -> None:
        with self._lock:
            self._subscribers = [item for item in self._subscribers if item[1] is not channel]

    def stream(self, project_id: str = "") -> Iterator[dict[str, Any]]:
        channel = self.subscribe(project_id)
        try:
            while True:
                try:
                    event = channel.get(timeout=15.0)
                except queue.Empty:
                    yield {"type": "heartbeat", "at": time.time()}
                    continue
                yield event.to_payload()
        finally:
            self.unsubscribe(channel)

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


BUS = EventBus()


def publish(event_type: str, data: dict[str, Any], *, project_id: str = "", job_id: str = "") -> Event:
    return BUS.publish(event_type, data, project_id=project_id, job_id=job_id)


__all__ = ["BUS", "Event", "EventBus", "publish"]
