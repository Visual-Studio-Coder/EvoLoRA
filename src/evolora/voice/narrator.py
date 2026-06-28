"""The Narrator: turns the orchestrator's live event stream into spoken narration.

Design goals:
* **Never block the caller.** ``observe(event)`` only enqueues and returns; all work
  happens in a background worker task. The orchestrator never touches this code — the
  TUI calls ``observe`` after it has already rendered the event.
* **Throttled + milestone-aware.** Milestones (baseline done, new best, run finished,
  approvals) speak immediately; everything else speaks at most once per interval, with
  newest-wins coalescing so a burst of training-step events becomes one line.
* **Best-effort.** Every failure is swallowed; mute short-circuits before synthesis.

The audio pipeline (``speak``) is injected so the scheduling logic can be tested with
no network or audio device.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable

from evolora.models.events import Event, EventKind

from .templates import MILESTONES, render_event

_SHUTDOWN = object()


def is_milestone(kind: EventKind) -> bool:
    return kind in MILESTONES


class Narrator:
    def __init__(
        self,
        *,
        speak: Callable[[str], Awaitable[None]],
        muted: Callable[[], bool],
        loop: asyncio.AbstractEventLoop,
        interval: float = 30.0,
        render: Callable[[Event], str | None] = render_event,
        time_fn: Callable[[], float] = time.monotonic,
        max_queue: int = 256,
    ) -> None:
        self._speak = speak
        self._muted = muted
        self._loop = loop
        self._interval = interval
        self._render = render
        self._now = time_fn
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self._last_spoken = float("-inf")
        self._last_text: str | None = None

    def observe(self, event: Event) -> None:
        """Enqueue an event for narration. Non-blocking; drops on overflow."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # narration is lossy by design — never back-pressure the TUI

    def start(self) -> None:
        if self._task is None:
            self._task = self._loop.create_task(self._worker())

    async def aclose(self) -> None:
        if self._task is None:
            return
        with contextlib.suppress(Exception):
            self._queue.put_nowait(_SHUTDOWN)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._task, timeout=2.0)
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(Exception):
                await self._task
        self._task = None

    async def _emit(self, text: str) -> None:
        if self._muted() or not text or text == self._last_text:
            return
        self._last_text = text
        self._last_spoken = self._now()
        with contextlib.suppress(Exception):
            await self._speak(text)

    async def _worker(self) -> None:
        pending: str | None = None
        while True:
            timeout: float | None = None
            if pending is not None:
                timeout = max(0.0, self._interval - (self._now() - self._last_spoken))
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout)
            except TimeoutError:
                if pending is not None:  # interval elapsed — flush the latest held line
                    await self._emit(pending)
                    pending = None
                continue

            if item is _SHUTDOWN:
                return

            sentence = None
            with contextlib.suppress(Exception):
                sentence = self._render(item)
            if not sentence:
                continue

            if is_milestone(item.kind):
                await self._emit(sentence)
                pending = None
            elif self._now() - self._last_spoken >= self._interval:
                await self._emit(sentence)
                pending = None
            else:
                pending = sentence  # newest wins; flushed when the interval elapses
