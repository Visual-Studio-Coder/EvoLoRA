"""Push-to-talk dictation: stream the mic through LiveKit STT into the input box.

A :class:`DictationSession` is created per hold. Start it when the user presses (mouse
down on the mic button or the global PTT key), and stop it on release; ``stop`` returns
the final transcript. Interim transcripts fire ``on_interim`` so the textbox updates
live. Fully isolated from training — failures are swallowed and never propagate.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable

from livekit.agents.stt import STT, SpeechEventType

from .audio_io import MicStream


class DictationSession:
    def __init__(
        self,
        stt: STT,
        mic: MicStream,
        loop: asyncio.AbstractEventLoop,
        on_interim: Callable[[str], None],
    ) -> None:
        self._stt = stt
        self._mic = mic
        self._loop = loop
        self._on_interim = on_interim
        self._stt_stream = None
        self._pump_task: asyncio.Task | None = None
        self._read_task: asyncio.Task | None = None
        self._finals: list[str] = []
        self._interim = ""

    def _emit(self) -> None:
        text = " ".join([*self._finals, self._interim]).strip()
        try:
            self._on_interim(text)
        except Exception:
            pass

    async def _pump_mic(self) -> None:
        try:
            async for frame in self._mic.frames():
                self._stt_stream.push_frame(frame)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                self._stt_stream.end_input()

    async def _read_events(self) -> None:
        try:
            async for ev in self._stt_stream:
                if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
                    text = ev.alternatives[0].text if ev.alternatives else ""
                    if text:
                        self._finals.append(text)
                    self._interim = ""
                    self._emit()
                elif ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    self._interim = ev.alternatives[0].text if ev.alternatives else ""
                    self._emit()
        except Exception:
            pass

    async def start(self) -> None:
        self._stt_stream = self._stt.stream()
        self._mic.start()
        self._pump_task = self._loop.create_task(self._pump_mic())
        self._read_task = self._loop.create_task(self._read_events())

    async def stop(self) -> str:
        """Stop capture and return the assembled final transcript."""
        self._mic.stop()  # ends mic.frames() -> _pump_mic calls end_input()
        if self._pump_task is not None:
            with contextlib.suppress(Exception):
                await self._pump_task
        # Give the STT a brief window to flush trailing finals, then tear down.
        if self._read_task is not None:
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(self._read_task, timeout=3.0)
            if not self._read_task.done():
                self._read_task.cancel()
                with contextlib.suppress(Exception):
                    await self._read_task
        if self._stt_stream is not None:
            with contextlib.suppress(Exception):
                await self._stt_stream.aclose()
        return " ".join(self._finals).strip()
