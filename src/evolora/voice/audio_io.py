"""Local microphone capture and speaker playback for the voice features.

Bridges the PortAudio (``sounddevice``) callback thread to the asyncio loop with
thread-safe queues. Everything here is best-effort: import or device failures are
surfaced as exceptions at construction/start so the caller can disable voice without
ever affecting the training loop.

Audio formats:
* Mic capture: 16 kHz, mono, int16 — fed to LiveKit STT as ``rtc.AudioFrame``.
* Playback: whatever sample rate the TTS frames carry (LiveKit Inference emits 24 kHz
  mono); the output stream is opened lazily from the first frame.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd
from livekit import rtc

MIC_SAMPLE_RATE = 16_000
MIC_CHANNELS = 1
MIC_BLOCK = 1_600  # 100 ms per block


class MicStream:
    """Capture microphone audio and expose it as an async stream of ``rtc.AudioFrame``.

    The sounddevice callback runs on a separate thread; frames are handed to the
    asyncio loop via ``call_soon_threadsafe``. Call :meth:`start`, iterate
    :meth:`frames`, then :meth:`stop` (push-to-talk: start on key/mouse down, stop on
    release).
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[rtc.AudioFrame | None] = asyncio.Queue()
        self._stream: sd.InputStream | None = None
        self._active = False

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ANN001 - sd callback
        # Runs on the PortAudio thread. Copy out, never block, never raise.
        if not self._active:
            return
        try:
            pcm = bytes(indata)  # int16 little-endian, mono
            frame = rtc.AudioFrame(
                data=pcm,
                sample_rate=MIC_SAMPLE_RATE,
                num_channels=MIC_CHANNELS,
                samples_per_channel=frames,
            )
            self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)
        except Exception:
            pass

    def start(self) -> None:
        self._active = True
        self._stream = sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=MIC_CHANNELS,
            dtype="int16",
            blocksize=MIC_BLOCK,
            callback=self._on_audio,
        )
        self._stream.start()

    async def frames(self) -> AsyncIterator[rtc.AudioFrame]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame

    def stop(self) -> None:
        self._active = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        # Unblock any pending frames() consumer.
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)


class SpeakerPlayer:
    """Play a sequence of TTS audio frames through the default output device.

    Narration utterances are short, so frames are buffered and played as one blocking
    write offloaded to a worker thread (avoids stream-underrun bookkeeping). Playback
    can be cut off immediately by :meth:`stop` (used by mute).
    """

    def __init__(self) -> None:
        self._stopped = False

    def _blocking_play(self, buffer: np.ndarray, sample_rate: int) -> None:
        sd.play(buffer, samplerate=sample_rate)
        sd.wait()

    async def play(self, frames: list[rtc.AudioFrame]) -> None:
        if not frames:
            return
        sample_rate = frames[0].sample_rate
        chunks = [np.frombuffer(f.data, dtype=np.int16) for f in frames]
        buffer = np.concatenate(chunks)
        self._stopped = False
        await asyncio.to_thread(self._blocking_play, buffer, sample_rate)

    def stop(self) -> None:
        self._stopped = True
        try:
            sd.stop()
        except Exception:
            pass
