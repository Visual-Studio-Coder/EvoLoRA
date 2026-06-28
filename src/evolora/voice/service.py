"""VoiceService — the single facade the TUI talks to for all voice features.

Owns the LiveKit STT/TTS clients, the narrator, dictation lifecycle, the master mute,
and the optional global push-to-talk key. Designed to fail safe: if voice is disabled,
unconfigured, the ``voice`` extra is missing, or no audio device exists, every method
is a no-op and the TUI behaves exactly as before. Nothing here can affect training.

Heavy imports (livekit, sounddevice, pynput) are deferred to :meth:`start` so importing
this module never requires the optional ``voice`` extra.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable

from evolora.config import Config
from evolora.models.events import Event

from .narrator import Narrator

logger = logging.getLogger("evolora.voice")

_POLISH_SYSTEM = (
    "You are a live narrator for an AI model fine-tuning demo. Rewrite the given status "
    "line as ONE upbeat, natural spoken sentence of at most 12 words. No emojis, no "
    "quotes, no preamble — just the sentence."
)


class VoiceService:
    """Lifecycle + facade for dictation and narration. Construct via :meth:`create`."""

    def __init__(self, config: Config, loop: asyncio.AbstractEventLoop) -> None:
        self._config = config
        self._loop = loop
        self.enabled = False
        self.muted = False
        self.status = "disabled"  # human-readable reason, surfaced in the TUI

        self._session = None  # aiohttp.ClientSession
        self._stt = None
        self._tts = None
        self._speaker = None
        self._narrator: Narrator | None = None
        self._polish: Callable | None = None

        self._dictation = None
        self._dictating = False
        self._on_interim: Callable[[str], None] | None = None
        self._on_final: Callable[[str], None] | None = None
        self._ptt_listener = None

    # -- construction / gating -------------------------------------------------

    @classmethod
    def create(cls, config: Config, loop: asyncio.AbstractEventLoop) -> VoiceService:
        svc = cls(config, loop)
        if not config.voice_available:
            svc.status = "no LiveKit creds (voice off)"
        return svc

    async def start(self) -> str:
        """Bring voice online. Returns a short status string for the TUI to log."""
        if not self._config.voice_available:
            return self.status
        try:
            import aiohttp
            from livekit.agents import inference

            from .audio_io import SpeakerPlayer

            self._session = aiohttp.ClientSession()
            self._stt = inference.STT(
                model=self._config.stt_model,
                language="en",
                http_session=self._session,
                sample_rate=16_000,
            )
            self._tts = inference.TTS(
                model=self._config.tts_model,
                voice=self._config.tts_voice,
                http_session=self._session,
            )
            self._speaker = SpeakerPlayer()
            self._polish = self._build_polish()
            self._narrator = Narrator(
                speak=self._speak,
                muted=lambda: self.muted,
                loop=self._loop,
                interval=self._config.narrate_interval,
            )
            self._narrator.start()
            self.enabled = True
            self.status = f"on | stt={self._config.stt_model} tts={self._config.tts_model}"
            return self.status
        except Exception as exc:  # ImportError, device error, etc. — disable, never raise
            logger.warning("voice disabled: %s", exc)
            await self._cleanup_session()
            self.enabled = False
            self.status = f"unavailable ({type(exc).__name__})"
            return self.status

    def _build_polish(self):
        cfg = self._config
        if not (cfg.narrate_polish and cfg.minimax_available):
            return None
        try:
            from openai import AsyncOpenAI
        except Exception:
            return None
        client = AsyncOpenAI(api_key=cfg.minimax_api_key, base_url=cfg.minimax_base_url)

        async def polish(text: str) -> str:
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=cfg.minimax_model,
                        messages=[
                            {"role": "system", "content": _POLISH_SYSTEM},
                            {"role": "user", "content": text},
                        ],
                        max_tokens=40,
                        temperature=0.7,
                    ),
                    timeout=4.0,
                )
                out = (resp.choices[0].message.content or "").strip().strip('"')
                return out or text
            except Exception:
                return text  # demo-safe: fall back to the template line

        return polish

    # -- narration -------------------------------------------------------------

    def observe(self, event: Event) -> None:
        """Feed a live orchestrator event to the narrator (non-blocking, best-effort)."""
        if self.enabled and self._narrator is not None:
            self._narrator.observe(event)

    async def _speak(self, text: str) -> None:
        """Render one line to speech via LiveKit TTS and play it. Mute-aware."""
        if self.muted or self._tts is None or self._speaker is None:
            return
        if self._polish is not None:
            text = await self._polish(text)
            if self.muted:
                return
        frames = []
        stream = self._tts.stream()
        try:
            stream.push_text(text)
            stream.end_input()
            async for ev in stream:
                if self.muted:
                    break
                frames.append(getattr(ev, "frame", ev))
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
        if not self.muted and frames:
            await self._speaker.play(frames)

    # -- mute ------------------------------------------------------------------

    def toggle_mute(self) -> bool:
        self.set_muted(not self.muted)
        return self.muted

    def set_muted(self, value: bool) -> None:
        self.muted = value
        if value and self._speaker is not None:
            self._speaker.stop()  # cut any narration currently playing

    # -- dictation (push-to-talk) ---------------------------------------------

    def register_dictation_handlers(
        self,
        on_interim: Callable[[str], None],
        on_final: Callable[[str], None],
    ) -> None:
        """Where transcripts go. Used by both the mic button and the global PTT key."""
        self._on_interim = on_interim
        self._on_final = on_final

    async def begin_dictation(self) -> None:
        if not self.enabled or self._dictating or self._stt is None:
            return
        try:
            from .audio_io import MicStream
            from .dictation import DictationSession

            self._dictating = True
            mic = MicStream(self._loop)
            self._dictation = DictationSession(
                self._stt, mic, self._loop, self._on_interim or (lambda _t: None)
            )
            await self._dictation.start()
        except Exception as exc:
            logger.warning("dictation start failed: %s", exc)
            self._dictating = False
            self._dictation = None

    async def end_dictation(self) -> str:
        if not self._dictating or self._dictation is None:
            return ""
        try:
            text = await self._dictation.stop()
        except Exception as exc:
            logger.warning("dictation stop failed: %s", exc)
            text = ""
        finally:
            self._dictating = False
            self._dictation = None
        if text and self._on_final is not None:
            with contextlib.suppress(Exception):
                self._on_final(text)
        return text

    # -- global push-to-talk key (optional) -----------------------------------

    def start_ptt_key(self) -> None:
        """Listen for a global hold key (down=begin, up=end). Best-effort; optional."""
        if not self.enabled:
            return
        try:
            from pynput import keyboard
        except Exception:
            return
        key_name = (self._config.ptt_key or "").strip().lower()
        if not key_name:
            return
        target = getattr(keyboard.Key, key_name, None)
        if target is None and len(key_name) == 1:
            target = keyboard.KeyCode.from_char(key_name)
        if target is None:
            return

        def _matches(key) -> bool:  # noqa: ANN001
            return key == target

        def on_press(key) -> None:  # noqa: ANN001 - pynput thread
            if _matches(key) and not self._dictating:
                self._loop.call_soon_threadsafe(
                    lambda: self._loop.create_task(self.begin_dictation())
                )

        def on_release(key) -> None:  # noqa: ANN001 - pynput thread
            if _matches(key) and self._dictating:
                self._loop.call_soon_threadsafe(
                    lambda: self._loop.create_task(self.end_dictation())
                )

        try:
            self._ptt_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._ptt_listener.start()
        except Exception as exc:
            logger.warning("global PTT key unavailable: %s", exc)
            self._ptt_listener = None

    # -- teardown --------------------------------------------------------------

    async def _cleanup_session(self) -> None:
        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.close()
            self._session = None

    async def aclose(self) -> None:
        if self._ptt_listener is not None:
            with contextlib.suppress(Exception):
                self._ptt_listener.stop()
            self._ptt_listener = None
        if self._dictating:
            with contextlib.suppress(Exception):
                await self.end_dictation()
        if self._narrator is not None:
            await self._narrator.aclose()
            self._narrator = None
        await self._cleanup_session()
        self.enabled = False
