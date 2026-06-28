"""Voice features for EvoLoRA: push-to-talk dictation and the live Narrator.

Fully decoupled from the training loop and best-effort: if LiveKit creds, the optional
``voice`` extra, or an audio device are missing, voice silently disables and the TUI is
unaffected. The TUI interacts only with :class:`VoiceService`.
"""

from __future__ import annotations

from .service import VoiceService

__all__ = ["VoiceService"]
