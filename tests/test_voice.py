"""Voice scheduling, templates, and config gating — no audio device or network."""

from __future__ import annotations

import asyncio

import pytest

from evolora.config import Config
from evolora.models.events import Event, EventKind
from evolora.voice.narrator import Narrator, is_milestone
from evolora.voice.templates import render_event


def _ev(kind: EventKind, **data) -> Event:
    return Event(kind=kind, run_id="run0001", data=data)


# -- config gating ------------------------------------------------------------


def test_voice_available_requires_enabled_and_all_creds():
    full = dict(livekit_url="wss://x", livekit_api_key="k", livekit_api_secret="s")
    assert Config(voice_enabled=True, **full).voice_available
    assert not Config(voice_enabled=False, **full).voice_available
    assert not Config(voice_enabled=True, livekit_url="wss://x").voice_available
    assert not Config(voice_enabled=True).voice_available


# -- templates ----------------------------------------------------------------


def test_templates_render_key_events_with_values():
    assert "0.62" in render_event(_ev(EventKind.BASELINE_COMPLETE, score=0.62))
    assert "0.81" in render_event(_ev(EventKind.BEST_UPDATED, score=0.81))
    line = render_event(_ev(EventKind.TRAINING_PROGRESS, step=12, total_steps=60, loss=0.34))
    assert "12" in line and "60" in line


def test_templates_skip_noisy_events():
    assert render_event(_ev(EventKind.LOG)) is None
    assert render_event(_ev(EventKind.STATUS_CHANGED)) is None


def test_milestone_classification():
    assert is_milestone(EventKind.RUN_COMPLETE)
    assert is_milestone(EventKind.BASELINE_COMPLETE)
    assert not is_milestone(EventKind.TRAINING_PROGRESS)


# -- narrator scheduling ------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.said: list[str] = []

    async def speak(self, text: str) -> None:
        self.said.append(text)


async def _settle() -> None:
    # Let the narrator worker drain the queue.
    for _ in range(5):
        await asyncio.sleep(0.02)


async def test_milestone_speaks_immediately_even_with_large_interval():
    rec = _Recorder()
    n = Narrator(speak=rec.speak, muted=lambda: False,
                 loop=asyncio.get_running_loop(), interval=1000.0)
    n.start()
    n.observe(_ev(EventKind.BASELINE_COMPLETE, score=0.5))
    await _settle()
    await n.aclose()
    assert rec.said and "0.50" in rec.said[0]


async def test_non_milestone_throttled_within_interval():
    rec = _Recorder()
    n = Narrator(speak=rec.speak, muted=lambda: False,
                 loop=asyncio.get_running_loop(), interval=1000.0)
    n.start()
    # First non-milestone speaks (last_spoken == -inf); the second is held back.
    n.observe(_ev(EventKind.TRAINING_PROGRESS, step=1, total_steps=10, loss=0.9))
    await _settle()
    n.observe(_ev(EventKind.TRAINING_PROGRESS, step=2, total_steps=10, loss=0.8))
    await _settle()
    await n.aclose()
    assert len(rec.said) == 1
    assert "step 1" in rec.said[0].lower() or "1" in rec.said[0]


async def test_newest_wins_flush_after_interval():
    rec = _Recorder()
    n = Narrator(speak=rec.speak, muted=lambda: False,
                 loop=asyncio.get_running_loop(), interval=0.15)
    n.start()
    n.observe(_ev(EventKind.RUN_STARTED))  # milestone -> speaks, sets last_spoken
    await _settle()
    n.observe(_ev(EventKind.TRAINING_PROGRESS, step=1, total_steps=9, loss=0.9))
    n.observe(_ev(EventKind.TRAINING_PROGRESS, step=2, total_steps=9, loss=0.5))
    await asyncio.sleep(0.4)  # exceed interval so the held (newest) line flushes
    await n.aclose()
    # Run-started spoke first; then exactly one held progress line (the newest).
    assert rec.said[0].startswith("Starting")
    assert any("step 2" in s.lower() or "2 of 9" in s.lower() for s in rec.said[1:])


async def test_mute_suppresses_narration():
    rec = _Recorder()
    muted = True
    n = Narrator(speak=rec.speak, muted=lambda: muted,
                 loop=asyncio.get_running_loop(), interval=1000.0)
    n.start()
    n.observe(_ev(EventKind.RUN_COMPLETE, best_score=0.9))
    await _settle()
    await n.aclose()
    assert rec.said == []
