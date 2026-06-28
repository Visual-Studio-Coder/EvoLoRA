# EvoLoRA Voice — Design Spec

**Date:** 2026-06-27
**Status:** Approved (brainstorm), implementing

## Goal

Add two LiveKit-powered voice features to the EvoLoRA TUI, fully decoupled from the
training loop:

1. **Dictation (push-to-talk):** hold the 🎤 button (mouse) or a global key (F9) to
   speak; live transcript streams into the goal/chat input box. Serves both
   "describe the model to build" and chat-mode drafting (same `#goal-input`).
2. **The Narrator:** observes the orchestrator's live event stream and speaks short
   (~10-word) narration sentences about what the loop is doing, throttled to ~30s
   with immediate lines on key milestones.

Plus a **persistent corner mute button** (topbar, always visible) that silences ALL
voice/sound output. Also bound to `ctrl+m`.

## Hard constraints

- **Zero interference with training.** Voice never runs on the orchestrator path.
  The orchestrator does not import or call voice code. The TUI calls
  `voice.narrator.observe(event)` only *after* it has already rendered the event,
  and `observe()` is non-blocking (enqueue + return). All STT/TTS/audio/network work
  happens in dedicated workers/threads. Every voice failure is swallowed — it can
  never break a run or the TUI (same discipline as `observability/run_logger.py`).
- **Graceful disable.** No LiveKit creds, missing packages, or no audio device →
  voice silently disables, TUI behaves exactly as before. Honors "mock works with no
  keys."
- **No secret exposure.** Creds live only in gitignored `.env`.

## Approach

LiveKit **Inference** standalone STT + TTS, in-process — no room, no separate agent
worker, no extra provider API keys. Auth via `LIVEKIT_URL/API_KEY/API_SECRET`; usage
bills against LiveKit credits. (Rejected: full room+worker = overkill; MiniMax-only =
no credit use, no STT.)

## Module layout — new `evolora/src/evolora/voice/`

| Module | Responsibility |
|---|---|
| `audio_io.py` | Mic capture + speaker playback via `sounddevice`; bridges PortAudio callback thread ↔ asyncio with thread-safe queues; converts `numpy ↔ rtc.AudioFrame`. |
| `dictation.py` | `DictationSession` wrapping `inference.STT().stream()`. `start()` opens mic + streams frames; interim transcripts → callback; `stop()` ends input → final transcript. |
| `narrator.py` | `Narrator.observe(event)` enqueues; a worker throttles (~30s) + fires immediately on milestones, coalesces bursts, renders sentence (template [+MiniMax]), synthesizes via `inference.TTS()`, plays. Honors mute. |
| `templates.py` | `EventKind → ~10-word sentence`. |
| `service.py` | `VoiceService` facade: owns STT/TTS/narrator/mute, lifecycle, graceful-disable gating. The only thing the TUI imports. |

## TUI integration (`tui/app.py`)

- Topbar gets a persistent **🔊/🔇 mute** button (corner) + `ctrl+m` binding.
- Input bar gets a **🎤 hold-to-talk** button (Textual `MouseDown`→start,
  `MouseUp`→stop). Optional global F9 PTT via `pynput` thread.
- Dictation interim text → live into `#goal-input`; final on release.
- `_apply_event` gains one line: `self._voice.narrator.observe(event)`.

## Data flow

- **Dictation:** mic → frame queue → LiveKit STT → interim/final → textbox.
- **Narration:** `Event` → `observe()` (non-blocking) → worker throttle/coalesce →
  sentence → LiveKit TTS → audio frames → speaker. Mute short-circuits before synth.

## Config (`.env` + `config.py` fields)

`LIVEKIT_URL/API_KEY/API_SECRET`, `VOICE_ENABLED`, `STT_MODEL=deepgram/nova-3`,
`TTS_MODEL=cartesia/sonic-3`, `TTS_VOICE`, `NARRATE_INTERVAL=30`, `PTT_KEY=f9`.
`Config.voice_available` gates everything.

## Dependencies (new optional `voice` extra)

`livekit-agents`, `sounddevice`, `pynput`. No VAD needed (PTT = manual start/stop;
silero optional for future auto-stop).

## Testing

- Unit (no hardware/network): templates, narrator throttle/coalesce/mute, config
  gating — fake STT/TTS sinks. pytest stays green with no audio device.
- Manual smoke checklist: standalone TTS, dictation, full-loop narration, mute.

## Out of scope

CLI narration (log-tailing). Voice is TUI-first for this build.
