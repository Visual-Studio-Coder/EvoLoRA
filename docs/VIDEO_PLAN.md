# Demo Video Plan

The submission requires a **one-minute** demo video. It must clearly show what *this team built during the hackathon* (per the rules, unclear original contribution = disqualification). The goal: in 60 seconds, make judges understand the project, see it actually running, and remember why it is technically deeper than a chatbot.

## What The Video Must Land (rubric-driven)

The judging weights are Technicality 40%, Live Demo 20%, Creativity 25%, Future Potential 15%. So the video front-loads **technical depth** and **a working loop**, not slides.

- **Theme fit (say it once, fast):** EvoLoRA is Recursive Intelligence + the Self-Improvement Stack — one model proposes how another model's LoRA weights should change, while Python enforces the bounds. It is also Continual Learning: the same loop reruns on new goals while preserving run records and the best adapter.
- **The hard part:** a planner agent with *bounded* tools, a *locked + hashed* eval set the planner cannot self-grade, validated LoRA hyperparameters, an LLM-as-judge, persisted auditable runs, and a remote-GPU handoff — all built this weekend.
- **Honesty:** mock mode runs end-to-end with **no API keys**; live integrations (MiniMax planner, DigitalOcean judge, MongoDB Atlas, remote GPU VM) are optional and labeled.

## One-Minute Structure

| Time | Visual | Voiceover |
| --- | --- | --- |
| 0–6 sec | TUI title / repo README | "EvoLoRA is a bounded self-improvement loop: an agent plans how to improve a model's LoRA weights, and Python keeps every step auditable." |
| 6–15 sec | TUI run setup (goal + sample count) | "The task is strict-JSON customer spending summaries. The user sets a goal and how many training samples to generate — or lets the agent decide." |
| 15–28 sec | Plan / log pane: tool calls + generated examples + chosen hyperparameters | "The planner calls bounded tools — create evals, add training examples, start training — and picks LoRA hyperparameters. Python validates all of it before anything runs." |
| 28–38 sec | Eval lock + training progress | "The eval set is locked and hashed so the planner can't grade itself. Training runs in mock mode for a reliable demo, or hands the same config to a remote GPU VM." |
| 38–50 sec | Eval score → judge rating → approval prompt | "After training, the system scores the adapter against the locked evals, a DigitalOcean LLM judge rates it, and the loop only continues if it asks the user for another round." |
| 50–60 sec | Run record (Mongo / in-memory) + architecture doc | "Every run is traceable, the best adapter is kept, and it all works with zero API keys. This is infrastructure for auditable, recursive model improvement." |

## Must Show

- The TUI (or CLI mock) **actually running** — this is 20% of the score.
- A visible plan: tool calls, hyperparameters, and generated training examples.
- The locked/hashed eval and a real eval score.
- The judge rating/summary (or its labeled heuristic fallback).
- One on-screen line proving mock mode needs no keys, with live integrations optional.

## Avoid

- Do not pitch it as a generic chatbot (a banned/uninteresting category).
- Do not claim real Unsloth/GPU training, MongoDB, MiniMax, or DigitalOcean unless it was smoke-tested live — label fallbacks honestly.
- Do **not** claim the LiveKit or Gemini 3.5 prizes — those integrations are not in the core loop. Target the **DigitalOcean** prize instead (the judge path uses DO inference).
- Do not show secrets: `.env`, API keys, MongoDB URIs, or SSH keys.
- Do not burn the minute on slides — show the loop.

## Three-Minute Live Demo Backup (Round One)

Round One is a ~3-minute live demo + 1–2 min Q&A. Same story, more room to breathe:

1. `uv run evolora doctor` — prove config detection and every fallback mode.
2. `uv run evolora tui` — launch the live loop (or `uv run evolora demo --mock --iterations 1` if the TUI has rendering issues).
3. Narrate one full mock iteration: eval lock → baseline → plan → validation → training → eval → judge → stop/approval.
4. Open one file for technical credibility: `src/evolora/orchestration/orchestrator.py`, `src/evolora/agent/tools.py`, or `src/evolora/training/remote_config.py`.
5. Close on the future: swap mock training for the GPU VM path, keep the exact same eval + judge loop, persist every run.

## Recording Checklist

- Terminal sized so the TUI is fully readable; zoom in if recording from a laptop.
- Close unrelated tabs/windows; keep `.env` closed at all times.
- Do a dry run first — the 60-second cut has no room for a stall.
- Have `uv run evolora demo --mock --iterations 1` queued as the instant fallback if the TUI misbehaves.
- State original contribution clearly (on screen or in voiceover): "everything in this loop was built this weekend."
