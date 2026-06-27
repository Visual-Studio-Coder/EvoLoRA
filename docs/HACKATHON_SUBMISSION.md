# Hackathon Submission Guide

This document is the judge-facing front door for EvoLoRA at the 2026 AI Engineer World's Fair Hackathon.

## Event Fit

Chosen themes:

- Recursive Intelligence: MiniMax plans LoRA data and hyperparameters that directly affect future model weights.
- The Self-Improvement Stack: EvoLoRA provides the bounded infrastructure around planning, training, locked evaluation, judging, persistence, and stop decisions.
- Continual Learning: the loop can be rerun on future goals or failures while preserving run records and best adapters.

Primary demo task: specialize `microsoft/Phi-3-mini-128k-instruct` for strict JSON customer spending summaries.

## What Was Built For The Hackathon

- Native Textual TUI for the live self-improvement loop.
- MiniMax tool-calling planner with bounded tools: `create_evals`, `add_training_examples`, and `start_training_model`.
- Python-side validation of LoRA hyperparameters, training examples, eval locks, stop conditions, and user approval.
- Mock backend that runs end to end without API keys or GPU access.
- DigitalOcean-compatible LLM-as-judge path with heuristic fallback.
- MongoDB Atlas run persistence with in-memory fallback.
- Remote VM bridge: SFTP config push to `REMOTE_CONFIG_PATH` and result pull from `REMOTE_RESULTS_PATH`.
- Tests covering tools, planner, evaluator, persistence, remote config, remote runner, orchestrator, models, and TUI behavior.

## Integration Status

| Integration | Status | Notes |
| --- | --- | --- |
| MiniMax | Real when `MINIMAX_API_KEY` is set; heuristic fallback otherwise | Used for planning and retrain advice. |
| DigitalOcean judge | Real when `DIGITAL_OCEAN_MODEL_ACCESS_KEY` is set; heuristic fallback otherwise | Reviews post-training behavior and produces a summary/rating. |
| MongoDB Atlas | Real when `MONGODB_URI` is set; in-memory fallback otherwise | Stores run records, hyperparameters by iteration, and judge reports. |
| Remote GPU VM | SSH/SFTP bridge implemented; real training still must be smoke-tested on-site | Config push sends hyperparameters, training examples, and eval prompts only. Expected answers stay local. |
| LiveKit | Not active in the current core loop | Do not claim this prize unless a real feature is added. |
| Gemini 3.5 | Not active in the current core loop | Do not claim this prize unless a real feature is added. |

## Rubric Map

Technicality, 40 percent:

- Multi-agent workflow with explicit tool contracts.
- Bounded LoRA hyperparameter selection and training-data validation.
- Locked eval hash and local scoring so the planner cannot self-grade.
- Remote GPU handoff over SFTP with dry-run safety.
- MongoDB persistence for auditability.

Live demo, 20 percent:

- Start from the TUI.
- Show the baseline score, plan, training progress, eval score, judge rating, and approval prompt.
- Keep a CLI mock demo ready as the fallback path.

Creativity and originality, 25 percent:

- The core idea is not another chatbot. It is a visible system where one model proposes how another model should improve its weights, while Python enforces bounds.

Future potential and AI impact, 15 percent:

- Points toward auditable self-improving model systems: continuous evaluation, controlled training, traceable runs, and human approval when the agent wants another loop.

## Commands For Judges Or Teammates

```powershell
uv sync
uv run evolora doctor
uv run evolora demo --mock --iterations 1
uv run evolora tui
uv run pytest -q
uv run ruff check .
```

## Submission Checklist

- Repository is public.
- Demo video is accessible.
- Submission page includes all team members.
- The demo only highlights features built during the hackathon.
- `.env` is not committed.
- Mock mode works without API keys.
- Real integrations are labeled as real only after smoke tests.
- One-minute video follows `docs/VIDEO_PLAN.md`.
