# Demo Video Plan

The submission requires a short one-minute video. The goal is to make the judges understand the project, see it working, and remember why it is technically deeper than a chatbot.

## One-Minute Structure

| Time | Visual | Voiceover |
| --- | --- | --- |
| 0-5 sec | TUI title or repo README | "EvoLoRA is a bounded self-improvement loop: an agent plans how to improve a model's LoRA weights, and Python keeps the loop auditable." |
| 5-15 sec | TUI run setup with goal/sample count | "The demo task is strict JSON customer spending summaries. The user can request an exact number of training samples, or leave it for the agent." |
| 15-28 sec | Planning/log pane and generated examples | "MiniMax calls tools to create evals, add training examples, and choose LoRA hyperparameters. The app validates everything before training." |
| 28-40 sec | Training/eval progress | "The eval set is locked and hashed. Training can run in mock mode for reliable demos or hand off config to a remote GPU VM." |
| 40-50 sec | Judge rating and approval prompt | "After training, a DigitalOcean LLM judge rates the adapter and summarizes whether another round is worth asking the user for." |
| 50-60 sec | Mongo/run record or architecture doc | "Every run is traceable in MongoDB or in-memory fallback, and the best adapter is preserved. This is infrastructure for auditable recursive improvement." |

## Must Show

- The TUI or CLI actually running.
- A visible plan with hyperparameters and training examples.
- Locked eval or eval score.
- Judge rating/summary or its heuristic fallback label.
- A clear statement that mock mode works without API keys, while live integrations are optional.

## Avoid

- Do not pitch it as a generic chatbot.
- Do not claim real Unsloth/GPU training unless it has been smoke-tested.
- Do not show secrets, `.env`, API keys, MongoDB passwords, or SSH keys.
- Do not spend the whole minute on slides; show the working loop.

## Three-Minute Live Demo Backup

1. Run `uv run evolora doctor` to prove config and fallback modes.
2. Launch `uv run evolora tui`.
3. Start a short mock run and narrate the loop: eval lock, baseline, plan, validation, training, eval, judge, stop/approval.
4. Show one code file for technical depth: `src/evolora/orchestration/orchestrator.py`, `src/evolora/agent/tools.py`, or `src/evolora/training/remote_config.py`.
5. End with the future: swap mock training for the VM path, keep the same eval and judge loop, persist every run.

## Recording Checklist

- Use a clean terminal size where the TUI is readable.
- Zoom the terminal if recording from a laptop.
- Close unrelated browser tabs and windows.
- Keep `.env` closed.
- Have `uv run evolora demo --mock --iterations 1` ready if the TUI recording has rendering issues.
