EvoLoRA helps an AI system practice, judge its own work, and improve while people can watch. It is a hackathon demo about making self-improvement feel controlled instead of mysterious.

# EvoLoRA

EvoLoRA is an auditable self-improvement loop for small model specialization. The demo starts with a simple task, measures how well the model performs, asks an agent to plan better training examples and LoRA settings, runs a training backend, evaluates the result against a locked benchmark, and keeps the best adapter.

The current demo task is a structured customer spending summary. The model must read customer purchase data and return strict JSON with fields like top customer, total revenue, customer count, and a short summary.

## Why It Exists

Self-improving AI systems are easy to pitch and hard to trust. EvoLoRA makes the loop visible:

- every run has a clear starting score
- every improvement attempt has a plan
- training data and LoRA settings are validated before use
- evaluation happens against a locked benchmark
- fallback mode works with no API keys
- the loop stops for explicit reasons: target score, max iterations, patience, cancellation, eval tampering, or failure

## Demo Flow

1. Lock the evaluation set and record its hash.
2. Run a baseline score.
3. Ask MiniMax for a training plan, or use the heuristic planner when no key is available.
4. Validate the generated training examples and LoRA hyperparameters.
5. Train with the mock backend by default.
6. Evaluate the adapter on the locked benchmark.
7. Preserve the best result and repeat until a stop condition is reached.

## What Is Built

- `evolora demo`: plain CLI demo run
- `evolora tui`: live Textual interface with agent reasoning, training examples, LoRA config, metrics, and start/cancel controls
- `evolora doctor`: environment check for keys, persistence, and backend mode
- `evolora smoke-minimax`: MiniMax connectivity check when `MINIMAX_API_KEY` is set
- Mock training backend that runs locally with no GPU
- MiniMax planner through the OpenAI SDK, with heuristic fallback
- Locked evaluation set and objective JSON scorer
- In-memory run store and local artifact store
- Optional exact training sample count; blank means the planner chooses, a number means Python enforces that count

## Quick Start

From this directory:

```powershell
uv sync
uv run evolora doctor
uv run evolora demo --mock --iterations 2
uv run evolora tui
```

Mock mode is the default. You can run the core demo without `.env`, external services, or a GPU.

## Useful Commands

```powershell
# Run the mock CLI demo
uv run evolora demo --mock --iterations 2

# Launch the terminal UI
uv run evolora tui

# Check local configuration
uv run evolora doctor

# Test MiniMax only when a real key is present
uv run evolora smoke-minimax

# Run tests and lint
uv run pytest
uv run ruff check .
```

## Environment

Secrets stay in `.env`, which is ignored by Git. Use `.env.example` for the expected variable names.

Key variables:

- `MINIMAX_API_KEY`: enables live MiniMax planning
- `MINIMAX_MODEL`: defaults to `MiniMax-M2.7-highspeed`
- `MINIMAX_BASE_URL`: defaults to `https://api.minimax.io/v1`
- `MONGODB_URI`: future persistent run history
- `TRAINING_BACKEND`: `mock` by default; `unsloth` and `remote` are optional paths
- `MODEL_RUNNER`: `mock` by default
- `MAX_ITERATIONS`, `TARGET_SCORE`, `IMPROVEMENT_THRESHOLD`, `PATIENCE`: loop controls

## Project Layout

```text
src/evolora/
  cli.py              Typer commands
  config.py           Environment-backed configuration
  agent/              MiniMax planner and heuristic fallback
  orchestration/      Run loop, events, stop conditions, cancellation
  evaluation/         Locked benchmark and objective scoring
  training/           Mock backend and real-backend boundaries
  persistence/        Run records and artifact storage
  tui/                Textual live interface
  demo/               Customer spending demo data
tests/                Unit and integration tests
docs/                 Durable project notes
scripts/              Optional helper scripts
artifacts/            Ignored runtime outputs
```

## Boundaries

Mock mode is real and supported. Live MiniMax planning is optional. Real Unsloth or remote GPU training should stay clearly labeled until it is smoke-tested. Do not commit `.env`, generated adapters, checkpoints, or secrets.

## Current Status

The P0 path is the focus: mock end-to-end loop, locked evaluation, planner fallback, validated examples and hyperparameters, TUI progress, cancellation, and CLI fallback. P1 items such as real GPU training, richer run history, and deployment polish should come after the mock loop stays stable.
