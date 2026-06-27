EvoLoRA uses a MiniMax agent to train Phi-3-mini-128k-instruct so it gets better at one focused job. It turns that learning process into a visible loop where the agent plans, Phi trains, results are scored, and the best version is kept.

# EvoLoRA

EvoLoRA is an auditable self-improvement loop for specializing Phi-3-mini-128k-instruct with LoRA. The demo starts with a task, measures how well Phi performs, asks MiniMax to plan better training examples and LoRA settings, runs a training backend, evaluates the result against a locked benchmark, and keeps the best result.

Design sketch: https://excalidraw.com/#room=eb97987f23a5b8e55daa,sGaQZ17EchEOWLst6Xxo0Q

The current demo task is a structured customer spending summary. The model must read customer purchase data and return strict JSON with fields like top customer, total revenue, customer count, and a short summary.

## Why It Exists

Self-improving AI systems are easy to pitch and hard to trust. EvoLoRA narrows the idea into one concrete workflow: MiniMax acts as the training strategist, Phi-3-mini-128k-instruct is the model being specialized, and Python keeps the loop bounded and inspectable.

EvoLoRA makes the loop visible:

- every run has a clear starting score
- every improvement attempt has a plan
- training data and LoRA settings are validated before use
- evaluation happens against a locked benchmark
- DigitalOcean can review the trained adapter as an LLM judge, then MiniMax decides whether another iteration is worth asking the user for
- fallback mode works with no API keys
- the loop stops for explicit reasons: target score, max iterations, patience, judge acceptance, user-declined retrain, cancellation, eval tampering, or failure

## Demo Flow

1. Lock the evaluation set and record its hash.
2. Run a baseline score.
3. Ask MiniMax for a training plan, or use the heuristic planner when no key is available.
4. Validate the generated training examples and LoRA hyperparameters.
5. Train Phi-3-mini-128k-instruct with the mock backend by default.
6. Evaluate the adapter on the locked benchmark.
7. Run a DigitalOcean LLM judge when `DIGITAL_OCEAN_MODEL_ACCESS_KEY` is set; otherwise use a labeled heuristic judge.
8. Send the judge rating and summary to MiniMax for a retrain/stop recommendation.
9. In the TUI, ask the user before retraining when MiniMax recommends another round.
10. Preserve the best result and repeat until a stop condition is reached.

## What Is Built

- `evolora demo`: plain CLI demo run
- `evolora tui`: live Textual interface with agent reasoning, training examples, LoRA config, metrics, and start/cancel controls
- `evolora doctor`: environment check for keys, persistence, and backend mode
- `evolora smoke-minimax`: MiniMax connectivity check when `MINIMAX_API_KEY` is set
- Mock training backend that simulates Phi specialization locally with no GPU
- MiniMax planner through the OpenAI SDK, with heuristic fallback
- DigitalOcean judge through the OpenAI-compatible Inference endpoint, with heuristic fallback
- MiniMax retrain decision after the judge report, with user approval controls in the TUI
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
- `DIGITAL_OCEAN_MODEL_ACCESS_KEY`: enables the real DigitalOcean judge
- `DIGITAL_OCEAN_INFERENCE_BASE_URL`: defaults to `https://inference.do-ai.run/v1/`
- `DIGITAL_OCEAN_JUDGE_MODEL`: defaults to `llama3.3-70b-instruct`
- `BASE_MODEL_ID`: defaults to `microsoft/Phi-3-mini-128k-instruct`
- `MONGODB_URI`: enables MongoDB Atlas run persistence
- `MONGODB_DB_NAME`: defaults to `evolora`
- `MONGODB_RUNS_COLLECTION`: defaults to `runs`
- `MONGODB_SERVER_SELECTION_TIMEOUT_MS`: defaults to `3000`
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

Mock mode is real and supported. Live MiniMax planning and live DigitalOcean judging are optional. When the DigitalOcean model access key is missing or the call fails, EvoLoRA labels the judge result as a heuristic fallback. When `MONGODB_URI` is configured, each run is upserted into MongoDB with `run_id` as `_id`; every iteration's hyperparameters and LLM-as-judge report are also copied into query-friendly `hyperparams_by_iteration` and `judge_reports` arrays linked by the same `run_id`. If MongoDB is unreachable, the run falls back to in-memory storage so demos keep working. Real Unsloth or remote GPU training should stay clearly labeled until it is smoke-tested. Do not commit `.env`, generated adapters, checkpoints, or secrets.

## Current Status

The P0 path is the focus: mock end-to-end loop for MiniMax-guided Phi specialization, locked evaluation, planner fallback, validated examples and hyperparameters, TUI progress, cancellation, and CLI fallback. P1 items such as real GPU training, richer run history, and deployment polish should come after the mock loop stays stable.
