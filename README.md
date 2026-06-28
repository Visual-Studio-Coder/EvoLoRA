# EvoLoRA

EvoLoRA is an auditable, bounded self-improvement loop for LoRA fine-tuning. You give it a plain-English goal; a MiniMax agent plans the evaluation set, the training data, and the LoRA hyperparameters; Python validates and controls every step; a small model is trained and scored against a locked benchmark; and the best adapter is preserved. The whole learning process is made visible: the agent plans, the model trains, results are scored and judged, and you decide whether to keep going.

The current demo specializes `microsoft/Phi-3-mini-128k-instruct` for structured outputs that must return strict JSON (the built-in task is a customer spending summary, but any goal can drive the loop).

Design sketch: https://excalidraw.com/#room=eb97987f23a5b8e55daa,sGaQZ17EchEOWLst6Xxo0Q

## Hackathon Fit

- Event: 2026 AI Engineer World's Fair Hackathon.
- Theme: Recursive Intelligence + The Self-Improvement Stack.
- Core claim: a bounded agent loop can propose a goal-specific eval set, synthesize LoRA training data, choose hyperparameters from safe choice sets, train and evaluate an adapter, judge the result, and decide whether another iteration is worth doing — all under explicit Python control and user approval.
- Judge packet: [submission guide](docs/HACKATHON_SUBMISSION.md), [architecture](docs/ARCHITECTURE.md), and [video plan](docs/VIDEO_PLAN.md).

## What It Does

- A MiniMax agent acts as the training strategist, driving each iteration through three bounded tools.
- Phi-3-mini-128k-instruct is the model being specialized.
- Python validates every plan, snaps hyperparameters onto safe values, and controls all file writes.
- The eval set is locked and SHA-256 hashed before scoring; tampering halts the run.
- The mock backend runs end to end with no API keys, no GPU, and no external services.
- Optional live services are clearly labeled when used: MiniMax planning, DigitalOcean judging, MongoDB persistence, and remote GPU training.

## How The Loop Works

1. Start with a specialization goal (or the built-in customer spending demo).
2. When a goal is given, MiniMax generates a goal-specific objective eval set, which you approve before it is used.
3. Lock the eval set and record its hash.
4. Run a baseline score. If the base model already aces the evals, regenerate a harder set once so there is room to improve.
5. Ask MiniMax to plan eval criteria, training examples, and LoRA settings via its tools.
6. Fall back to a heuristic planner when MiniMax is not configured.
7. Validate and de-duplicate training examples; snap LoRA settings to allowed values. Examples stack across iterations.
8. Train through the selected backend (mock by default).
9. Score the result against the locked eval set (objective scorer, or an LLM-as-judge for remote/generated evals).
10. Run the DigitalOcean judge when configured, or a labeled heuristic judge.
11. Ask MiniMax whether another retraining round is worth doing.
12. When the model is "good enough," ask the user whether to keep training to push it further.
13. Preserve the best result and stop on a target score, max iterations, patience, judge acceptance, user decline, cancellation, eval tampering, or repeated training failure.

## Built Pieces

- `evolora demo`: command-line demo loop
- `evolora tui`: Textual interface for live runs (default when `evolora` is run with no subcommand)
- `evolora doctor`: environment and mode check
- `evolora smoke-minimax`: MiniMax connectivity smoke test
- MiniMax tool-calling planner with three bounded tools: `create_evals`, `add_training_examples`, `start_training_model`
- MiniMax goal-driven eval generation, with adaptive difficulty and user approval
- Heuristic planner fallback for no-key runs
- Mock training backend and mock model runner
- Remote GPU training backend over SSH/SFTP, plus an `unsloth` backend boundary
- Locked objective evaluator for strict JSON, plus a generic evaluator and an LLM-as-judge
- DigitalOcean LLM judge with heuristic fallback
- MiniMax retrain advisor with heuristic fallback
- MongoDB run persistence with in-memory fallback
- Per-run observability logs and a local artifact store under ignored `artifacts/`
- Pytest coverage for planner, tools, evaluation, persistence, orchestration, models, remote backend, and TUI behavior

## Quick Start

From this directory:

```powershell
uv sync
uv run evolora doctor
uv run evolora demo --mock --iterations 2
uv run evolora tui
```

Mock mode is the default path. It is meant to work without `.env`, API keys, MongoDB, or a GPU.

## Useful Commands

| Command | Purpose |
| --- | --- |
| `uv run evolora` | Launch the TUI by default |
| `uv run evolora tui` | Launch the Textual interface |
| `uv run evolora demo --mock --iterations 2` | Run a short mock demo |
| `uv run evolora doctor` | Check local configuration |
| `uv run evolora smoke-minimax` | Test MiniMax when `MINIMAX_API_KEY` is set |
| `uv run pytest` | Run the test suite |
| `uv run ruff check .` | Run lint checks |

## Configuration

Secrets stay in `.env`, which is ignored by Git. Create it locally (see `.env.example`) when live services are needed.

| Variable | Purpose |
| --- | --- |
| `BASE_MODEL_ID` | Model being specialized; demo default `microsoft/Phi-3-mini-128k-instruct` |
| `MINIMAX_API_KEY` | Enables live MiniMax planning, eval generation, and retrain advice |
| `MINIMAX_MODEL` | Defaults to `MiniMax-M2.7-highspeed` |
| `MINIMAX_BASE_URL` | Defaults to `https://api.minimax.io/v1` |
| `MINIMAX_GROUP_ID` | MiniMax account group id |
| `DIGITAL_OCEAN_MODEL_ACCESS_KEY` | Enables the real DigitalOcean judge |
| `DIGITAL_OCEAN_INFERENCE_BASE_URL` | Defaults to `https://inference.do-ai.run/v1/` |
| `DIGITAL_OCEAN_JUDGE_MODEL` | Defaults to `llama3.3-70b-instruct` |
| `MONGODB_URI` | Enables MongoDB Atlas run persistence |
| `MONGODB_DB_NAME` | Defaults to `evolora` |
| `MONGODB_RUNS_COLLECTION` | Defaults to `runs` |
| `MONGODB_SERVER_SELECTION_TIMEOUT_MS` | Defaults to `3000` |
| `TRAINING_BACKEND` | `mock` by default; optional paths are `remote` and `unsloth` |
| `MODEL_RUNNER` | `mock` by default; `remote` reads VM eval outputs |
| `SSH_HOST`, `SSH_USER`, `SSH_PORT`, `SSH_KEY_PATH` | Enables remote VM SFTP config/result exchange |
| `REMOTE_CONFIG_PATH` | Where EvoLoRA writes the VM config JSON |
| `REMOTE_RESULTS_PATH` | Where the VM writes eval outputs for EvoLoRA to score |
| `MAX_ITERATIONS` | Max training loop iterations |
| `TARGET_SCORE` | Score target for stopping |
| `IMPROVEMENT_THRESHOLD` | Minimum improvement before patience counts down |
| `PATIENCE` | Stop after repeated low-improvement iterations |

## Repo Layout

```text
src/evolora/
  cli.py              Typer command entrypoint (TUI by default)
  config.py           Environment-backed configuration
  agent/              MiniMax planner, bounded tool schemas, heuristic fallback
  demo/               Customer spending demo task and evals
  evaluation/         Locked/generic scoring, DigitalOcean judge, LLM-as-judge
  inference/          Inference package boundary
  models/             Pydantic run, plan, event, judge, and result models
  observability/      Per-run event logging
  orchestration/      Main loop, stop conditions, approval gates, retrain advice
  persistence/        In-memory store, MongoDB store, local artifacts
  training/           Mock backend, remote SSH/SFTP backend, unsloth boundary
  tui/                Native Textual app
src/virtual_machine_code/   GPU VM scripts: train.py, evaluate.py, chat.py, data
scripts/vm/                 Baseline/eval helpers pushed to the GPU VM
tests/              Unit and integration tests
docs/               Submission, architecture, and demo-video guide
artifacts/          Runtime outputs, ignored by Git
```

## Safety Boundaries

- Never commit `.env`, generated adapters, checkpoints, or secrets.
- Mock mode must stay functional with no API keys.
- MiniMax can propose evals, examples, and hyperparameters, but Python validates the data and snaps hyperparameters onto safe values.
- MiniMax cannot pick arbitrary model IDs or write arbitrary files.
- Expected eval answers stay local; remote config pushes only eval prompts plus the eval-set hash.
- DigitalOcean judge failures fall back to a labeled heuristic judge.
- MongoDB failures fall back to in-memory persistence.
- Remote GPU training is clearly labeled and gated on SSH configuration.

## Current Status

The main path is the mock end-to-end loop: goal-driven MiniMax planning, locked evaluation with adaptive difficulty, planner fallback, validated and stacked training examples, judged iterations, TUI progress, cancellation, user-approved retraining, and a CLI fallback.

Next useful work is real training and richer history: a fully exercised remote GPU path, artifact checksums for real adapters, deeper MongoDB history views, and polished demo docs.
