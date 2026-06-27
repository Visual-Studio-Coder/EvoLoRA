EvoLoRA uses a MiniMax agent to train Phi-3-mini-128k-instruct so it gets better at one focused job. It makes the whole learning process visible: the agent plans, Phi trains, results are scored, and the best version is kept.

# EvoLoRA

EvoLoRA is a hackathon project for building an auditable, bounded self-improvement loop around LoRA fine-tuning. The current demo specializes `microsoft/Phi-3-mini-128k-instruct` for structured customer spending summaries that must return strict JSON.

Design sketch: https://excalidraw.com/#room=eb97987f23a5b8e55daa,sGaQZ17EchEOWLst6Xxo0Q

## Hackathon Fit

- Event: 2026 AI Engineer World's Fair Hackathon.
- Theme: Recursive Intelligence + The Self-Improvement Stack.
- Core claim: a bounded agent loop can propose LoRA training data, choose hyperparameters, train/evaluate an adapter, judge the result, and decide whether another iteration is worth user approval.
- Judge packet: [submission guide](docs/HACKATHON_SUBMISSION.md), [architecture](docs/ARCHITECTURE.md), and [video plan](docs/VIDEO_PLAN.md).
- Contribution boundary: this repo highlights the features built for the hackathon and labels mock/fallback/real integrations explicitly.

## What It Does

- MiniMax acts as the training strategist.
- Phi-3-mini-128k-instruct is the model being specialized.
- Python validates every plan before training starts.
- The eval set is locked and hashed before scoring.
- The mock backend runs with no API keys, no GPU, and no external services.
- Optional live services are clearly labeled when used: MiniMax planning, DigitalOcean judging, MongoDB persistence, and future GPU training.

## How The Loop Works

1. Start with a specialization goal or the built-in customer spending demo.
2. Lock the objective eval set and record its hash.
3. Run a baseline score.
4. Ask MiniMax to plan eval criteria, training examples, and LoRA settings.
5. Fall back to a heuristic planner when MiniMax is not configured.
6. Validate training examples and snap LoRA settings to safe allowed values.
7. Train through the selected backend, using the mock backend by default.
8. Score the result against the locked eval set.
9. Run the DigitalOcean judge when configured, or use a labeled heuristic judge.
10. Ask MiniMax whether another retraining round is worth doing.
11. Preserve the best result and stop when the loop reaches a target score, max iterations, patience, judge acceptance, user decline, cancellation, eval tampering, or failure.

## Built Pieces

- `evolora demo`: command-line demo loop
- `evolora tui`: Textual interface for live runs
- `evolora doctor`: environment and mode check
- `evolora smoke-minimax`: MiniMax connectivity smoke test
- MiniMax tool-calling planner with three bounded tools:
  - `create_evals`
  - `add_training_examples`
  - `start_training_model`
- Heuristic planner fallback for no-key runs
- Mock training backend and mock model runner
- Optional `unsloth` and `remote` backend boundaries
- Locked objective evaluator for strict JSON outputs
- DigitalOcean LLM judge with heuristic fallback
- MiniMax retrain advisor with heuristic fallback
- MongoDB run persistence with in-memory fallback
- Remote GPU config push and VM result pull over SSH/SFTP
- Local artifact store under ignored `artifacts/`
- Pytest coverage for planner, tools, evaluation, persistence, orchestration, models, and TUI behavior

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

Secrets stay in `.env`, which is ignored by Git. Create it locally using the variables below when live services are needed.

| Variable | Purpose |
| --- | --- |
| `BASE_MODEL_ID` | Defaults to `microsoft/Phi-3-mini-128k-instruct` |
| `MINIMAX_API_KEY` | Enables live MiniMax planning and retrain advice |
| `MINIMAX_MODEL` | Defaults to `MiniMax-M2.7-highspeed` |
| `MINIMAX_BASE_URL` | Defaults to `https://api.minimax.io/v1` |
| `DIGITAL_OCEAN_MODEL_ACCESS_KEY` | Enables the real DigitalOcean judge |
| `DIGITAL_OCEAN_INFERENCE_BASE_URL` | Defaults to `https://inference.do-ai.run/v1/` |
| `DIGITAL_OCEAN_JUDGE_MODEL` | Defaults to `llama3.3-70b-instruct` |
| `MONGODB_URI` | Enables MongoDB Atlas run persistence |
| `MONGODB_DB_NAME` | Defaults to `evolora` |
| `MONGODB_RUNS_COLLECTION` | Defaults to `runs` |
| `MONGODB_SERVER_SELECTION_TIMEOUT_MS` | Defaults to `3000` |
| `TRAINING_BACKEND` | `mock` by default; optional paths are `unsloth` and `remote` |
| `MODEL_RUNNER` | `mock` by default; `vm`/`remote` reads VM eval outputs |
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
  cli.py              Typer command entrypoint
  config.py           Environment-backed configuration
  agent/              MiniMax planner, tool schemas, heuristic fallback
  demo/               Customer spending demo task and evals
  evaluation/         Locked eval scoring and DigitalOcean judge
  inference/          Inference package boundary
  models/             Pydantic run, plan, event, and result models
  orchestration/      Main loop, stop conditions, approval, retrain advice
  persistence/        In-memory store, MongoDB store, local artifacts
  training/           Mock backend plus real-backend boundaries
  tui/                Native Textual app
tests/                Unit and integration tests
docs/                 Submission, architecture, and demo-video guide
artifacts/            Runtime outputs, ignored by Git
```

## Safety Boundaries

- Never commit `.env`, generated adapters, checkpoints, or secrets.
- Mock mode must stay functional with no API keys.
- MiniMax can suggest plans, but Python validates data and hyperparameters.
- MiniMax cannot pick arbitrary model IDs or write arbitrary files.
- DigitalOcean judge failures fall back to a labeled heuristic judge.
- MongoDB failures fall back to in-memory persistence.
- Real Unsloth or remote GPU training should stay clearly labeled until smoke-tested.

## Current Status

The main path is the mock end-to-end loop: MiniMax-guided Phi specialization, locked evaluation, planner fallback, validated examples and hyperparameters, TUI progress, cancellation, user-approved retraining, and CLI fallback.

Next useful work is real training and stronger run history: smoke-tested Unsloth or remote GPU training, artifact checksums for real adapters, richer MongoDB history views, and polished demo docs.
