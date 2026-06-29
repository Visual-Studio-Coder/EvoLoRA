# EvoLoRA - AI Engineer World's Fair Hackathon 2026 (FINALIST)
Submission: https://cerebralvalley.ai/e/aiewf-hackathon-2026/hackathon/gallery?project=14

EvoLoRA is an auditable, bounded self-improvement loop for LoRA fine-tuning. You give it a plain-English goal; a MiniMax agent plans the evaluation set, the training data, and the LoRA hyperparameters; Python validates and controls every step; a small model is trained on a GPU and scored against a locked benchmark; and the best adapter is preserved with a human-readable name. The whole learning process is made visible: the agent plans, the model trains, results are scored and judged, and you decide whether to keep going — or hand the whole thing off to run autonomously.

Design sketch: https://excalidraw.com/#room=eb97987f23a5b8e55daa,sGaQZ17EchEOWLst6Xxo0Q

## Hackathon Fit

- **Event:** 2026 AI Engineer World's Fair Hackathon.
- **Theme:** Recursive Intelligence + The Self-Improvement Stack.
- **Core claim:** a bounded agent loop can propose a goal-specific eval set, synthesize LoRA training data, choose hyperparameters from safe choice sets, train and evaluate an adapter, judge the result, and decide whether another iteration is worth doing — all under explicit Python control, with optional human approval gates.
- **Judge packet:** [submission guide](docs/HACKATHON_SUBMISSION.md), [architecture](docs/ARCHITECTURE.md), and [video plan](docs/VIDEO_PLAN.md).

---

## Project Specification

### What it does

You state a specialization goal in plain English (e.g. *"a model that specializes in generating strict JSON from a described schema"*). EvoLoRA then:

1. **Generates a goal-specific eval set** with MiniMax — the objective benchmark the run is scored against.
2. **Locks and hashes** that eval set (SHA-256) so it cannot be silently changed mid-run.
3. **Baselines** the untrained model. If it already aces the evals, EvoLoRA regenerates a *harder* set once so there is room to improve.
4. **Plans each iteration** through three bounded MiniMax tools: eval criteria, synthetic training data, and LoRA hyperparameters.
5. **Trains** a LoRA adapter on a GPU (via Unsloth) — or in a no-GPU mock backend.
6. **Scores** the adapter against the locked eval set, runs an LLM-as-judge, and asks a retrain advisor whether another round is worthwhile.
7. **Stacks** training examples across iterations and **archives** each adapter under a readable, goal-derived name.
8. **Stops** on a target score, max iterations, patience, judge acceptance, user decline, cancellation, eval tampering, or repeated training failure — preserving the best adapter.

### Design principles

- **Bounded agency.** MiniMax proposes; Python disposes. The agent can never pick an arbitrary model id, set an out-of-range hyperparameter, or write an arbitrary file. Hyperparameters are snapped onto fixed safe choice sets at the tool boundary.
- **Self-contained prompts.** The fine-tuned model is offline at inference — it sees only the prompt text, with no web, file, or tool access. Generated eval/training prompts must embed the full schema/data inline; a `$schema` URL is metadata inside an inlined schema, never a substitute for it.
- **Auditable.** Every run emits a structured event log. The eval set is hashed; expected answers never leave the host.
- **Degrades gracefully.** Every live service has a labeled fallback: heuristic planner, heuristic judge, heuristic retrain advisor, in-memory persistence. The mock path runs end-to-end with no keys and no GPU.
- **Optional autonomy.** Approval gates can be turned off for unattended, fully autonomous runs.

### Primary path vs. demo

The primary path is **goal-driven**: any plain-English goal drives eval generation, data synthesis, and training. A built-in **customer-spending-summary** demo task (with a strict-JSON objective scorer) exists for zero-config runs and tests.

---

## Technical Specifications

### Models

| Role | Model | Notes |
| --- | --- | --- |
| Model being specialized (default) | `unsloth/Phi-3-mini-4k-instruct` | 4-bit (`unsloth/phi-3-mini-4k-instruct-bnb-4bit`) on the VM |
| Alternate base (TUI dropdown) | `unsloth/Meta-Llama-3.1-8B-Instruct` | Llama 3.1 8B |
| Planner / eval & data generator / retrain advisor | `MiniMax-M2.7-highspeed` | OpenAI-compatible tool-calling at `https://api.minimax.io/v1` |
| LLM-as-judge | `llama3.3-70b-instruct` | via DigitalOcean inference (`https://inference.do-ai.run/v1/`) |
| Voice STT / TTS (optional) | `deepgram/nova-3` / `cartesia/sonic-3` | via LiveKit Inference |

The base model is selectable per run from the TUI; the chosen id is written into the VM config so the GPU trainer specializes the *selected* model, not a hardcoded default.

### The three bounded agent tools

MiniMax drives each iteration through OpenAI-style function calls, validated in `agent/tools.py`:

1. **`create_evals`** — declare the criteria a correct answer must satisfy and emit concrete `{prompt, expected_output}` pairs. Called once, first.
2. **`add_training_examples`** — synthesize targeted `{prompt, completion}` pairs for the observed failures. Called one or more times; never copies eval ground-truth.
3. **`start_training_model`** — choose LoRA hyperparameters from the allowed values and launch. Called once, last.

### LoRA hyperparameter choice sets (snapped at the tool boundary)

| Parameter | Allowed values |
| --- | --- |
| `learning_rate` | `1e-5, 2e-5, 5e-5, 1e-4, 2e-4` |
| `lora_rank_r` | `8, 16, 32, 64` |
| `lora_alpha` multiplier | `1, 2` (alpha = rank × multiplier) |
| `num_train_epochs` | `2, 3, 4, 5, 6` |
| `per_device_train_batch_size` | `1, 2, 4` |

Out-of-range or missing values are snapped onto the nearest allowed choice, so a misbehaving model can never push an unsafe configuration into training.

### Loop control (defaults)

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_iterations` | `3` | Hard cap on training rounds |
| `target_score` | `0.85` | Stop once the locked-eval score reaches this |
| `improvement_threshold` | `0.01` | Minimum gain before patience counts down |
| `patience` | `2` | Stop after this many low-improvement rounds |

### Evaluation & judging

- **Locked eval set** (`evaluation/locked.py`): the eval prompts/answers are canonicalized and **SHA-256 hashed**; any tampering halts the run.
- **Objective evaluator** (`evaluation/evaluator.py`): strict-JSON field scoring for the demo task.
- **LLM-as-judge** (`evaluation/llm_judge.py`): scores remote / dynamically generated evals where there is no hardcoded scorer.
- **DigitalOcean judge** (`evaluation/digitalocean_judge.py`): `llama3.3-70b-instruct`, with a labeled heuristic fallback.
- **Retrain advisor** (`orchestration/retrain_advisor.py`): MiniMax decides whether the judge result justifies another round; heuristic fallback when no key.

### Training backends

| Backend | Selector | Description |
| --- | --- | --- |
| Mock | `TRAINING_BACKEND=mock` (default) | End-to-end with no GPU, no keys, no network |
| Remote GPU | `TRAINING_BACKEND=remote` | Paramiko SSH/SFTP to a GPU VM; streams live train/eval logs |
| Unsloth | `TRAINING_BACKEND=unsloth` | In-process Unsloth boundary (optional `evolora[unsloth]` extra) |

**Remote VM flow.** EvoLoRA pushes three files to the VM — `config.json` (bare hyperparameter keys, including the selected `base_model_id`), `data/training_data.jsonl` (Alpaca-style), and `data/evals.json` (prompts only) — then runs `train.py` and `evaluate.py`, streaming output line-by-line over a PTY. The VM fills `"actual"` into `evals.json`, which is pulled back and scored by the LLM-judge. **Expected answers never leave the host**; only eval prompts plus the eval-set hash are pushed.

> VM note: `train.py` imports `unsloth` **before** `trl`/`transformers`/`peft`/`datasets` (Unsloth requires this to apply its patches) and uses TRL's `SFTConfig`/`SFTTrainer` with `dataset_text_field="text"`, `max_length`, and `dataset_num_proc=1`.

### Adapter archiving & chat

After each successful run, the trained `lora_model` is archived on the VM to `adapters/<specialty>-<id>` and becomes selectable for chat. Names **lead with the goal's distinguishing specialty** (generic "make a model that specializes in…" lead-ins are stripped) and end with a short, traceable run-id tag — e.g. `strict-json-schema-1a2b3c`, not `make-a-model-that-specializes-in-1a2b3c`. The TUI **CHAT** mode runs one-shot inference against any base model or archived adapter.

### Persistence & observability

- **MongoDB** run persistence (`motor`/`pymongo`) with an in-memory fallback.
- **Local artifact store** under the ignored `artifacts/`.
- **Per-run logs** (JSONL + readable `.log`) under the ignored `logs/`.

### Voice (optional, fully decoupled)

LiveKit Inference STT (`deepgram/nova-3`) + TTS (`cartesia/sonic-3`) provide dictation into the goal/chat box and spoken narration of run progress, with an optional global push-to-talk key (`pynput`, default `f9`). Voice is gated by `VOICE_ENABLED` and the `evolora[voice]` extra and never affects the training loop.

### Stack

Python ≥ 3.11. Core deps: `pydantic≥2.7`, `openai≥1.30` (MiniMax + DO via OpenAI-compatible SDK), `typer` (CLI), `textual` (TUI), `pymongo`/`motor` (persistence), `paramiko` (remote VM), `httpx`, `rich`, `python-dotenv`. Optional extras: `unsloth` (`unsloth`, `torch`, `transformers`, `datasets`, `trl`) and `voice` (`livekit-agents`, `sounddevice`, `pynput`).

---

## Quick Start

From this directory:

```powershell
uv sync
uv run evolora doctor
uv run evolora demo --mock --iterations 2
uv run evolora tui
```

Mock mode is the default path and is meant to work without `.env`, API keys, MongoDB, or a GPU. For real GPU training and live services, see [IMPLEMENT.md](IMPLEMENT.md) and `.env.example`.

## CLI Commands

| Command | Purpose |
| --- | --- |
| `uv run evolora` | Launch the TUI (default when no subcommand is given) |
| `uv run evolora tui` | Launch the Textual interface |
| `uv run evolora demo --mock --iterations 2` | Run a short mock demo loop |
| `uv run evolora doctor` | Check local configuration and mode |
| `uv run evolora smoke-minimax` | MiniMax connectivity smoke test (needs `MINIMAX_API_KEY`) |
| `uv run evolora history --limit 10` | Show recent persisted runs |
| `uv run pytest` | Run the test suite |
| `uv run ruff check .` | Run lint checks |

## Configuration

Secrets stay in `.env`, which is ignored by Git. Create it locally (see `.env.example`) when live services are needed.

| Variable | Purpose |
| --- | --- |
| `BASE_MODEL_ID` | Model being specialized; default `unsloth/Phi-3-mini-4k-instruct` |
| `MINIMAX_API_KEY` | Enables live MiniMax planning, eval/data generation, and retrain advice |
| `MINIMAX_MODEL` | Defaults to `MiniMax-M2.7-highspeed` |
| `MINIMAX_BASE_URL` | Defaults to `https://api.minimax.io/v1` |
| `MINIMAX_GROUP_ID` | MiniMax account group id |
| `DIGITAL_OCEAN_MODEL_ACCESS_KEY` | Enables the real DigitalOcean LLM judge |
| `DIGITAL_OCEAN_INFERENCE_BASE_URL` | Defaults to `https://inference.do-ai.run/v1/` |
| `DIGITAL_OCEAN_JUDGE_MODEL` | Defaults to `llama3.3-70b-instruct` |
| `MONGODB_URI` | Enables MongoDB Atlas run persistence |
| `MONGODB_DB_NAME` | Defaults to `evolora` |
| `MONGODB_RUNS_COLLECTION` | Defaults to `runs` |
| `MONGODB_SERVER_SELECTION_TIMEOUT_MS` | Defaults to `3000` |
| `TRAINING_BACKEND` | `mock` (default), `remote`, or `unsloth` |
| `MODEL_RUNNER` | `mock` (default), `local`, or `remote` (reads VM eval outputs) |
| `SSH_HOST`, `SSH_USER`, `SSH_PORT`, `SSH_KEY_PATH` | Remote GPU VM SFTP config/result exchange |
| `REMOTE_CONFIG_PATH` | Where EvoLoRA writes the VM config JSON |
| `REMOTE_RESULTS_PATH` | Where the VM writes eval outputs to score |
| `MAX_ITERATIONS` | Max training loop iterations (default `3`) |
| `TARGET_SCORE` | Score target for stopping (default `0.85`) |
| `IMPROVEMENT_THRESHOLD` | Minimum improvement before patience counts down (default `0.01`) |
| `PATIENCE` | Stop after repeated low-improvement iterations (default `2`) |
| `AUTO_APPROVE` | When true, skip the eval-set and keep-training gates → fully autonomous run |
| `VOICE_ENABLED` | Toggle the optional voice features (default on; needs LiveKit creds + `voice` extra) |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit Inference credentials for voice |
| `STT_MODEL`, `TTS_MODEL`, `TTS_VOICE`, `PTT_KEY` | Voice STT/TTS models, voice id, push-to-talk key |

Unattended/recorded runs can also pre-fill and auto-start the TUI via `EVOLORA_AUTOSTART`, `EVOLORA_GOAL`, and `EVOLORA_SAMPLES`.

## Repo Layout

```text
src/evolora/
  cli.py              Typer command entrypoint (TUI by default)
  config.py           Environment-backed configuration
  agent/              MiniMax planner, bounded tool schemas, heuristic fallback
  demo/               Customer-spending demo task and evals
  evaluation/         Locked/generic scoring, DigitalOcean judge, LLM-as-judge
  inference/          Inference package boundary
  models/             Pydantic run, plan, event, judge, and result models
  observability/      Per-run event logging and narration
  orchestration/      Main loop, stop conditions, approval gates, retrain advisor
  persistence/        In-memory store, MongoDB store, local artifacts
  training/           Mock backend, remote SSH/SFTP backend, unsloth boundary, adapter naming
  tui/                Native Textual app (live runs, base-model select, chat)
  voice/              Optional LiveKit dictation + narrator (decoupled from training)
src/virtual_machine_code/   GPU VM scripts: train.py, evaluate.py, chat.py, config.json, data/
scripts/vm/                 Baseline/eval helpers pushed to the GPU VM
tests/              Unit and integration tests
docs/               Submission, architecture, and demo-video guide
artifacts/          Runtime outputs, ignored by Git
logs/               Per-run logs, ignored by Git
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

The full loop runs goal-driven: MiniMax planning, locked evaluation with adaptive difficulty, planner/judge/advisor fallbacks, validated and stacked training examples, judged iterations, TUI progress with cancellation, optional autonomous mode, post-run chat against archived adapters, and a CLI fallback. Both the mock path (no keys, no GPU) and the remote GPU path (Unsloth on a VM over SSH) are exercised.

Next useful work: artifact checksums for real adapters, deeper MongoDB history views, and continued polish of the demo docs.
