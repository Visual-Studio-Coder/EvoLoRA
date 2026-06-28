# EvoLoRA — Setup & Run Guide

EvoLoRA is an auditable, bounded self-improvement loop for LoRA fine-tuning: you give it a
plain-English goal, a MiniMax agent plans the eval set + training data + hyperparameters, a
small model (Phi-3-mini by default) is trained and scored on a remote GPU, an LLM judges the
result, and it iterates. This guide gets a fresh machine running.

---

## 1. Prerequisites

- **Python 3.11+** (3.13 recommended).
- **[uv](https://docs.astral.sh/uv/)** — the package manager. Install:
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`
- **git**, and a terminal (Windows Terminal / iTerm / etc.).
- For **real training**: SSH access to the GPU pod (key file + host/port — your teammate sends these).

## 2. Get the code

```bash
git clone https://github.com/Visual-Studio-Coder/EvoLoRA.git
cd EvoLoRA
uv sync          # creates .venv and installs deps
```

## 3. Configure secrets (`.env`)

```bash
cp .env.example .env
```

Then fill in `.env`. **The real key values are NOT in the repo** — your teammate sends them
to you over a secure channel (password manager / encrypted DM), never over git or plain chat.

You need:

| Key | What it's for | Where to get it |
|-----|---------------|-----------------|
| `MINIMAX_API_KEY` + `MINIMAX_GROUP_ID` | the planning agent | platform.minimax.io |
| `DIGITAL_OCEAN_MODEL_ACCESS_KEY` | the LLM-as-a-judge | DigitalOcean inference |
| `SSH_HOST` / `SSH_PORT` / `SSH_USER` / `SSH_KEY_PATH` | the GPU pod for real training | from your teammate |
| `MONGODB_URI` | optional run history | optional (leave blank) |

**SSH key file:** your teammate sends you a private key file (e.g. `id_ed25519_pi_usb`). Save it
somewhere **outside the repo** (e.g. `~/.ssh/`), `chmod 600` it on macOS/Linux, and point
`SSH_KEY_PATH` at it.

Key settings:
- `TRAINING_BACKEND=remote` → real GPU training. `=mock` → no-GPU demo loop (no keys needed).
- `BASE_MODEL_ID=unsloth/Phi-3-mini-4k-instruct` → the verified base model.
- `AUTO_APPROVE=true` → fully autonomous (no prompts); `=false` → the TUI asks for approvals.

## 4. Run it

```bash
uv run evolora          # launches the TUI
# or, if installed on PATH:
evolora
```

Check your setup first:
```bash
uv run evolora doctor   # shows which keys/services are detected
```

In the TUI:
1. Pick a base model in the **LORA CONFIG** dropdown (Phi-3-mini = fast).
2. Type a goal, e.g. `Make a model that specializes in writing SQL queries`.
3. Set **# samples** (30–100 is quick; 1000 is a long run).
4. Press **START**.
5. Use the **CHAT** toggle (after a real run) to talk to your trained adapter.

Keyboard: `Ctrl+R` start · `Ctrl+X` cancel · `Ctrl+Y` copy the agent log · `q` quit.

## 5. Autonomous (unattended) runs

Set `AUTO_APPROVE=true` in `.env`. The loop then runs with **no approval prompts** — it
auto-approves the generated eval set and keeps iterating on its own until the judge is
satisfied / the target score is hit / max iterations, then stops. Good for leaving it running.

## 6. How long a run takes

- **Planning** scales with sample count (~12 MiniMax rounds for 200, ~44 for 1000) — minutes.
- **Training** on the GPU scales with examples × epochs. 30 examples ≈ seconds; 1000 ≈ many minutes.
- A real 1000-sample run can be **~40–60 min** end to end. That's expected.

## 7. Troubleshooting

- `evolora doctor` flags missing keys.
- **Run failed at training** → check `logs/run-*.log` (every run is logged there, gitignored).
- **`ModuleNotFoundError: livekit`** → voice is optional; set `VOICE_ENABLED=false` (default).
- **Wrong model trained / crash in `fix_untrained_tokens`** → make sure `BASE_MODEL_ID` is set
  and you're on the latest `main` (the model is read from `config.json` per run).
- **MiniMax falls back to heuristic** → usually a huge sample count or a transient blip; the
  planner scales rounds + retries, but very large counts take longer.

## 8. Tests

```bash
uv run pytest -q
uv run ruff check .
```
