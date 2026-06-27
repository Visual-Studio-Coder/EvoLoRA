"""MiniMax agent tools: bounded create_evals / add_training_examples / start_training_model.

The MiniMax planner drives each training iteration by calling these three tools. Python
validates and controls everything the model proposes — the model never writes files, picks
arbitrary model ids, or sets out-of-range hyperparameters. LoRA hyperparameters are
restricted to fixed safe choice sets at the tool-schema boundary and re-snapped onto those
sets here before they can reach a training backend.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Allowed hyperparameter choices (from the agent design).
# ---------------------------------------------------------------------------

LEARNING_RATES: list[float] = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
LORA_RANKS: list[int] = [8, 16, 32, 64]
ALPHA_MULTIPLIERS: list[int] = [1, 2]  # lora_alpha = lora_rank_r * multiplier
NUM_TRAIN_EPOCHS: list[int] = [2, 3, 4, 5, 6]
BATCH_SIZES: list[int] = [1, 2, 4]


# ---------------------------------------------------------------------------
# OpenAI-compatible tool (function-calling) schemas.
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "create_evals",
            "description": (
                "Declare what a correct answer looks like for this run. Use it once, first, to "
                "set the evaluation focus before generating training data. This records the "
                "criteria the agent is optimizing toward; it does not alter the locked objective "
                "scorer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short bullet criteria a good model output must satisfy.",
                    }
                },
                "required": ["criteria"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_training_examples",
            "description": (
                "Contribute targeted prompt/completion training pairs aimed at the observed "
                "failures. Call one or more times. Never copy the evaluation ground-truth answers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "examples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "completion": {"type": "string"},
                            },
                            "required": ["prompt", "completion"],
                        },
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why these examples address the current failures.",
                    },
                },
                "required": ["examples"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_training_model",
            "description": (
                "Choose LoRA hyperparameters from the allowed values and launch training. Call "
                "exactly once, last, after adding training examples."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "learning_rate": {"type": "number", "enum": LEARNING_RATES},
                    "lora_rank_r": {"type": "integer", "enum": LORA_RANKS},
                    "lora_alpha_multiplier": {
                        "type": "integer",
                        "enum": ALPHA_MULTIPLIERS,
                        "description": "lora_alpha = lora_rank_r * this (1 = same as rank, 2 = double).",
                    },
                    "num_train_epochs": {"type": "integer", "enum": NUM_TRAIN_EPOCHS},
                    "per_device_train_batch_size": {"type": "integer", "enum": BATCH_SIZES},
                },
                "required": [
                    "learning_rate",
                    "lora_rank_r",
                    "lora_alpha_multiplier",
                    "num_train_epochs",
                    "per_device_train_batch_size",
                ],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Bounded coercion — never trust the model to stay inside the choice sets.
# ---------------------------------------------------------------------------


def _nearest(value: float, allowed: list) -> object:
    """Snap an arbitrary value onto the closest allowed choice."""
    return min(allowed, key=lambda a: abs(a - value))


def coerce_hyperparams(args: dict) -> dict:
    """Snap start_training_model arguments onto the allowed choice sets.

    Returns a dict of LoraHyperparams fields (r, lora_alpha, learning_rate,
    num_epochs, batch_size). Out-of-range or missing values fall back to the
    nearest allowed value (or a sensible default), so a misbehaving model can
    never push an unsafe configuration into training.
    """
    try:
        r = int(_nearest(float(args.get("lora_rank_r", 8)), LORA_RANKS))
    except (TypeError, ValueError):
        r = 8

    mult = args.get("lora_alpha_multiplier", 2)
    mult = mult if mult in ALPHA_MULTIPLIERS else 2

    try:
        lr = float(_nearest(float(args.get("learning_rate", 2e-4)), LEARNING_RATES))
    except (TypeError, ValueError):
        lr = 2e-4

    try:
        epochs = int(_nearest(float(args.get("num_train_epochs", 3)), NUM_TRAIN_EPOCHS))
    except (TypeError, ValueError):
        epochs = 3

    try:
        batch = int(_nearest(float(args.get("per_device_train_batch_size", 4)), BATCH_SIZES))
    except (TypeError, ValueError):
        batch = 4

    return {
        "r": r,
        "lora_alpha": r * mult,
        "learning_rate": lr,
        "num_epochs": epochs,
        "batch_size": batch,
    }
