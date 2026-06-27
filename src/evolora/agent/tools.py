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
                "Create the evaluation set for this run. Call once, first: declare the criteria a "
                "correct answer must satisfy AND generate concrete eval examples (each a prompt "
                "plus the exact correct JSON output). These examples become the objective eval "
                "set the run is scored against."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short bullet criteria a good model output must satisfy.",
                    },
                    "eval_examples": {
                        "type": "array",
                        "description": "Concrete evaluation examples used to objectively score the model.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string"},
                                "expected_output": {
                                    "type": "object",
                                    "description": "The exact correct JSON object the model should output.",
                                },
                            },
                            "required": ["prompt", "expected_output"],
                        },
                    },
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
                "Contribute targeted prompt/completion training pairs as one JSON payload aimed "
                "at the observed failures. Call one or more times. Never copy the evaluation "
                "ground-truth answers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "training_json": {
                        "type": "object",
                        "description": (
                            "JSON parameters for this training-data batch. Must include "
                            "examples and may include rationale or metadata."
                        ),
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
                            "metadata": {
                                "type": "object",
                                "description": "Optional JSON metadata for traceability.",
                            },
                        },
                        "required": ["examples"],
                    },
                    "examples": {
                        "type": "array",
                        "description": "Backward-compatible flat examples array.",
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
                        "description": "Backward-compatible flat rationale string.",
                    },
                },
                "required": ["training_json"],
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


def extract_training_payload(args: dict) -> tuple[list[dict[str, str]], str]:
    """Extract training examples from the add_training_examples JSON parameters.

    The preferred tool shape is {"training_json": {"examples": [...], "rationale": "..."}},
    but we keep the older flat {"examples": [...], "rationale": "..."} shape working so
    in-flight model responses and tests remain compatible.
    """
    payload = args.get("training_json")
    if not isinstance(payload, dict):
        payload = args

    raw_examples = payload.get("examples", [])
    accepted = [
        {"prompt": str(example["prompt"]), "completion": str(example["completion"])}
        for example in raw_examples
        if isinstance(example, dict) and example.get("prompt") and example.get("completion")
    ]
    rationale = payload.get("rationale") or args.get("rationale") or ""
    return accepted, str(rationale)


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
