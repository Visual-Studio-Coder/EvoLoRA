"""Tests for the MiniMax agent tool definitions and bounded hyperparameter coercion."""

from evolora.agent.tools import (
    ALPHA_MULTIPLIERS,
    BATCH_SIZES,
    LEARNING_RATES,
    LORA_RANKS,
    NUM_TRAIN_EPOCHS,
    TOOLS,
    coerce_hyperparams,
    extract_training_payload,
)
from evolora.models.core import LoraHyperparams


def test_three_tools_exposed():
    names = {t["function"]["name"] for t in TOOLS}
    assert names == {"create_evals", "add_training_examples", "start_training_model"}
    assert all(t["type"] == "function" for t in TOOLS)


def test_add_training_examples_accepts_json_parameters():
    schema = next(
        t["function"]["parameters"]
        for t in TOOLS
        if t["function"]["name"] == "add_training_examples"
    )
    assert schema["required"] == ["training_json"]
    training_json = schema["properties"]["training_json"]
    assert training_json["type"] == "object"
    assert training_json["required"] == ["examples"]
    assert training_json["properties"]["examples"]["type"] == "array"


def test_start_training_schema_uses_choice_enums():
    props = next(
        t["function"]["parameters"]["properties"]
        for t in TOOLS
        if t["function"]["name"] == "start_training_model"
    )
    assert props["learning_rate"]["enum"] == LEARNING_RATES
    assert props["lora_rank_r"]["enum"] == LORA_RANKS
    assert props["num_train_epochs"]["enum"] == NUM_TRAIN_EPOCHS
    assert props["per_device_train_batch_size"]["enum"] == BATCH_SIZES
    assert props["lora_alpha_multiplier"]["enum"] == ALPHA_MULTIPLIERS


def test_extract_training_payload_prefers_training_json():
    examples, rationale = extract_training_payload(
        {
            "training_json": {
                "examples": [{"prompt": "ticket: outage", "completion": '{"urgency":"high"}'}],
                "rationale": "covers outage urgency",
                "metadata": {"source": "test"},
            },
            "examples": [{"prompt": "old", "completion": "old"}],
        }
    )

    assert examples == [{"prompt": "ticket: outage", "completion": '{"urgency":"high"}'}]
    assert rationale == "covers outage urgency"


def test_extract_training_payload_wraps_raw_sql_completion_as_json():
    examples, _ = extract_training_payload(
        {
            "training_json": {
                "examples": [
                    {
                        "prompt": "Write a SQL query to count orders.",
                        "completion": "SELECT COUNT(*) FROM orders",
                    }
                ]
            }
        }
    )

    assert examples == [
        {
            "prompt": "Write a SQL query to count orders.",
            "completion": '{"sql": "SELECT COUNT(*) FROM orders"}',
        }
    ]


def test_extract_training_payload_keeps_flat_shape_compatible():
    examples, rationale = extract_training_payload(
        {
            "examples": [{"prompt": "customer data", "completion": '{"ok":true}'}],
            "rationale": "legacy shape",
        }
    )

    assert examples == [{"prompt": "customer data", "completion": '{"ok":true}'}]
    assert rationale == "legacy shape"


def test_coerce_valid_choices_pass_through():
    hp = coerce_hyperparams(
        {
            "learning_rate": 5e-5,
            "lora_rank_r": 32,
            "lora_alpha_multiplier": 2,
            "num_train_epochs": 6,
            "per_device_train_batch_size": 2,
        }
    )
    assert hp == {
        "r": 32,
        "lora_alpha": 64,  # r * multiplier
        "learning_rate": 5e-5,
        "num_epochs": 6,
        "batch_size": 2,
    }


def test_coerce_snaps_out_of_range_onto_allowed_sets():
    hp = coerce_hyperparams(
        {
            "learning_rate": 0.5,  # absurd -> nearest allowed (2e-4)
            "lora_rank_r": 7,  # -> 8
            "lora_alpha_multiplier": 9,  # invalid -> default 2
            "num_train_epochs": 99,  # -> 6
            "per_device_train_batch_size": 3,  # -> 2 or 4 (nearest)
        }
    )
    assert hp["r"] in LORA_RANKS
    assert hp["learning_rate"] in LEARNING_RATES
    assert hp["num_epochs"] in NUM_TRAIN_EPOCHS
    assert hp["batch_size"] in BATCH_SIZES
    assert hp["lora_alpha"] == hp["r"] * 2  # multiplier fell back to 2


def test_coerce_handles_missing_and_garbage_args():
    hp = coerce_hyperparams({"learning_rate": "not-a-number", "lora_rank_r": None})
    # falls back to safe defaults, never raises
    assert hp["r"] in LORA_RANKS
    assert hp["learning_rate"] in LEARNING_RATES


def test_coerced_hyperparams_build_valid_lora_model():
    for epochs in NUM_TRAIN_EPOCHS:
        hp = coerce_hyperparams(
            {
                "learning_rate": 1e-4,
                "lora_rank_r": 64,
                "lora_alpha_multiplier": 1,
                "num_train_epochs": epochs,
                "per_device_train_batch_size": 4,
            }
        )
        model = LoraHyperparams(**hp)  # must not raise (epochs up to 6 allowed)
        assert model.num_epochs == epochs
        assert model.lora_alpha == 64
