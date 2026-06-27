"""Tests for planners — heuristic and MiniMax fallback."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from evolora.agent.planner import (
    MINIMAX_TOOL_MAX_TOKENS,
    HeuristicPlanner,
    MiniMaxPlanner,
    _parse_plan,
    get_planner,
)
from evolora.models.core import EvalResult


def _failures():
    return [EvalResult(sample_id="x", score=0.3, passed=False, details={})]


def test_heuristic_plan_valid():
    p = HeuristicPlanner()
    plan = p.plan(1, 0.5, 0.5, _failures())
    plan.hyperparams  # should not raise
    assert plan.hyperparams.r in {1, 2, 4, 8, 16, 32, 64}


def test_heuristic_plan_increases_r_over_iterations():
    p = HeuristicPlanner()
    plan1 = p.plan(1, 0.5, 0.5, _failures())
    plan3 = p.plan(3, 0.5, 0.5, _failures())
    assert plan3.hyperparams.r >= plan1.hyperparams.r


def test_get_planner_defaults_to_heuristic_without_key():
    planner = get_planner(use_minimax=True)
    assert isinstance(planner, HeuristicPlanner)


def test_parse_plan_valid_json():
    raw = json.dumps({
        "hyperparams": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                        "learning_rate": 0.0002, "num_epochs": 1,
                        "batch_size": 4, "warmup_steps": 10, "weight_decay": 0.01},
        "data_spec": {"examples": [{"prompt": "a", "completion": "b"}],
                      "rationale": "test", "max_examples": 50},
        "rationale": "test plan",
        "focus_areas": ["json_format"],
    })
    plan = _parse_plan(raw)
    assert plan.hyperparams.r == 8


def test_parse_plan_strips_think_tags():
    inner = json.dumps({
        "hyperparams": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.05,
                        "learning_rate": 0.0002, "num_epochs": 1,
                        "batch_size": 4, "warmup_steps": 10, "weight_decay": 0.01},
        "data_spec": {"examples": [], "rationale": "", "max_examples": 50},
        "rationale": "ok", "focus_areas": [],
    })
    raw = f"<think>internal thoughts</think>{inner}"
    plan = _parse_plan(raw)
    assert plan.hyperparams.r == 4


def test_parse_plan_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_plan("not json at all")


def test_hyperparams_invalid_r_raises():
    with pytest.raises(ValidationError):
        _parse_plan(json.dumps({
            "hyperparams": {"r": 7, "lora_alpha": 16, "lora_dropout": 0.05,
                            "learning_rate": 0.0002, "num_epochs": 1,
                            "batch_size": 4, "warmup_steps": 10, "weight_decay": 0.01},
            "data_spec": {"examples": [], "rationale": "", "max_examples": 50},
            "rationale": "", "focus_areas": [],
        }))


def _tool_response(name: str, args: str, *, finish_reason: str = "tool_calls"):
    tool_call = SimpleNamespace(
        id=f"call-{name}",
        function=SimpleNamespace(name=name, arguments=args),
    )
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("unexpected MiniMax call")
        return self._responses.pop(0)


class _FakeMiniMaxClient:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


class _StubMiniMaxPlanner(MiniMaxPlanner):
    def __init__(self, responses):
        super().__init__(api_key="x", model="m", base_url="u")
        self.client = _FakeMiniMaxClient(responses)

    def _make_client(self):
        return self.client


@pytest.mark.asyncio
async def test_minimax_plan_retries_after_truncated_training_examples_tool_call():
    create_args = json.dumps({"criteria": ["sql correctness"]})
    truncated_add_args = '{"training_json": '
    valid_add_args = json.dumps({
        "training_json": {
            "rationale": "sql examples",
            "examples": [
                {
                    "prompt": "Write SQL to count orders.",
                    "completion": '{"sql":"SELECT COUNT(*) FROM orders"}',
                },
                {
                    "prompt": "Write SQL to list customers.",
                    "completion": '{"sql":"SELECT * FROM customers"}',
                },
            ],
        }
    })
    start_args = json.dumps({
        "learning_rate": 1e-4,
        "lora_rank_r": 16,
        "lora_alpha_multiplier": 2,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
    })
    planner = _StubMiniMaxPlanner([
        _tool_response("create_evals", create_args),
        _tool_response("add_training_examples", truncated_add_args, finish_reason="length"),
        _tool_response("create_evals", create_args),
        _tool_response("add_training_examples", valid_add_args),
        _tool_response("start_training_model", start_args),
    ])

    plan, fallback = await planner.plan(
        1,
        0.8,
        0.8,
        [],
        2,
        "make a model that specializes in writing sql queries",
    )

    assert fallback is False
    assert len(plan.data_spec.examples) == 2
    assert "SQL" in plan.data_spec.examples[0]["prompt"]
    assert plan.hyperparams.r == 16
    assert len(planner.client.completions.calls) == 5
    assert all(call["max_tokens"] == MINIMAX_TOOL_MAX_TOKENS for call in planner.client.completions.calls)
    first_retry_prompt = planner.client.completions.calls[2]["messages"][1]["content"]
    assert "at most 5 examples per call" in first_retry_prompt


@pytest.mark.asyncio
async def test_minimax_plan_records_fallback_reason_after_retries_fail():
    create_args = json.dumps({"criteria": ["sql correctness"]})
    truncated_add_args = '{"training_json": '
    planner = _StubMiniMaxPlanner([
        _tool_response("create_evals", create_args),
        _tool_response("add_training_examples", truncated_add_args, finish_reason="length"),
        _tool_response("create_evals", create_args),
        _tool_response("add_training_examples", truncated_add_args, finish_reason="length"),
    ])

    plan, fallback = await planner.plan(
        1,
        0.8,
        0.8,
        [],
        2,
        "make a model that specializes in writing sql queries",
    )

    assert fallback is True
    assert "invalid JSON" in planner.last_error
    assert "Heuristic plan" in plan.rationale
