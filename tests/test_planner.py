"""Tests for planners — heuristic and MiniMax fallback."""

import json

import pytest
from pydantic import ValidationError

from evolora.agent.planner import HeuristicPlanner, _parse_plan, get_planner
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
