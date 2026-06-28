"""Tests for Pydantic models and validation."""

import pytest
from pydantic import ValidationError

from evolora.models.core import (
    AgentPlan,
    LoraHyperparams,
    RunConfig,
    RunRecord,
    TrainingDataSpec,
)


def test_lora_r_must_be_power_of_two():
    with pytest.raises(ValidationError):
        LoraHyperparams(r=7)


def test_lora_r_valid():
    hp = LoraHyperparams(r=16)
    assert hp.r == 16


def test_lora_learning_rate_bounds():
    with pytest.raises(ValidationError):
        LoraHyperparams(learning_rate=0.5)


def test_training_data_deduplication():
    ex = {"prompt": "a", "completion": "b"}
    spec = TrainingDataSpec(examples=[ex, ex, ex])
    assert len(spec.examples) == 1


def test_training_data_capped():
    examples = [{"prompt": str(i), "completion": str(i)} for i in range(300)]
    spec = TrainingDataSpec(examples=examples, max_examples=50)
    assert len(spec.examples) == 50


def test_agent_plan_no_path_traversal():
    with pytest.raises(ValidationError):
        AgentPlan(target_adapter_name="../evil")


def test_run_record_is_not_terminal_initially():
    cfg = RunConfig()
    rec = RunRecord(config=cfg)
    assert not rec.is_terminal


def test_training_sample_count_bounds():
    assert RunConfig().training_sample_count == 30
    assert RunConfig(training_sample_count=25).training_sample_count == 25
    assert RunConfig(training_sample_count=5000).training_sample_count == 5000  # 4+ digits allowed
    assert RunConfig(training_sample_count=None).training_sample_count is None
    with pytest.raises(ValidationError):
        RunConfig(training_sample_count=0)
    with pytest.raises(ValidationError):
        RunConfig(training_sample_count=100001)


def test_no_improvement_count_zero_initially():
    cfg = RunConfig()
    rec = RunRecord(config=cfg)
    assert rec.no_improvement_count() == 0
