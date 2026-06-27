"""MiniMax-generated evals: generic scoring, JSON extraction, and orchestrator wiring."""

import json

import pytest

from evolora.agent.planner import MiniMaxPlanner, _extract_json
from evolora.evaluation.evaluator import GenericEvaluator, generic_evaluate_sample
from evolora.evaluation.locked import LockedEvalSet
from evolora.models.core import (
    AgentPlan,
    EvalSample,
    LoraHyperparams,
    RunConfig,
    TrainingDataSpec,
)
from evolora.orchestration.orchestrator import Orchestrator


def _sample(expected):
    return EvalSample(sample_id="s1", prompt="p", expected=expected)


def test_generic_exact_match():
    r = generic_evaluate_sample(_sample({"label": "urgent", "score": 5}),
                                json.dumps({"label": "urgent", "score": 5}))
    assert r.score == 1.0 and r.passed


def test_generic_partial_and_case_insensitive():
    r = generic_evaluate_sample(_sample({"label": "urgent", "score": 5}),
                                json.dumps({"label": "URGENT", "score": 9}))
    assert r.score == 0.5 and not r.passed  # label matches, score wrong


def test_generic_invalid_json_scores_zero():
    r = generic_evaluate_sample(_sample({"a": 1}), "not json at all")
    assert r.score == 0.0 and not r.passed


def test_generic_empty_expected_requires_valid_json():
    assert generic_evaluate_sample(_sample({}), json.dumps({"x": 1})).score == 1.0
    assert generic_evaluate_sample(_sample({}), "nope").score == 0.0


def test_generic_numeric_tolerance():
    assert generic_evaluate_sample(_sample({"total": 100.0}),
                                   json.dumps({"total": 100.005})).passed


def test_generic_evaluator_aggregates():
    samples = [
        EvalSample(sample_id="a", prompt="p", expected={"k": "v"}),
        EvalSample(sample_id="b", prompt="p", expected={"k": "v"}),
    ]
    responses = {"a": json.dumps({"k": "v"}), "b": json.dumps({"k": "x"})}
    score, results = GenericEvaluator()(samples, responses)
    assert score == 0.5 and len(results) == 2


def test_extract_json_handles_fences_and_think():
    raw = '<think>reasoning</think>```json\n[{"prompt":"a","expected":{"b":1}}]\n```'
    data = _extract_json(raw)
    assert data[0]["expected"]["b"] == 1


class _FakePlanner(MiniMaxPlanner):
    """MiniMaxPlanner with network calls stubbed out."""

    def __init__(self):
        super().__init__(api_key="x", model="m", base_url="u")

    async def generate_evals(self, goal, count=5):
        return [
            {"prompt": "ticket: server down", "expected": {"urgency": "high"}},
            {"prompt": "ticket: minor typo", "expected": {"urgency": "low"}},
        ]

    async def plan(self, *args, **kwargs):
        return (
            AgentPlan(
                hyperparams=LoraHyperparams(),
                data_spec=TrainingDataSpec(examples=[{"prompt": "x", "completion": "y"}]),
            ),
            False,
        )


@pytest.mark.asyncio
async def test_orchestrator_uses_minimax_generated_evals():
    cfg = RunConfig(
        goal="classify ticket urgency",
        max_iterations=1,
        target_score=1.0,
        training_sample_count=None,
    )
    demo = LockedEvalSet([EvalSample(sample_id="demo", prompt="p", expected={"x": 1})])
    orch = Orchestrator(config=cfg, eval_set=demo, planner=_FakePlanner())

    async for _ in await orch.run():
        pass

    # The demo eval set was replaced by the 2 MiniMax-generated examples,
    # and scoring switched to the generic evaluator.
    assert len(orch._eval_set) == 2
    assert orch._eval_set.samples[0].sample_id.startswith("gen-")
    assert isinstance(orch._evaluator, GenericEvaluator)
