"""Tests for DigitalOcean judge fallback behavior."""

from __future__ import annotations

import pytest

from evolora.evaluation.digitalocean_judge import HeuristicJudge, get_judge
from evolora.models.core import AgentPlan, EvalResult


@pytest.mark.asyncio
async def test_get_judge_without_key_uses_heuristic_fallback():
    judge = get_judge(api_key="")

    assert isinstance(judge, HeuristicJudge)
    assert judge.is_mock is True


@pytest.mark.asyncio
async def test_heuristic_judge_rates_from_objective_scores():
    judge = HeuristicJudge()
    report = await judge.judge(
        goal="classify urgent tickets",
        task_name="ticket_classifier",
        base_model_id="microsoft/Phi-3-mini-128k-instruct",
        iteration=1,
        score=0.5,
        adaptive_score=0.25,
        plan=AgentPlan(),
        eval_results=[
            EvalResult(sample_id="case-1", score=0.0, passed=False, details={"error": "bad JSON"}),
            EvalResult(sample_id="case-2", score=1.0, passed=True, details={}),
        ],
        responses={"case-1": "not json", "case-2": '{"ok": true}'},
    )

    assert report.is_mock is True
    assert report.source == "heuristic_do_judge_fallback"
    assert report.rating == pytest.approx(0.4375)
    assert report.weaknesses
