"""The user's goal/use-case threads from RunConfig through to the planner."""

import json

from evolora.agent.planner import HeuristicPlanner, MiniMaxPlanner
from evolora.models.core import EvalResult, RunConfig


def _failures():
    return [EvalResult(sample_id="x", score=0.3, passed=False, details={})]


def test_runconfig_carries_goal():
    assert RunConfig(goal="classify tickets").goal == "classify tickets"
    assert RunConfig().goal == ""  # additive default


def test_heuristic_plan_mentions_goal():
    plan = HeuristicPlanner().plan(1, 0.5, 0.5, _failures(), goal="summarize invoices")
    assert "summarize invoices" in plan.rationale


def test_user_goal_reaches_planner_prompt():
    p = MiniMaxPlanner(api_key="x", model="m", base_url="u")
    data = json.loads(p._build_user_prompt(1, 0.5, 0.5, _failures(), 5, "build a tagger"))
    assert data["user_goal"] == "build a tagger"


def test_empty_goal_is_none_in_prompt():
    p = MiniMaxPlanner(api_key="x", model="m", base_url="u")
    data = json.loads(p._build_user_prompt(1, 0.5, 0.5, _failures(), None, ""))
    assert data["user_goal"] is None
