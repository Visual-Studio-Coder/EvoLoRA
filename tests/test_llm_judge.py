"""LLM-as-a-judge evaluator: canonical record format, scoring, aggregation, fallback."""

import pytest

from evolora.evaluation.llm_judge import (
    LLMJudgeEvaluator,
    _coerce_score,
    _parse_json,
    make_eval_records,
)


def test_make_eval_records_from_agent_shape():
    recs = make_eval_records([{"input": "q", "expected_output": "a"}])
    assert recs == [
        {"input": "q", "expected_output": "a", "actual_output": "", "score": None, "reason": ""}
    ]


def test_make_eval_records_accepts_legacy_prompt_expected_dict():
    recs = make_eval_records([{"prompt": "q", "expected": {"k": 1}}])
    assert recs[0]["input"] == "q"
    assert recs[0]["expected_output"] == '{"k": 1}'  # dict json-stringified
    assert recs[0]["actual_output"] == "" and recs[0]["score"] is None


def test_parse_json_strips_fences_and_think():
    assert _parse_json('<think>x</think>```json\n{"score": 8, "reason": "ok"}\n```') == {
        "score": 8,
        "reason": "ok",
    }
    assert _parse_json("prefix {\"score\": 3} suffix")["score"] == 3
    assert _parse_json("not json") == {}


def test_coerce_score_clamps_and_handles_garbage():
    assert _coerce_score(7) == 7
    assert _coerce_score("9") == 9
    assert _coerce_score(99) == 10
    assert _coerce_score(-4) == 0
    assert _coerce_score("nope") is None


@pytest.mark.asyncio
async def test_judge_unconfigured_returns_records_unchanged():
    agg, recs = await LLMJudgeEvaluator(api_key="").judge(
        make_eval_records([{"input": "q", "expected_output": "a"}])
    )
    assert agg == 0.0 and recs[0]["score"] is None


@pytest.mark.asyncio
async def test_judge_empty_actual_scores_zero():
    j = _StubJudge({"any": (10, "perfect")})
    recs = make_eval_records([{"input": "q", "expected_output": "a"}])  # actual_output empty
    agg, out = await j.judge(recs)
    assert out[0]["score"] == 0 and "no model output" in out[0]["reason"]
    assert agg == 0.0


class _StubJudge(LLMJudgeEvaluator):
    """Judge with the network call stubbed; maps actual_output -> (score, reason)."""

    def __init__(self, by_actual: dict):
        super().__init__(api_key="x")
        self._by_actual = by_actual

    async def _score(self, client, record):
        return self._by_actual.get(record["actual_output"], (5, "default"))


@pytest.mark.asyncio
async def test_judge_fills_scores_and_aggregates_0to1():
    j = _StubJudge({"good": (10, "match"), "bad": (0, "wrong")})
    recs = [
        {"input": "q1", "expected_output": "e1", "actual_output": "good", "score": None, "reason": ""},
        {"input": "q2", "expected_output": "e2", "actual_output": "bad", "score": None, "reason": ""},
    ]
    agg, out = await j.judge(recs)
    assert out[0]["score"] == 10 and out[0]["reason"] == "match"
    assert out[1]["score"] == 0
    assert agg == 0.5  # mean(10, 0) / 10
