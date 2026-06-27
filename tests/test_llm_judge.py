"""LLM-as-a-judge evaluator: canonical {input, expected} format, scoring, aggregation."""

import pytest

from evolora.evaluation.llm_judge import (
    LLMJudgeEvaluator,
    _coerce_score,
    _parse_json,
    make_eval_records,
)


def test_make_eval_records_is_two_field_input_expected():
    assert make_eval_records([{"input": "q", "expected": "a"}]) == [
        {"input": "q", "expected": "a"}
    ]


def test_make_eval_records_accepts_legacy_prompt_and_dict_expected():
    recs = make_eval_records([{"prompt": "q", "expected": {"k": 1}}])
    assert recs[0]["input"] == "q"
    assert recs[0]["expected"] == '{"k": 1}'  # dict json-stringified
    assert set(recs[0]) == {"input", "expected"}  # only the two file fields


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
    agg, recs = await LLMJudgeEvaluator(api_key="").judge([{"input": "q", "expected": "a"}])
    assert agg == 0.0 and "score" not in recs[0]


@pytest.mark.asyncio
async def test_judge_empty_actual_scores_zero():
    j = _StubJudge({"good": (10, "perfect")})
    agg, out = await j.judge([{"input": "q", "expected": "a", "actual": ""}])
    assert out[0]["score"] == 0 and "no model output" in out[0]["reason"]
    assert agg == 0.0


class _StubJudge(LLMJudgeEvaluator):
    """Judge with the network call stubbed; maps actual -> (score, reason)."""

    def __init__(self, by_actual: dict):
        super().__init__(api_key="x")
        self._by_actual = by_actual

    async def _score(self, client, record):
        return self._by_actual.get(record["actual"], (5, "default"))


@pytest.mark.asyncio
async def test_judge_fills_score_reason_and_aggregates_0to1():
    j = _StubJudge({"good": (10, "match"), "bad": (0, "wrong")})
    recs = [
        {"input": "q1", "expected": "e1", "actual": "good"},
        {"input": "q2", "expected": "e2", "actual": "bad"},
    ]
    agg, out = await j.judge(recs)
    assert out[0]["score"] == 10 and out[0]["reason"] == "match"
    assert out[1]["score"] == 0
    assert agg == 0.5  # mean(10, 0) / 10
    assert set(out[0]) >= {"input", "expected", "actual", "score", "reason"}
