"""Tests for the locked eval set and objective evaluator."""

import json
import pytest

from evolora.evaluation.evaluator import evaluate_sample
from evolora.evaluation.locked import LockedEvalSet
from evolora.models.core import EvalSample


def _sample():
    return EvalSample(
        sample_id="t1",
        prompt="Summarize.",
        expected={
            "top_customer": "Alice",
            "top_customer_total": 300.0,
            "customer_count": 2,
            "total_revenue": 400.0,
            "summary": "Alice leads.",
        },
    )


def test_eval_perfect_score():
    s = _sample()
    response = json.dumps({
        "top_customer": "Alice",
        "top_customer_total": 300.0,
        "customer_count": 2,
        "total_revenue": 400.0,
        "summary": "Alice leads.",
    })
    result = evaluate_sample(s, response)
    assert result.passed
    assert result.score == 1.0


def test_eval_wrong_customer():
    s = _sample()
    response = json.dumps({
        "top_customer": "Bob",
        "top_customer_total": 300.0,
        "customer_count": 2,
        "total_revenue": 400.0,
        "summary": "Bob leads.",
    })
    result = evaluate_sample(s, response)
    assert not result.passed
    assert result.score < 1.0


def test_eval_invalid_json():
    s = _sample()
    result = evaluate_sample(s, "not json")
    assert result.score == 0.0
    assert not result.passed


def test_eval_extra_fields_fail():
    s = _sample()
    response = json.dumps({
        "top_customer": "Alice",
        "top_customer_total": 300.0,
        "customer_count": 2,
        "total_revenue": 400.0,
        "summary": "Alice leads.",
        "extra_field": "bad",
    })
    result = evaluate_sample(s, response)
    assert not result.passed


def test_eval_markdown_fence_stripped():
    s = _sample()
    inner = json.dumps({
        "top_customer": "Alice",
        "top_customer_total": 300.0,
        "customer_count": 2,
        "total_revenue": 400.0,
        "summary": "Alice leads.",
    })
    result = evaluate_sample(s, f"```json\n{inner}\n```")
    assert result.passed


def test_locked_eval_hash_stable():
    samples = [EvalSample(sample_id="x", prompt="p", expected={})]
    ev = LockedEvalSet(samples)
    h1 = ev.hash
    _ = ev.samples
    assert ev.hash == h1


def test_locked_eval_mutation_detected():
    samples = [EvalSample(sample_id="x", prompt="p", expected={})]
    ev = LockedEvalSet(samples)
    ev._samples[0] = EvalSample(sample_id="x", prompt="TAMPERED", expected={})
    with pytest.raises(RuntimeError, match="integrity"):
        _ = ev.samples


def test_prompts_only_excludes_expected():
    samples = [EvalSample(sample_id="x", prompt="p", expected={"secret": "answer"})]
    ev = LockedEvalSet(samples)
    prompts = ev.prompts_only()
    for p in prompts:
        assert "expected" not in p
        assert "secret" not in str(p)
