"""Objective evaluator for the customer spending summary task."""

from __future__ import annotations

import json
import time
from typing import Any

from evolora.models.core import EvalResult, EvalSample

REQUIRED_FIELDS = {
    "top_customer",
    "top_customer_total",
    "customer_count",
    "total_revenue",
    "summary",
}
ALLOWED_FIELDS = REQUIRED_FIELDS
MAX_RESPONSE_LENGTH = 2000
TOTAL_TOLERANCE = 0.01  # 1% relative tolerance


def _relative_close(a: float, b: float, tol: float = TOTAL_TOLERANCE) -> bool:
    if b == 0:
        return abs(a) < tol
    return abs(a - b) / abs(b) <= tol


def evaluate_sample(sample: EvalSample, raw_response: str) -> EvalResult:
    """Score a single model response against the locked expected output."""
    t0 = time.monotonic()
    details: dict[str, Any] = {}
    score = 0.0
    passed = False

    # Strip markdown fences
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Length guard
    if len(text) > MAX_RESPONSE_LENGTH:
        details["error"] = f"response too long ({len(text)} chars)"
        return EvalResult(
            sample_id=sample.sample_id,
            score=0.0,
            passed=False,
            details=details,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # Parse JSON
    try:
        parsed: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        details["error"] = f"invalid JSON: {exc}"
        return EvalResult(
            sample_id=sample.sample_id,
            score=0.0,
            passed=False,
            details=details,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    checks: dict[str, bool] = {}

    # Required fields present
    checks["has_required_fields"] = REQUIRED_FIELDS.issubset(parsed.keys())

    # No unsupported fields
    extra = set(parsed.keys()) - ALLOWED_FIELDS
    checks["no_extra_fields"] = len(extra) == 0
    if extra:
        details["extra_fields"] = list(extra)

    expected = sample.expected

    # Customer count
    try:
        checks["customer_count"] = int(parsed.get("customer_count", -1)) == int(
            expected.get("customer_count", 0)
        )
    except (TypeError, ValueError):
        checks["customer_count"] = False

    # Top customer
    checks["top_customer"] = (
        str(parsed.get("top_customer", "")).strip().lower()
        == str(expected.get("top_customer", "")).strip().lower()
    )

    # top_customer_total within tolerance
    try:
        checks["top_customer_total"] = _relative_close(
            float(parsed.get("top_customer_total", 0)),
            float(expected.get("top_customer_total", 0)),
        )
    except (TypeError, ValueError):
        checks["top_customer_total"] = False

    # total_revenue within tolerance
    try:
        checks["total_revenue"] = _relative_close(
            float(parsed.get("total_revenue", 0)),
            float(expected.get("total_revenue", 0)),
        )
    except (TypeError, ValueError):
        checks["total_revenue"] = False

    # summary non-empty
    checks["summary_present"] = bool(str(parsed.get("summary", "")).strip())

    details["checks"] = checks
    passing = sum(checks.values())
    total = len(checks)
    score = passing / total
    passed = all(checks.values())

    return EvalResult(
        sample_id=sample.sample_id,
        score=score,
        passed=passed,
        details=details,
        latency_ms=(time.monotonic() - t0) * 1000,
    )


class ObjectiveEvaluator:
    """Run the locked eval set and return aggregate score."""

    def __call__(
        self,
        samples: list[EvalSample],
        responses: dict[str, str],
    ) -> tuple[float, list[EvalResult]]:
        results = [evaluate_sample(s, responses.get(s.sample_id, "")) for s in samples]
        score = sum(r.score for r in results) / len(results) if results else 0.0
        return score, results


# ---------------------------------------------------------------------------
# Generic evaluator — scores against an arbitrary expected JSON object, so
# MiniMax-generated, goal-specific eval sets can be scored objectively without
# the customer-spending field assumptions baked into evaluate_sample().
# ---------------------------------------------------------------------------


def _values_match(got: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(got) == expected
    if isinstance(expected, (int, float)):
        try:
            return _relative_close(float(got), float(expected))
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str):
        return str(got).strip().lower() == expected.strip().lower()
    return got == expected


def generic_evaluate_sample(sample: EvalSample, raw_response: str) -> EvalResult:
    """Score a response against an arbitrary expected JSON object (any goal)."""
    t0 = time.monotonic()
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    def _fail(msg: str) -> EvalResult:
        return EvalResult(
            sample_id=sample.sample_id,
            score=0.0,
            passed=False,
            details={"error": msg},
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    if len(text) > MAX_RESPONSE_LENGTH:
        return _fail(f"response too long ({len(text)} chars)")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return _fail(f"invalid JSON: {exc}")
    if not isinstance(parsed, dict):
        return _fail("response is not a JSON object")

    expected = sample.expected or {}
    if not expected:
        # No ground-truth fields — valid JSON is the only requirement.
        return EvalResult(
            sample_id=sample.sample_id,
            score=1.0,
            passed=True,
            details={"note": "valid JSON (no expected fields)"},
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    checks = {key: _values_match(parsed.get(key), exp) for key, exp in expected.items()}
    score = sum(checks.values()) / len(checks)
    return EvalResult(
        sample_id=sample.sample_id,
        score=score,
        passed=all(checks.values()),
        details={"checks": checks},
        latency_ms=(time.monotonic() - t0) * 1000,
    )


class GenericEvaluator:
    """Score responses against an arbitrary generated eval set (any goal)."""

    def __call__(
        self,
        samples: list[EvalSample],
        responses: dict[str, str],
    ) -> tuple[float, list[EvalResult]]:
        results = [generic_evaluate_sample(s, responses.get(s.sample_id, "")) for s in samples]
        score = sum(r.score for r in results) / len(results) if results else 0.0
        return score, results
