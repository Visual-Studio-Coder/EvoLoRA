"""Demo task: structured customer spending summary."""

from __future__ import annotations

from evolora.evaluation.locked import LockedEvalSet
from evolora.models.core import EvalSample

# ---------------------------------------------------------------------------
# Locked eval set — ground truth never passed to the planner
# ---------------------------------------------------------------------------

_LOCKED_SAMPLES = [
    EvalSample(
        sample_id="lock-001",
        prompt=(
            'Customers: [{"name":"Alice","purchases":[120.50,340.00,89.99]},'
            '{"name":"Bob","purchases":[200.00]},'
            '{"name":"Carol","purchases":[450.00,100.00]}]. '
            "Respond with a JSON summary."
        ),
        expected={
            "top_customer": "Carol",
            "top_customer_total": 550.0,
            "customer_count": 3,
            "total_revenue": 1300.49,
            "summary": "Carol leads with $550.00.",
        },
    ),
    EvalSample(
        sample_id="lock-002",
        prompt=(
            'Customers: [{"name":"Dave","purchases":[99.99]},'
            '{"name":"Eve","purchases":[1000.00,250.00]}]. '
            "Respond with a JSON summary."
        ),
        expected={
            "top_customer": "Eve",
            "top_customer_total": 1250.0,
            "customer_count": 2,
            "total_revenue": 1349.99,
            "summary": "Eve leads with $1250.00.",
        },
    ),
    EvalSample(
        sample_id="lock-003",
        prompt=(
            'Customers: [{"name":"Frank","purchases":[500.00,500.00]},'
            '{"name":"Grace","purchases":[250.00,250.00,250.00]},'
            '{"name":"Henry","purchases":[100.00]}]. '
            "Respond with a JSON summary."
        ),
        expected={
            "top_customer": "Frank",
            "top_customer_total": 1000.0,
            "customer_count": 3,
            "total_revenue": 1850.0,
            "summary": "Frank leads with $1000.00.",
        },
    ),
]

# ---------------------------------------------------------------------------
# Adaptive eval set — separate from locked, not used for official score
# ---------------------------------------------------------------------------

_ADAPTIVE_SAMPLES = [
    EvalSample(
        sample_id="adapt-001",
        prompt=(
            'Customers: [{"name":"Ivy","purchases":[75.00,75.00,75.00]},'
            '{"name":"Jack","purchases":[300.00]}]. '
            "Respond with a JSON summary."
        ),
        expected={
            "top_customer": "Jack",
            "top_customer_total": 300.0,
            "customer_count": 2,
            "total_revenue": 525.0,
            "summary": "Jack leads with $300.00.",
        },
    ),
]

LOCKED_EVAL_SET = LockedEvalSet(_LOCKED_SAMPLES)
ADAPTIVE_EVAL_SET = LockedEvalSet(_ADAPTIVE_SAMPLES)
