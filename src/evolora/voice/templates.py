"""Deterministic ~10-word narration sentences for orchestrator events.

These are the demo-safe fallback (always available, no network). The Narrator may
optionally rephrase them via MiniMax, but if that fails the template line is spoken
verbatim. Returning ``None`` means "this event is not worth narrating".
"""

from __future__ import annotations

from evolora.models.events import Event, EventKind

# Events that should be spoken immediately, bypassing the throttle.
MILESTONES: frozenset[EventKind] = frozenset(
    {
        EventKind.RUN_STARTED,
        EventKind.BASELINE_COMPLETE,
        EventKind.EVAL_APPROVAL_REQUIRED,
        EventKind.USER_APPROVAL_REQUIRED,
        EventKind.BEST_UPDATED,
        EventKind.STOP_CONDITION_MET,
        EventKind.RUN_COMPLETE,
        EventKind.RUN_FAILED,
        EventKind.RUN_CANCELLED,
    }
)


def _f(data: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def render_event(event: Event) -> str | None:
    """Return a short narration sentence for *event*, or ``None`` to stay silent."""
    kind = event.kind
    data = event.data or {}

    if kind == EventKind.RUN_STARTED:
        return "Starting a new EvoLoRA self-improvement run now."
    if kind == EventKind.EVAL_SET_LOCKED:
        return "Benchmark locked. The evaluation set is now frozen."
    if kind == EventKind.BASELINE_COMPLETE:
        return f"Baseline scored {_f(data, 'score'):.2f}. Let's try to beat it."
    if kind == EventKind.PLANNING_STARTED:
        return "The agent is planning LoRA hyperparameters and training data."
    if kind == EventKind.PLAN_RECEIVED:
        focus = ", ".join((data.get("focus_areas") or [])[:2]) or "format and accuracy"
        return f"Plan ready. Focusing on {focus}."
    if kind == EventKind.AGENT_FALLBACK_USED:
        return "MiniMax unavailable, so using the heuristic planner instead."
    if kind == EventKind.EVAL_APPROVAL_REQUIRED:
        n = len(data.get("evals") or [])
        return f"Please review the {n} generated evaluation examples."
    if kind == EventKind.VALIDATION_COMPLETE:
        return "Hyperparameters validated. Getting the training run ready."
    if kind == EventKind.TRAINING_STARTED:
        return "Training has started on the new adapter."
    if kind == EventKind.TRAINING_PROGRESS:
        step = int(_f(data, "step"))
        total = int(_f(data, "total_steps"))
        return f"Training step {step} of {total}, loss {_f(data, 'loss'):.2f}."
    if kind == EventKind.TRAINING_COMPLETE:
        return "Training finished. Now evaluating the new adapter."
    if kind == EventKind.EVAL_STARTED:
        return "Scoring the adapter on the locked benchmark now."
    if kind == EventKind.EVAL_COMPLETE:
        return f"The new adapter scored {_f(data, 'score'):.2f} on the benchmark."
    if kind == EventKind.ADAPTIVE_COMPLETE:
        return f"Adaptive challenge scored {_f(data, 'score'):.2f} this round."
    if kind == EventKind.JUDGE_STARTED:
        return "An external judge is reviewing the model's answers."
    if kind == EventKind.JUDGE_COMPLETE:
        return f"The judge rated this model {_f(data, 'rating'):.1f}."
    if kind == EventKind.RETRAIN_DECISION_RECEIVED:
        rec = "retrain" if data.get("retrain_recommended") else "stop"
        return f"The advisor recommends to {rec}."
    if kind == EventKind.USER_APPROVAL_REQUIRED:
        return "Waiting for your decision. Press yes or no."
    if kind == EventKind.USER_APPROVAL_RECEIVED:
        return "Decision received. Continuing the loop."
    if kind == EventKind.BEST_UPDATED:
        return f"New best score {_f(data, 'score'):.2f}. Progress is improving."
    if kind == EventKind.ITERATION_COMPLETE:
        return f"Iteration complete. Best score so far {_f(data, 'best'):.2f}."
    if kind == EventKind.STOP_CONDITION_MET:
        return "A stopping condition was met. Wrapping up."
    if kind == EventKind.RUN_COMPLETE:
        return f"Run complete. Best score {_f(data, 'best_score'):.2f}. Nice work."
    if kind == EventKind.RUN_FAILED:
        return "The run failed. Please check the logs."
    if kind == EventKind.RUN_CANCELLED:
        return "The run was cancelled."

    # STATUS_CHANGED and LOG are too noisy / low-signal to narrate.
    return None
