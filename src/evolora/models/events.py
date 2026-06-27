"""Async event stream types for TUI/CLI integration."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


class EventKind(StrEnum):
    RUN_STARTED = "run_started"
    STATUS_CHANGED = "status_changed"
    EVAL_SET_LOCKED = "eval_set_locked"
    BASELINE_COMPLETE = "baseline_complete"
    PLANNING_STARTED = "planning_started"
    PLAN_RECEIVED = "plan_received"
    AGENT_FALLBACK_USED = "agent_fallback_used"
    VALIDATION_COMPLETE = "validation_complete"
    TRAINING_STARTED = "training_started"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETE = "training_complete"
    EVAL_STARTED = "eval_started"
    EVAL_COMPLETE = "eval_complete"
    ADAPTIVE_COMPLETE = "adaptive_complete"
    JUDGE_STARTED = "judge_started"
    JUDGE_COMPLETE = "judge_complete"
    RETRAIN_DECISION_RECEIVED = "retrain_decision_received"
    USER_APPROVAL_REQUIRED = "user_approval_required"
    USER_APPROVAL_RECEIVED = "user_approval_received"
    ITERATION_COMPLETE = "iteration_complete"
    BEST_UPDATED = "best_updated"
    STOP_CONDITION_MET = "stop_condition_met"
    RUN_COMPLETE = "run_complete"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    LOG = "log"


class Event(BaseModel):
    kind: EventKind
    run_id: str
    iteration: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    ts: datetime = Field(default_factory=_now)

    @classmethod
    def log(cls, run_id: str, message: str, iteration: int | None = None) -> Event:
        return cls(kind=EventKind.LOG, run_id=run_id, message=message, iteration=iteration)
