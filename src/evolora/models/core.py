"""Core Pydantic models for EvoLoRA runs, plans, and results."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _now() -> datetime:
    return datetime.now(UTC)


def _run_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RunStatus(StrEnum):
    PENDING = "pending"
    PREPARING = "preparing"
    LOCKING_EVAL = "locking_eval"
    BASELINE = "baseline"
    PLANNING = "planning"
    VALIDATING = "validating"
    TRAINING = "training"
    EVALUATING = "evaluating"
    ADAPTIVE = "adaptive"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StopReason(StrEnum):
    TARGET_SCORE = "target_score"
    MAX_ITERATIONS = "max_iterations"
    PATIENCE = "patience"
    TRAINING_FAILURE = "training_failure"
    EVAL_HASH_CHANGED = "eval_hash_changed"
    CANCELLED = "cancelled"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    JUDGE_ACCEPTED = "judge_accepted"
    USER_DECLINED_RETRAIN = "user_declined_retrain"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class LoraHyperparams(BaseModel):
    r: int = Field(default=8, ge=1, le=64)
    lora_alpha: int = Field(default=16, ge=1, le=256)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    learning_rate: float = Field(default=2e-4, gt=0.0, le=0.1)
    num_epochs: int = Field(default=1, ge=1, le=6)
    batch_size: int = Field(default=4, ge=1, le=32)
    warmup_steps: int = Field(default=10, ge=0, le=500)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.5)

    @field_validator("r")
    @classmethod
    def r_power_of_two(cls, v: int) -> int:
        if v not in {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError(f"r must be a power of two (1-64), got {v}")
        return v


class TrainingDataSpec(BaseModel):
    examples: list[dict[str, str]] = Field(default_factory=list)
    rationale: str = Field(default="")
    max_examples: int = Field(default=50, ge=1, le=500)

    @field_validator("examples")
    @classmethod
    def deduplicate_examples(cls, v: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        unique = []
        for ex in v:
            key = str(sorted(ex.items()))
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        return unique

    @model_validator(mode="after")
    def cap_examples(self) -> TrainingDataSpec:
        if len(self.examples) > self.max_examples:
            self.examples = self.examples[: self.max_examples]
        return self


class AgentPlan(BaseModel):
    hyperparams: LoraHyperparams = Field(default_factory=LoraHyperparams)
    data_spec: TrainingDataSpec = Field(default_factory=TrainingDataSpec)
    rationale: str = Field(default="")
    focus_areas: list[str] = Field(default_factory=list)
    # Never allow MiniMax to specify file paths or arbitrary model IDs
    target_adapter_name: str = Field(default="")

    @field_validator("target_adapter_name")
    @classmethod
    def no_path_traversal(cls, v: str) -> str:
        if any(c in v for c in ["/", "\\", "..", "~"]):
            raise ValueError("target_adapter_name must not contain path components")
        return v


class EvalSample(BaseModel):
    sample_id: str
    prompt: str
    expected: dict[str, Any]  # ground truth — never passed to MiniMax planner


class EvalResult(BaseModel):
    sample_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0)


class JudgeReport(BaseModel):
    rating: float = Field(ge=0.0, le=1.0)
    summary: str = Field(default="")
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommended_focus: list[str] = Field(default_factory=list)
    source: str = Field(default="heuristic")
    is_mock: bool = True


class RetrainDecision(BaseModel):
    retrain_recommended: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = Field(default="")
    suggested_focus: list[str] = Field(default_factory=list)
    source: str = Field(default="heuristic")
    is_mock: bool = True


class ArtifactMeta(BaseModel):
    artifact_id: str = Field(default_factory=_run_id)
    run_id: str
    iteration: int
    adapter_path: str
    score: float
    checksum: str
    is_mock: bool = True
    created_at: datetime = Field(default_factory=_now)


class IterationResult(BaseModel):
    iteration: int
    plan: AgentPlan
    agent_fallback_used: bool = False
    training_cost_usd: float = Field(default=0.0, ge=0.0)
    training_duration_s: float = Field(default=0.0, ge=0.0)
    eval_results: list[EvalResult] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    adaptive_score: float | None = None
    judge_report: JudgeReport | None = None
    retrain_decision: RetrainDecision | None = None
    artifact: ArtifactMeta | None = None
    stop_reason: StopReason | None = None
    error: str | None = None
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.artifact is not None


# ---------------------------------------------------------------------------
# Run-level models
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    run_id: str = Field(default_factory=_run_id)
    task_name: str = Field(default="customer_spending_summary")
    goal: str = Field(default="")  # user's stated use case from the TUI/CLI, sent to the planner
    max_iterations: int = Field(default=3, ge=1, le=20)
    target_score: float = Field(default=0.85, ge=0.0, le=1.0)
    improvement_threshold: float = Field(default=0.01, ge=0.0)
    patience: int = Field(default=2, ge=1)
    training_backend: str = Field(default="mock")
    model_runner: str = Field(default="mock")
    base_model_id: str = Field(default="microsoft/Phi-3-mini-128k-instruct")
    training_sample_count: int | None = Field(default=30, ge=1, le=500)
    require_retrain_approval: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_now)


class RunRecord(BaseModel):
    config: RunConfig
    status: RunStatus = RunStatus.PENDING
    eval_set_hash: str = ""
    baseline_score: float = 0.0
    iterations: list[IterationResult] = Field(default_factory=list)
    best_iteration: int | None = None
    best_score: float = 0.0
    stop_reason: StopReason | None = None
    total_cost_usd: float = 0.0
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
    error: str | None = None

    @property
    def run_id(self) -> str:
        return self.config.run_id

    @property
    def is_terminal(self) -> bool:
        return self.status in {RunStatus.COMPLETE, RunStatus.FAILED, RunStatus.CANCELLED}

    def current_iteration(self) -> int:
        return len(self.iterations)

    def no_improvement_count(self) -> int:
        if len(self.iterations) < 2:
            return 0
        count = 0
        prev = self.iterations[-2].score if len(self.iterations) >= 2 else self.baseline_score
        for it in reversed(self.iterations):
            if it.score - prev < self.config.improvement_threshold:
                count += 1
                prev = it.score
            else:
                break
        return count
