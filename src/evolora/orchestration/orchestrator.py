"""EvoLoRA orchestrator — bounded self-improvement state machine."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone

from evolora.agent.planner import HeuristicPlanner, MiniMaxPlanner
from evolora.evaluation.evaluator import ObjectiveEvaluator
from evolora.evaluation.locked import LockedEvalSet
from evolora.models.core import (
    AgentPlan,
    ArtifactMeta,
    EvalResult,
    IterationResult,
    RunConfig,
    RunRecord,
    RunStatus,
    StopReason,
)
from evolora.models.events import Event, EventKind
from evolora.persistence.artifacts import ArtifactStore, LocalArtifactStore
from evolora.persistence.store import InMemoryRunStore, RunStore
from evolora.training.backends import MockTrainingBackend, TrainingBackend
from evolora.training.runner import MockModelRunner, ModelRunner


def _now():
    return datetime.now(timezone.utc)


class Orchestrator:
    """Drives the EvoLoRA loop: baseline → plan → train → eval → repeat."""

    def __init__(
        self,
        config: RunConfig,
        eval_set: LockedEvalSet,
        *,
        planner: MiniMaxPlanner | HeuristicPlanner | None = None,
        training_backend: TrainingBackend | None = None,
        model_runner: ModelRunner | None = None,
        run_store: RunStore | None = None,
        artifact_store: ArtifactStore | None = None,
        adaptive_eval_set: LockedEvalSet | None = None,
    ) -> None:
        self._config = config
        self._eval_set = eval_set
        self._adaptive_set = adaptive_eval_set
        self._planner = planner or HeuristicPlanner()
        self._backend = training_backend or MockTrainingBackend()
        self._runner = model_runner or MockModelRunner()
        self._store = run_store or InMemoryRunStore()
        self._artifacts = artifact_store or LocalArtifactStore("./artifacts")
        self._evaluator = ObjectiveEvaluator()
        self._cancelled = False
        self._record = RunRecord(config=config)

    def cancel(self) -> None:
        self._cancelled = True

    async def run(self) -> AsyncIterator[Event]:
        return self._run()

    async def _run(self) -> AsyncIterator[Event]:
        rec = self._record
        rid = rec.run_id

        async def emit(kind: EventKind, msg: str = "", **data) -> Event:
            ev = Event(kind=kind, run_id=rid, iteration=rec.current_iteration(), message=msg, data=data)
            return ev

        # --- PREPARING ---
        rec.status = RunStatus.PREPARING
        yield await emit(EventKind.RUN_STARTED, "EvoLoRA run started", mock=self._backend.is_mock)
        await self._store.save(rec)

        # --- LOCK EVAL SET ---
        rec.status = RunStatus.LOCKING_EVAL
        yield await emit(EventKind.EVAL_SET_LOCKED, f"Eval set locked (hash={self._eval_set.hash[:12]}…)", hash=self._eval_set.hash, size=len(self._eval_set))
        await self._store.save(rec)
        rec.eval_set_hash = self._eval_set.hash

        # --- BASELINE ---
        rec.status = RunStatus.BASELINE
        yield await emit(EventKind.STATUS_CHANGED, "Running baseline evaluation")
        baseline_score, _ = await self._eval(adapter_path=None)
        rec.baseline_score = baseline_score
        rec.best_score = baseline_score
        yield await emit(EventKind.BASELINE_COMPLETE, f"Baseline score: {baseline_score:.3f}", score=baseline_score)
        await self._store.save(rec)

        # --- MAIN LOOP ---
        for iteration in range(1, rec.config.max_iterations + 1):
            if self._cancelled:
                yield await self._finish(rec, RunStatus.CANCELLED, StopReason.CANCELLED, emit)
                return

            # --- PLAN ---
            rec.status = RunStatus.PLANNING
            yield await emit(EventKind.PLANNING_STARTED, f"Iteration {iteration}: requesting plan", iteration=iteration)

            prev_score = rec.iterations[-1].score if rec.iterations else baseline_score
            failures = self._get_failures(rec)
            plan, fallback = await self._plan(iteration, baseline_score, prev_score, failures)

            if fallback:
                yield await emit(EventKind.AGENT_FALLBACK_USED, "MiniMax unavailable — heuristic plan used")
            yield await emit(EventKind.PLAN_RECEIVED, "Plan received", rationale=plan.rationale[:200], focus_areas=plan.focus_areas)

            # --- VALIDATE DATA ---
            rec.status = RunStatus.VALIDATING
            validated_plan = self._validate_plan(plan)
            yield await emit(EventKind.VALIDATION_COMPLETE, f"Plan validated: {len(validated_plan.data_spec.examples)} examples")

            # --- TRAIN ---
            rec.status = RunStatus.TRAINING
            yield await emit(EventKind.TRAINING_STARTED, "Training started", backend=self._backend.name)

            artifact: ArtifactMeta | None = None
            train_cost = 0.0
            train_duration = 0.0
            train_error: str | None = None

            try:
                stream = await self._backend.train(rid, iteration, validated_plan, rec.config.base_model_id)
                async for progress in stream:
                    if self._cancelled:
                        break
                    if progress.get("done"):
                        artifact = progress.get("artifact")
                        train_cost = progress.get("cost_usd", 0.0)
                        train_duration = progress.get("duration_s", 0.0)
                    else:
                        yield await emit(EventKind.TRAINING_PROGRESS, "", **{k: v for k, v in progress.items() if k != "done"})
            except Exception as exc:
                train_error = str(exc)

            if train_error or artifact is None:
                it_result = IterationResult(
                    iteration=iteration,
                    plan=validated_plan,
                    agent_fallback_used=fallback,
                    error=train_error or "No artifact produced",
                    started_at=_now(),
                    finished_at=_now(),
                )
                rec.iterations.append(it_result)
                await self._store.save(rec)

                failed_count = sum(1 for it in rec.iterations if it.error is not None)
                if failed_count >= 3:  # repeated training failures
                    yield await self._finish(rec, RunStatus.FAILED, StopReason.TRAINING_FAILURE, emit)
                    return
                continue

            yield await emit(EventKind.TRAINING_COMPLETE, "Training complete", cost_usd=train_cost, duration_s=train_duration, is_mock=self._backend.is_mock)

            # Save mock artifact
            if self._backend.is_mock:
                artifact = await self._artifacts.save_mock(artifact)

            # --- EVAL ---
            rec.status = RunStatus.EVALUATING
            yield await emit(EventKind.EVAL_STARTED, "Running locked evaluation")

            # Integrity check before eval
            try:
                self._eval_set._assert_integrity()
            except RuntimeError:
                yield await self._finish(rec, RunStatus.FAILED, StopReason.EVAL_HASH_CHANGED, emit)
                return

            score, eval_results = await self._eval(adapter_path=artifact.adapter_path)
            artifact = artifact.model_copy(update={"score": score})
            yield await emit(EventKind.EVAL_COMPLETE, f"Eval score: {score:.3f}", score=score)

            # --- ADAPTIVE ---
            adaptive_score = None
            if self._adaptive_set:
                rec.status = RunStatus.ADAPTIVE
                a_score, _ = await self._eval(adapter_path=artifact.adapter_path, eval_set=self._adaptive_set)
                adaptive_score = a_score
                yield await emit(EventKind.ADAPTIVE_COMPLETE, f"Adaptive score: {a_score:.3f}", score=a_score)

            # --- RECORD ITERATION ---
            it_result = IterationResult(
                iteration=iteration,
                plan=validated_plan,
                agent_fallback_used=fallback,
                training_cost_usd=train_cost,
                training_duration_s=train_duration,
                eval_results=eval_results,
                score=score,
                adaptive_score=adaptive_score,
                artifact=artifact,
                started_at=_now(),
                finished_at=_now(),
            )
            rec.iterations.append(it_result)
            rec.total_cost_usd += train_cost

            # Update best
            if score > rec.best_score:
                rec.best_score = score
                rec.best_iteration = iteration
                yield await emit(EventKind.BEST_UPDATED, f"New best: {score:.3f} (iteration {iteration})", score=score, iteration=iteration)

            yield await emit(EventKind.ITERATION_COMPLETE, f"Iteration {iteration} done: {score:.3f}", score=score, best=rec.best_score)
            await self._store.save(rec)

            # --- STOP CONDITIONS ---
            stop = self._check_stop(rec, score)
            if stop:
                yield await emit(EventKind.STOP_CONDITION_MET, f"Stop: {stop.value}", reason=stop.value)
                yield await self._finish(rec, RunStatus.COMPLETE, stop, emit)
                return

        yield await self._finish(rec, RunStatus.COMPLETE, StopReason.MAX_ITERATIONS, emit)

    async def _eval(
        self, adapter_path: str | None, eval_set: LockedEvalSet | None = None
    ) -> tuple[float, list[EvalResult]]:
        es = eval_set or self._eval_set
        prompts = es.prompts_only()
        responses = await self._runner.run_batch(prompts, adapter_path=adapter_path)
        return self._evaluator(es.samples, responses)

    async def _plan(self, iteration, baseline_score, current_score, failures):
        if isinstance(self._planner, MiniMaxPlanner):
            return await self._planner.plan(iteration, baseline_score, current_score, failures)
        return self._planner.plan(iteration, baseline_score, current_score, failures), False

    def _validate_plan(self, plan: AgentPlan) -> AgentPlan:
        # Already validated by Pydantic — just return
        return plan

    def _get_failures(self, rec: RunRecord) -> list[EvalResult]:
        if not rec.iterations:
            return []
        last = rec.iterations[-1]
        return [r for r in last.eval_results if not r.passed]

    def _check_stop(self, rec: RunRecord, score: float) -> StopReason | None:
        cfg = rec.config
        if score >= cfg.target_score:
            return StopReason.TARGET_SCORE
        if rec.total_cost_usd >= cfg.max_budget_usd:
            return StopReason.BUDGET_CAP
        if rec.no_improvement_count() >= cfg.patience:
            return StopReason.PATIENCE
        return None

    async def _finish(self, rec: RunRecord, status: RunStatus, reason: StopReason, emit) -> Event:
        rec.status = status
        rec.stop_reason = reason
        rec.finished_at = _now()
        await self._store.save(rec)
        kind = {
            RunStatus.COMPLETE: EventKind.RUN_COMPLETE,
            RunStatus.FAILED: EventKind.RUN_FAILED,
            RunStatus.CANCELLED: EventKind.RUN_CANCELLED,
        }.get(status, EventKind.RUN_COMPLETE)
        return Event(
            kind=kind,
            run_id=rec.run_id,
            message=f"Run {status.value}: {reason.value}",
            data={"reason": reason.value, "best_score": rec.best_score, "best_iteration": rec.best_iteration},
        )
