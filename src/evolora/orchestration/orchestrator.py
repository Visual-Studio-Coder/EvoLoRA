"""EvoLoRA orchestrator — bounded self-improvement state machine."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from evolora.agent.planner import HeuristicPlanner, MiniMaxPlanner
from evolora.evaluation.digitalocean_judge import CandidateJudge
from evolora.evaluation.evaluator import GenericEvaluator, ObjectiveEvaluator
from evolora.evaluation.locked import LockedEvalSet
from evolora.models.core import (
    AgentPlan,
    ArtifactMeta,
    EvalResult,
    EvalSample,
    IterationResult,
    RetrainDecision,
    RunConfig,
    RunRecord,
    RunStatus,
    StopReason,
)
from evolora.models.events import Event, EventKind
from evolora.orchestration.retrain_advisor import RetrainAdvisor
from evolora.persistence.artifacts import ArtifactStore, LocalArtifactStore
from evolora.persistence.store import RunStore, get_run_store
from evolora.training.backends import MockTrainingBackend, TrainingBackend
from evolora.training.remote_config import build_training_config_payload, push_config
from evolora.training.runner import MockModelRunner, ModelRunner


def _now():
    return datetime.now(UTC)


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
        judge: CandidateJudge | None = None,
        retrain_advisor: RetrainAdvisor | None = None,
    ) -> None:
        self._config = config
        self._eval_set = eval_set
        self._adaptive_set = adaptive_eval_set
        self._planner = planner or HeuristicPlanner()
        self._backend = training_backend or MockTrainingBackend()
        self._runner = model_runner or MockModelRunner()
        self._judge = judge
        self._retrain_advisor = retrain_advisor
        self._store = run_store or get_run_store()
        self._artifacts = artifact_store or LocalArtifactStore("./artifacts")
        self._evaluator = ObjectiveEvaluator()
        self._cancelled = False
        self._approval_future: asyncio.Future[bool] | None = None
        self._record = RunRecord(config=config)

    def cancel(self) -> None:
        self._cancelled = True
        self.submit_retrain_approval(False)

    def submit_retrain_approval(self, approved: bool) -> None:
        if self._approval_future is not None and not self._approval_future.done():
            self._approval_future.set_result(approved)

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

        # --- MINIMAX-GENERATED EVALS (goal-driven) ---
        # When the user supplied a goal and the agent can generate evals, MiniMax
        # produces a goal-specific objective eval set; we score it with the generic
        # evaluator and skip the demo adaptive set (which is customer-spending).
        if rec.config.goal and isinstance(self._planner, MiniMaxPlanner):
            yield await emit(EventKind.STATUS_CHANGED, f"MiniMax generating evals for goal: {rec.config.goal[:60]}")
            try:
                generated = await self._planner.generate_evals(rec.config.goal)
                if generated:
                    samples = [
                        EvalSample(sample_id=f"gen-{i + 1:03d}", prompt=g["prompt"], expected=g["expected"])
                        for i, g in enumerate(generated)
                    ]
                    self._eval_set = LockedEvalSet(samples)
                    self._evaluator = GenericEvaluator()
                    self._adaptive_set = None
                    yield await emit(EventKind.LOG, f"MiniMax generated {len(samples)} goal-specific eval examples")
            except Exception as exc:  # pragma: no cover - defensive; fall back to default eval set
                yield await emit(EventKind.LOG, f"Eval generation failed, using default eval set ({exc})")

        # --- LOCK EVAL SET ---
        rec.status = RunStatus.LOCKING_EVAL
        yield await emit(EventKind.EVAL_SET_LOCKED, f"Eval set locked (hash={self._eval_set.hash[:12]}…)", hash=self._eval_set.hash, size=len(self._eval_set))
        await self._store.save(rec)
        rec.eval_set_hash = self._eval_set.hash

        # --- BASELINE ---
        rec.status = RunStatus.BASELINE
        yield await emit(EventKind.STATUS_CHANGED, "Running baseline evaluation")
        baseline_score, _, _ = await self._eval(adapter_path=None)
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
            plan, fallback = await self._plan(
                iteration,
                baseline_score,
                prev_score,
                failures,
                rec.config.training_sample_count,
                rec.config.goal,
            )

            if fallback:
                yield await emit(EventKind.AGENT_FALLBACK_USED, "MiniMax unavailable — heuristic plan used")
            yield await emit(EventKind.PLAN_RECEIVED, "Plan received", rationale=plan.rationale[:200], focus_areas=plan.focus_areas)

            # --- VALIDATE DATA ---
            rec.status = RunStatus.VALIDATING
            validated_plan = self._validate_plan(plan)
            exact_count = rec.config.training_sample_count
            count_note = (
                f"exact requested count {exact_count}"
                if exact_count is not None
                else "agent-selected count"
            )
            yield await emit(
                EventKind.VALIDATION_COMPLETE,
                f"Plan validated: {len(validated_plan.data_spec.examples)} examples ({count_note})",
                example_count=len(validated_plan.data_spec.examples),
                requested_training_sample_count=exact_count,
            )

            # --- TRAIN ---
            rec.status = RunStatus.TRAINING
            yield await emit(EventKind.TRAINING_STARTED, "Training started", backend=self._backend.name)

            artifact: ArtifactMeta | None = None
            train_cost = 0.0
            train_duration = 0.0
            train_error: str | None = None

            try:
                if rec.config.training_backend in {"remote", "unsloth"}:
                    remote_payload = build_training_config_payload(
                        run_id=rid,
                        iteration=iteration,
                        run_config=rec.config,
                        plan=validated_plan,
                        eval_set=self._eval_set,
                    )
                    remote_push = push_config(remote_payload)
                    yield await emit(
                        EventKind.LOG,
                        remote_push.message,
                        dry_run=remote_push.dry_run,
                        pushed=remote_push.pushed,
                        remote_path=remote_push.remote_path,
                        byte_count=remote_push.byte_count,
                    )

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

            score, eval_results, eval_responses = await self._eval(adapter_path=artifact.adapter_path)
            artifact = artifact.model_copy(update={"score": score})
            yield await emit(EventKind.EVAL_COMPLETE, f"Eval score: {score:.3f}", score=score)

            # --- ADAPTIVE ---
            adaptive_score = None
            if self._adaptive_set:
                rec.status = RunStatus.ADAPTIVE
                a_score, _, _ = await self._eval(adapter_path=artifact.adapter_path, eval_set=self._adaptive_set)
                adaptive_score = a_score
                yield await emit(EventKind.ADAPTIVE_COMPLETE, f"Adaptive score: {a_score:.3f}", score=a_score)

            judge_report = None
            retrain_decision: RetrainDecision | None = None
            if self._judge is not None:
                source = "heuristic" if self._judge.is_mock else "DigitalOcean"
                yield await emit(EventKind.JUDGE_STARTED, f"{source} judge reviewing iteration {iteration}")
                judge_report = await self._judge.judge(
                    goal=rec.config.goal,
                    task_name=rec.config.task_name,
                    base_model_id=rec.config.base_model_id,
                    iteration=iteration,
                    score=score,
                    adaptive_score=adaptive_score,
                    plan=validated_plan,
                    eval_results=eval_results,
                    responses=eval_responses,
                )
                yield await emit(
                    EventKind.JUDGE_COMPLETE,
                    f"Judge rating: {judge_report.rating:.2f}",
                    rating=judge_report.rating,
                    summary=judge_report.summary,
                    strengths=judge_report.strengths,
                    weaknesses=judge_report.weaknesses,
                    recommended_focus=judge_report.recommended_focus,
                    source=judge_report.source,
                    is_mock=judge_report.is_mock,
                )

                if self._retrain_advisor is not None:
                    retrain_decision = await self._retrain_advisor.decide(
                        goal=rec.config.goal or rec.config.task_name,
                        rating=judge_report.rating,
                        target_score=rec.config.target_score,
                        iteration=iteration,
                        max_iterations=rec.config.max_iterations,
                        judge_report=judge_report,
                    )
                    advisor_name = "heuristic advisor" if retrain_decision.is_mock else "MiniMax"
                    yield await emit(
                        EventKind.RETRAIN_DECISION_RECEIVED,
                        (
                            f"{advisor_name} recommends retrain"
                            if retrain_decision.retrain_recommended
                            else f"{advisor_name} accepts current adapter"
                        ),
                        retrain_recommended=retrain_decision.retrain_recommended,
                        confidence=retrain_decision.confidence,
                        reason=retrain_decision.reason,
                        suggested_focus=retrain_decision.suggested_focus,
                        source=retrain_decision.source,
                        is_mock=retrain_decision.is_mock,
                    )

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
                judge_report=judge_report,
                retrain_decision=retrain_decision,
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

            if retrain_decision is not None and not retrain_decision.retrain_recommended:
                advisor_name = "heuristic advisor" if retrain_decision.is_mock else "MiniMax"
                yield await emit(
                    EventKind.STOP_CONDITION_MET,
                    f"Stop: {advisor_name} accepted the current adapter after judge review",
                    reason=StopReason.JUDGE_ACCEPTED.value,
                )
                yield await self._finish(rec, RunStatus.COMPLETE, StopReason.JUDGE_ACCEPTED, emit)
                return

            if (
                retrain_decision is not None
                and retrain_decision.retrain_recommended
                and iteration < rec.config.max_iterations
                and rec.config.require_retrain_approval
            ):
                self._approval_future = asyncio.get_running_loop().create_future()
                yield await emit(
                    EventKind.USER_APPROVAL_REQUIRED,
                    f"Approve another training round? Judge rating {judge_report.rating:.2f}",
                    rating=judge_report.rating if judge_report else score,
                    summary=judge_report.summary if judge_report else "",
                    reason=retrain_decision.reason,
                    suggested_focus=retrain_decision.suggested_focus,
                )
                approved = await self._approval_future
                self._approval_future = None
                yield await emit(
                    EventKind.USER_APPROVAL_RECEIVED,
                    "User approved retrain" if approved else "User declined retrain",
                    approved=approved,
                )
                if self._cancelled:
                    yield await self._finish(rec, RunStatus.CANCELLED, StopReason.CANCELLED, emit)
                    return
                if not approved:
                    yield await self._finish(
                        rec,
                        RunStatus.COMPLETE,
                        StopReason.USER_DECLINED_RETRAIN,
                        emit,
                    )
                    return

            # --- STOP CONDITIONS ---
            stop = self._check_stop(rec, score)
            if stop:
                yield await emit(EventKind.STOP_CONDITION_MET, f"Stop: {stop.value}", reason=stop.value)
                yield await self._finish(rec, RunStatus.COMPLETE, stop, emit)
                return

        yield await self._finish(rec, RunStatus.COMPLETE, StopReason.MAX_ITERATIONS, emit)

    async def _eval(
        self, adapter_path: str | None, eval_set: LockedEvalSet | None = None
    ) -> tuple[float, list[EvalResult], dict[str, str]]:
        es = eval_set or self._eval_set
        prompts = es.prompts_only()
        responses = await self._runner.run_batch(prompts, adapter_path=adapter_path)
        score, results = self._evaluator(es.samples, responses)
        return score, results, responses

    async def _plan(
        self,
        iteration,
        baseline_score,
        current_score,
        failures,
        training_sample_count: int | None,
        goal: str = "",
    ):
        if isinstance(self._planner, MiniMaxPlanner):
            return await self._planner.plan(
                iteration,
                baseline_score,
                current_score,
                failures,
                training_sample_count,
                goal,
            )
        return (
            self._planner.plan(
                iteration,
                baseline_score,
                current_score,
                failures,
                training_sample_count,
                goal,
            ),
            False,
        )

    def _validate_plan(self, plan: AgentPlan) -> AgentPlan:
        requested = self._config.training_sample_count
        if requested is None:
            return plan

        examples = list(plan.data_spec.examples[:requested])
        while len(examples) < requested:
            examples.append(self._synthetic_training_example(len(examples) + 1))

        data_spec = plan.data_spec.model_copy(update={"examples": examples, "max_examples": requested})
        return plan.model_copy(update={"data_spec": data_spec})

    def _synthetic_training_example(self, index: int) -> dict[str, str]:
        alice_a = 100 + index
        alice_b = 200 + index
        bob = 50 + index
        alice_total = alice_a + alice_b
        revenue = alice_total + bob
        return {
            "prompt": (
                f'Customers: [{{"name":"Alice","purchases":[{alice_a},{alice_b}]}}, '
                f'{{"name":"Bob","purchases":[{bob}]}}]. Summarize.'
            ),
            "completion": (
                f'{{"top_customer":"Alice","top_customer_total":{alice_total},'
                f'"customer_count":2,"total_revenue":{revenue},'
                f'"summary":"Alice leads with ${alice_total} in purchases."}}'
            ),
        }

    def _get_failures(self, rec: RunRecord) -> list[EvalResult]:
        if not rec.iterations:
            return []
        last = rec.iterations[-1]
        return [r for r in last.eval_results if not r.passed]

    def _check_stop(self, rec: RunRecord, score: float) -> StopReason | None:
        cfg = rec.config
        if score >= cfg.target_score:
            return StopReason.TARGET_SCORE
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
