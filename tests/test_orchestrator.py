"""Integration tests for the orchestrator state machine."""

import pytest

from evolora.agent.planner import HeuristicPlanner, MiniMaxPlanner
from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET
from evolora.models.core import (
    AgentPlan,
    ArtifactMeta,
    JudgeReport,
    RetrainDecision,
    RunConfig,
    RunStatus,
    StopReason,
)
from evolora.models.events import EventKind
from evolora.orchestration.orchestrator import Orchestrator
from evolora.persistence.store import InMemoryRunStore


async def _collect(orch: Orchestrator) -> tuple[list, object]:
    events = []
    async for ev in await orch.run():
        events.append(ev)
    return events, orch._record


def _make_orch(**kwargs) -> Orchestrator:
    defaults = dict(max_iterations=2, target_score=0.99, patience=5)
    defaults.update(kwargs)
    cfg = RunConfig(**defaults)
    return Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        run_store=InMemoryRunStore(),
    )


class StaticJudge:
    is_mock = True

    def __init__(self, rating: float = 0.4) -> None:
        self.rating = rating

    async def judge(self, **kwargs):
        return JudgeReport(
            rating=self.rating,
            summary="Needs tighter JSON specialization.",
            weaknesses=["field accuracy"],
            recommended_focus=["field accuracy"],
            source="test_judge",
            is_mock=True,
        )


class StaticRetrainAdvisor:
    is_mock = True

    def __init__(self, retrain: bool) -> None:
        self.retrain = retrain

    async def decide(self, **kwargs):
        return RetrainDecision(
            retrain_recommended=self.retrain,
            confidence=0.8,
            reason="test decision",
            suggested_focus=["field accuracy"],
            source="test_advisor",
            is_mock=True,
        )


class EvalGatePlanner(MiniMaxPlanner):
    def __init__(self) -> None:
        pass

    async def generate_evals(self, goal: str, count: int = 5) -> list[dict]:
        return [{"prompt": f"{goal}: produce JSON", "expected": {"ok": True}}]

    async def plan(self, *args, **kwargs):
        return HeuristicPlanner().plan(*args, **kwargs), False


class FallbackReasonPlanner(MiniMaxPlanner):
    def __init__(self) -> None:
        self.last_error = "BadRequestError: invalid function arguments json string"

    async def plan(self, *args, **kwargs):
        return HeuristicPlanner().plan(*args, **kwargs), True


class StaticLLMJudge:
    async def judge(self, records: list[dict]) -> tuple[float, list[dict]]:
        scored = []
        scores = []
        for record in records:
            score = 2 if record.get("actual") == "base" else 8
            scored.append({**record, "score": score, "reason": "test judge"})
            scores.append(score)
        return sum(scores) / len(scores) / 10.0, scored


class RemoteBackendWithBaseline:
    is_mock = False
    name = "remote"

    def __init__(self) -> None:
        self.baseline_payload: dict | None = None
        self.train_payload: dict | None = None

    async def evaluate_base(self, remote_payload=None):
        self.baseline_payload = remote_payload

        async def stream():
            yield {"phase": "baseline", "message": "baseline live log", "done": False}
            yield {
                "done": True,
                "eval_records": [{"input": "q", "expected": "a", "actual": "base"}],
            }

        return stream()

    async def train(
        self,
        run_id: str,
        iteration: int,
        plan: AgentPlan,
        base_model_id: str,
        remote_payload=None,
    ):
        self.train_payload = remote_payload

        async def stream():
            yield {"phase": "train", "message": "unsloth step 1", "done": False}
            yield {
                "done": True,
                "artifact": ArtifactMeta(
                    run_id=run_id,
                    iteration=iteration,
                    adapter_path="/workspace/lora_model",
                    score=0.0,
                    checksum="remote-test",
                    is_mock=False,
                ),
                "eval_records": [{"input": "q", "expected": "a", "actual": "trained"}],
                "cost_usd": 0.0,
                "duration_s": 0.0,
            }

        return stream()

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_happy_path_completes():
    orch = _make_orch()
    events, rec = await _collect(orch)
    assert rec.status in {RunStatus.COMPLETE, RunStatus.FAILED}
    kinds = [e.kind for e in events]
    assert EventKind.RUN_STARTED in kinds
    assert EventKind.EVAL_SET_LOCKED in kinds
    assert EventKind.BASELINE_COMPLETE in kinds


@pytest.mark.asyncio
async def test_max_iterations_stop():
    orch = _make_orch(max_iterations=1, target_score=1.0, patience=99)
    events, rec = await _collect(orch)
    assert rec.status == RunStatus.COMPLETE
    assert rec.stop_reason == StopReason.MAX_ITERATIONS


@pytest.mark.asyncio
async def test_cancellation():
    orch = _make_orch(max_iterations=5)
    orch.cancel()
    events, rec = await _collect(orch)
    assert rec.status == RunStatus.CANCELLED
    assert rec.stop_reason == StopReason.CANCELLED


@pytest.mark.asyncio
async def test_event_ordering():
    orch = _make_orch(max_iterations=1)
    events, _ = await _collect(orch)
    kinds = [e.kind for e in events]
    assert kinds[0] == EventKind.RUN_STARTED
    assert kinds[1] == EventKind.EVAL_SET_LOCKED
    assert kinds[2] == EventKind.STATUS_CHANGED  # baseline status
    assert kinds[3] == EventKind.BASELINE_COMPLETE


@pytest.mark.asyncio
async def test_generated_evals_require_user_approval_before_locking():
    cfg = RunConfig(max_iterations=1, target_score=1.0, goal="custom receipt parser")
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        planner=EvalGatePlanner(),
        run_store=InMemoryRunStore(),
    )

    events = []
    async for event in await orch.run():
        events.append(event)
        if event.kind == EventKind.EVAL_APPROVAL_REQUIRED:
            assert event.data["evals"] == [
                {
                    "input": "custom receipt parser: produce JSON",
                    "expected": '{"ok": true}',
                }
            ]
            orch.submit_retrain_approval(False)

    kinds = [event.kind for event in events]
    assert EventKind.EVAL_APPROVAL_REQUIRED in kinds
    assert EventKind.EVAL_SET_LOCKED not in kinds
    assert orch._record.status == RunStatus.CANCELLED
    assert orch._record.stop_reason == StopReason.CANCELLED


@pytest.mark.asyncio
async def test_remote_baseline_uses_vm_records_and_llm_judge():
    backend = RemoteBackendWithBaseline()
    cfg = RunConfig(max_iterations=1, target_score=1.0, training_backend="remote")
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        training_backend=backend,
        llm_judge=StaticLLMJudge(),
        run_store=InMemoryRunStore(),
    )

    events, rec = await _collect(orch)
    log_messages = [event.message for event in events if event.kind == EventKind.LOG]

    assert backend.baseline_payload is not None
    assert backend.train_payload is not None
    assert rec.baseline_score == 0.2
    assert rec.iterations[0].score == 0.8
    assert "[remote:baseline] baseline live log" in log_messages
    assert "[remote:train] unsloth step 1" in log_messages


@pytest.mark.asyncio
async def test_fallback_event_includes_minimax_failure_reason():
    cfg = RunConfig(max_iterations=1, target_score=1.0)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        planner=FallbackReasonPlanner(),
        run_store=InMemoryRunStore(),
    )

    events, _ = await _collect(orch)
    fallback_event = next(event for event in events if event.kind == EventKind.AGENT_FALLBACK_USED)

    assert "invalid function arguments" in fallback_event.message
    assert fallback_event.data["reason"] == "BadRequestError: invalid function arguments json string"


@pytest.mark.asyncio
async def test_best_iteration_preserved():
    orch = _make_orch(max_iterations=2, target_score=1.0)
    _, rec = await _collect(orch)
    # best_score should be >= baseline
    assert rec.best_score >= rec.baseline_score


@pytest.mark.asyncio
async def test_exact_training_sample_count_is_enforced():
    orch = _make_orch(max_iterations=1, target_score=1.0, training_sample_count=12)
    _, rec = await _collect(orch)
    assert rec.iterations
    assert len(rec.iterations[0].plan.data_spec.examples) == 12
    assert rec.iterations[0].plan.data_spec.max_examples == 12


@pytest.mark.asyncio
async def test_training_examples_stack_across_iterations():
    cfg = RunConfig(max_iterations=2, target_score=1.0, training_sample_count=10)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=StaticJudge(rating=0.3),
        retrain_advisor=StaticRetrainAdvisor(retrain=True),
        run_store=InMemoryRunStore(),
    )
    _, rec = await _collect(orch)

    assert len(rec.iterations) == 2
    # Each iteration adds training_sample_count more examples (stacking): 10 -> 20.
    assert len(rec.iterations[0].plan.data_spec.examples) == 10
    assert len(rec.iterations[1].plan.data_spec.examples) == 20
    assert rec.iterations[1].plan.data_spec.max_examples == 20


@pytest.mark.asyncio
async def test_no_secrets_in_prompts():
    """Eval prompts passed to runner must not contain expected answers."""
    prompts = LOCKED_EVAL_SET.prompts_only()
    for p in prompts:
        assert "expected" not in p
        assert "top_customer_total" not in str(p)


@pytest.mark.asyncio
async def test_in_memory_store_persists():
    orch = _make_orch(max_iterations=1)
    _, rec = await _collect(orch)
    stored = await orch._store.get(rec.run_id)
    assert stored is not None
    assert stored.run_id == rec.run_id


@pytest.mark.asyncio
async def test_mongo_serialization_shape():
    orch = _make_orch(max_iterations=1)
    _, rec = await _collect(orch)
    from evolora.persistence.store import InMemoryRunStore
    store = orch._store
    assert isinstance(store, InMemoryRunStore)
    doc = store.to_mongo_doc(rec)
    assert "_id" in doc
    assert doc["_id"] == rec.run_id


@pytest.mark.asyncio
async def test_judge_and_retrain_decision_are_recorded():
    cfg = RunConfig(max_iterations=1, target_score=1.0)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=StaticJudge(rating=0.42),
        retrain_advisor=StaticRetrainAdvisor(retrain=True),
    )
    events, rec = await _collect(orch)
    kinds = [e.kind for e in events]

    assert EventKind.JUDGE_STARTED in kinds
    assert EventKind.JUDGE_COMPLETE in kinds
    assert EventKind.RETRAIN_DECISION_RECEIVED in kinds
    assert rec.iterations[0].judge_report is not None
    assert rec.iterations[0].judge_report.rating == 0.42
    assert rec.iterations[0].retrain_decision is not None
    assert rec.iterations[0].retrain_decision.retrain_recommended is True


@pytest.mark.asyncio
async def test_user_can_decline_retrain_after_judge_recommendation():
    cfg = RunConfig(max_iterations=2, target_score=1.0, require_retrain_approval=True)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=StaticJudge(rating=0.25),
        retrain_advisor=StaticRetrainAdvisor(retrain=True),
    )
    events = []
    async for event in await orch.run():
        events.append(event)
        if event.kind == EventKind.USER_APPROVAL_REQUIRED:
            orch.submit_retrain_approval(False)

    assert EventKind.USER_APPROVAL_REQUIRED in [e.kind for e in events]
    assert orch._record.status == RunStatus.COMPLETE
    assert orch._record.stop_reason == StopReason.USER_DECLINED_RETRAIN


@pytest.mark.asyncio
async def test_judge_acceptance_stops_without_more_training():
    cfg = RunConfig(max_iterations=3, target_score=1.0)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        judge=StaticJudge(rating=0.91),
        retrain_advisor=StaticRetrainAdvisor(retrain=False),
    )
    _, rec = await _collect(orch)

    assert len(rec.iterations) == 1
    assert rec.status == RunStatus.COMPLETE
    assert rec.stop_reason == StopReason.JUDGE_ACCEPTED
