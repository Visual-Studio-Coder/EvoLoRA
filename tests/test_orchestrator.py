"""Integration tests for the orchestrator state machine."""

import pytest

from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET
from evolora.models.core import JudgeReport, RetrainDecision, RunConfig, RunStatus, StopReason
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
