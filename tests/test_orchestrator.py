"""Integration tests for the orchestrator state machine."""

import pytest

from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET
from evolora.models.core import RunConfig, RunStatus, StopReason
from evolora.models.events import EventKind
from evolora.orchestration.orchestrator import Orchestrator


async def _collect(orch: Orchestrator) -> tuple[list, object]:
    events = []
    async for ev in await orch.run():
        events.append(ev)
    return events, orch._record


def _make_orch(**kwargs) -> Orchestrator:
    defaults = dict(max_iterations=2, target_score=0.99, patience=5)
    defaults.update(kwargs)
    cfg = RunConfig(**defaults)
    return Orchestrator(config=cfg, eval_set=LOCKED_EVAL_SET, adaptive_eval_set=ADAPTIVE_EVAL_SET)


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
async def test_no_secrets_in_prompts():
    """Eval prompts passed to runner must not contain expected answers."""
    orch = _make_orch(max_iterations=1)
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
