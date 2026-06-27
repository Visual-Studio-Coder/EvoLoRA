"""MongoDB persistence shape and store tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evolora.config import Config
from evolora.models.core import (
    AgentPlan,
    ArtifactMeta,
    IterationResult,
    JudgeReport,
    LoraHyperparams,
    RetrainDecision,
    RunConfig,
    RunRecord,
    TrainingDataSpec,
)
from evolora.persistence.store import (
    FallbackRunStore,
    InMemoryRunStore,
    MongoRunStore,
    get_run_store,
    run_record_from_mongo_doc,
    run_record_to_mongo_doc,
)


def _record_with_iteration() -> RunRecord:
    run_id = "run-test-001"
    plan = AgentPlan(
        hyperparams=LoraHyperparams(
            r=16,
            lora_alpha=32,
            learning_rate=1e-4,
            num_epochs=3,
            batch_size=2,
        ),
        data_spec=TrainingDataSpec(
            examples=[{"prompt": "p", "completion": '{"ok": true}'}],
            max_examples=1,
        ),
        focus_areas=["json_fields"],
    )
    iteration = IterationResult(
        iteration=1,
        plan=plan,
        score=0.72,
        judge_report=JudgeReport(
            rating=0.68,
            summary="Needs better strict JSON.",
            weaknesses=["missing field"],
            recommended_focus=["field coverage"],
            source="digitalocean:test",
            is_mock=False,
        ),
        retrain_decision=RetrainDecision(
            retrain_recommended=True,
            confidence=0.8,
            reason="rating below target",
            suggested_focus=["field coverage"],
            source="minimax:test",
            is_mock=False,
        ),
        artifact=ArtifactMeta(
            run_id=run_id,
            iteration=1,
            adapter_path="./artifacts/run-test-001/adapter",
            score=0.72,
            checksum="abc123",
            is_mock=True,
        ),
    )
    return RunRecord(
        config=RunConfig(run_id=run_id, goal="classify support tickets"),
        baseline_score=0.4,
        best_score=0.72,
        best_iteration=1,
        iterations=[iteration],
    )


class FakeCursor:
    def __init__(self, docs: list[dict]) -> None:
        self._docs = list(docs)
        self._limit = len(self._docs)

    def sort(self, spec: list[tuple[str, int]]) -> FakeCursor:
        for key, direction in reversed(spec):
            self._docs.sort(key=lambda item: str(item.get(key, "")), reverse=direction < 0)
        return self

    def limit(self, limit: int) -> FakeCursor:
        self._limit = limit
        return self

    async def to_list(self, length: int) -> list[dict]:
        return self._docs[: min(length, self._limit)]


class FakeCollection:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.indexes: list[tuple] = []

    async def replace_one(self, filter_doc: dict, doc: dict, *, upsert: bool) -> None:
        assert upsert is True
        self.docs[str(filter_doc["_id"])] = doc

    async def find_one(self, filter_doc: dict) -> dict | None:
        return self.docs.get(str(filter_doc["_id"]))

    def find(self, filter_doc: dict) -> FakeCursor:
        assert filter_doc == {}
        return FakeCursor(list(self.docs.values()))

    async def create_index(self, *args, **kwargs) -> None:
        self.indexes.append((args, kwargs))


def test_mongo_doc_links_run_id_hyperparams_and_judge_report() -> None:
    doc = run_record_to_mongo_doc(_record_with_iteration())

    assert doc["_id"] == "run-test-001"
    assert doc["run_id"] == "run-test-001"
    assert doc["config"]["run_id"] == "run-test-001"
    assert doc["hyperparams_by_iteration"][0]["run_id"] == "run-test-001"
    assert doc["hyperparams_by_iteration"][0]["iteration"] == 1
    assert doc["hyperparams_by_iteration"][0]["hyperparams"]["r"] == 16
    assert doc["hyperparams_by_iteration"][0]["training_example_count"] == 1
    assert doc["judge_reports"][0]["run_id"] == "run-test-001"
    assert doc["judge_reports"][0]["rating"] == 0.68
    assert doc["latest_judge_report"]["iteration"] == 1
    assert doc["latest_retrain_decision"]["retrain_recommended"] is True


def test_mongo_doc_round_trips_to_run_record() -> None:
    original = _record_with_iteration()
    restored = run_record_from_mongo_doc(run_record_to_mongo_doc(original))

    assert restored is not None
    assert restored.run_id == original.run_id
    assert restored.iterations[0].plan.hyperparams.learning_rate == 1e-4
    assert restored.iterations[0].judge_report is not None
    assert restored.iterations[0].judge_report.summary == "Needs better strict JSON."


@pytest.mark.asyncio
async def test_mongo_run_store_upserts_and_reads_records() -> None:
    collection = FakeCollection()
    store = MongoRunStore(collection=collection)
    record = _record_with_iteration()

    await store.save(record)
    restored = await store.get(record.run_id)
    runs = await store.list_runs(limit=10)

    assert restored is not None
    assert restored.run_id == record.run_id
    assert runs[0].run_id == record.run_id
    assert collection.indexes


@pytest.mark.asyncio
async def test_fallback_store_keeps_mock_mode_working_when_mongo_fails() -> None:
    class FailingStore:
        async def save(self, record: RunRecord) -> None:
            raise RuntimeError("mongo down")

        async def get(self, run_id: str) -> RunRecord | None:
            raise RuntimeError("mongo down")

        async def list_runs(self, limit: int = 50) -> list[RunRecord]:
            raise RuntimeError("mongo down")

    record = _record_with_iteration()
    store = FallbackRunStore(FailingStore())

    await store.save(record)

    assert store.last_error is not None
    assert await store.get(record.run_id) == record
    assert [r.run_id for r in await store.list_runs()] == [record.run_id]


def test_get_run_store_selects_mongo_when_uri_is_configured() -> None:
    cfg = Config(
        mongodb_uri="mongodb+srv://example.invalid/evolora",
        mongodb_db_name="custom_db",
        mongodb_runs_collection="custom_runs",
        mongodb_server_selection_timeout_ms=10,
    )
    store = get_run_store(cfg)

    assert isinstance(store, FallbackRunStore)


def test_get_run_store_without_uri_is_in_memory() -> None:
    assert isinstance(get_run_store(SimpleNamespace(mongodb_uri="")), InMemoryRunStore)
