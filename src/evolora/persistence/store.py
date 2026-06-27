"""Run store protocol plus in-memory and MongoDB implementations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from evolora.models.core import RunRecord


@runtime_checkable
class RunStore(Protocol):
    async def save(self, record: RunRecord) -> None: ...
    async def get(self, run_id: str) -> RunRecord | None: ...
    async def list_runs(self, limit: int = 50) -> list[RunRecord]: ...


class InMemoryRunStore:
    """Process-local store used for tests and no-key mock runs."""

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._order: list[str] = []

    async def save(self, record: RunRecord) -> None:
        rid = record.run_id
        if rid not in self._records:
            self._order.append(rid)
        self._records[rid] = record

    async def get(self, run_id: str) -> RunRecord | None:
        return self._records.get(run_id)

    async def list_runs(self, limit: int = 50) -> list[RunRecord]:
        return [self._records[rid] for rid in reversed(self._order)][:limit]

    def to_mongo_doc(self, record: RunRecord) -> dict:
        """Backward-compatible helper for tests and older callers."""
        return run_record_to_mongo_doc(record)


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _iteration_link(run_id: str, iteration: dict[str, Any]) -> dict[str, Any]:
    plan = iteration.get("plan") or {}
    data_spec = plan.get("data_spec") or {}
    artifact = iteration.get("artifact") or {}
    return {
        "run_id": run_id,
        "iteration": iteration.get("iteration"),
        "hyperparams": plan.get("hyperparams") or {},
        "training_example_count": len(data_spec.get("examples") or []),
        "focus_areas": plan.get("focus_areas") or [],
        "score": iteration.get("score"),
        "adaptive_score": iteration.get("adaptive_score"),
        "judge_report": iteration.get("judge_report"),
        "retrain_decision": iteration.get("retrain_decision"),
        "artifact_id": artifact.get("artifact_id") if isinstance(artifact, dict) else None,
        "adapter_path": artifact.get("adapter_path") if isinstance(artifact, dict) else None,
    }


def run_record_to_mongo_doc(record: RunRecord) -> dict[str, Any]:
    """Serialize a run with query-friendly run/iteration links.

    The full Pydantic record is preserved, while top-level derived fields make
    MongoDB Atlas queries straightforward during the hackathon:
    - ``run_id`` is duplicated from ``config.run_id`` and used as ``_id``.
    - ``hyperparams_by_iteration`` links every LoRA config to the same run id.
    - ``judge_reports`` links every LLM-as-judge result to run id + iteration.
    """
    doc = record.model_dump(mode="json")
    run_id = record.run_id
    iterations = doc.get("iterations") or []
    hyperparams_by_iteration = [_iteration_link(run_id, item) for item in iterations]

    judge_reports = []
    for item in hyperparams_by_iteration:
        report = item.get("judge_report")
        if report:
            judge_reports.append({
                "run_id": run_id,
                "iteration": item.get("iteration"),
                **report,
            })

    doc.update({
        "_id": run_id,
        "run_id": run_id,
        "task_name": doc.get("config", {}).get("task_name"),
        "goal": doc.get("config", {}).get("goal"),
        "created_at": doc.get("config", {}).get("created_at"),
        "updated_at": _utc_iso(),
        "persistence_version": 1,
        "iteration_count": len(iterations),
        "hyperparams_by_iteration": hyperparams_by_iteration,
        "judge_reports": judge_reports,
        "latest_hyperparams": (
            hyperparams_by_iteration[-1]["hyperparams"] if hyperparams_by_iteration else None
        ),
        "latest_judge_report": judge_reports[-1] if judge_reports else None,
        "latest_retrain_decision": (
            hyperparams_by_iteration[-1]["retrain_decision"]
            if hyperparams_by_iteration
            else None
        ),
    })
    return doc


def run_record_from_mongo_doc(doc: dict[str, Any] | None) -> RunRecord | None:
    if doc is None:
        return None
    payload = dict(doc)
    for key in (
        "_id",
        "run_id",
        "task_name",
        "goal",
        "created_at",
        "updated_at",
        "persistence_version",
        "iteration_count",
        "hyperparams_by_iteration",
        "judge_reports",
        "latest_hyperparams",
        "latest_judge_report",
        "latest_retrain_decision",
    ):
        payload.pop(key, None)
    return RunRecord.model_validate(payload)


class MongoRunStore:
    """MongoDB Atlas-backed run store.

    Stores one document per EvoLoRA run. The run id is the Mongo ``_id``, so all
    hyperparameter, artifact, evaluation, and judge information remains linked.
    """

    def __init__(
        self,
        uri: str = "",
        *,
        db_name: str = "evolora",
        collection_name: str = "runs",
        server_selection_timeout_ms: int = 3000,
        collection: Any | None = None,
    ) -> None:
        self._client = None
        self._indexes_ready = False
        if collection is not None:
            self._collection = collection
            return
        if not uri:
            raise ValueError("MongoRunStore requires a MongoDB URI")
        from motor.motor_asyncio import AsyncIOMotorClient

        self._client = AsyncIOMotorClient(
            uri,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
        )
        self._collection = self._client[db_name][collection_name]

    async def save(self, record: RunRecord) -> None:
        await self._ensure_indexes()
        doc = run_record_to_mongo_doc(record)
        await self._collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    async def get(self, run_id: str) -> RunRecord | None:
        doc = await self._collection.find_one({"_id": run_id})
        return run_record_from_mongo_doc(doc)

    async def list_runs(self, limit: int = 50) -> list[RunRecord]:
        cursor = (
            self._collection.find({})
            .sort([("started_at", -1), ("updated_at", -1)])
            .limit(limit)
        )
        docs = await cursor.to_list(length=limit)
        return [record for doc in docs if (record := run_record_from_mongo_doc(doc))]

    async def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        await self._collection.create_index([("run_id", 1)], unique=True)
        await self._collection.create_index([("status", 1), ("updated_at", -1)])
        await self._collection.create_index([("config.created_at", -1)])
        await self._collection.create_index([
            ("hyperparams_by_iteration.run_id", 1),
            ("hyperparams_by_iteration.iteration", 1),
        ])
        await self._collection.create_index([
            ("judge_reports.run_id", 1),
            ("judge_reports.iteration", 1),
        ])
        self._indexes_ready = True

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


class FallbackRunStore:
    """Use MongoDB when available, but keep mock mode functional if it fails."""

    def __init__(self, primary: RunStore, fallback: InMemoryRunStore | None = None) -> None:
        self._primary = primary
        self._fallback = fallback or InMemoryRunStore()
        self._primary_failed = False
        self.last_error: str | None = None

    async def save(self, record: RunRecord) -> None:
        await self._fallback.save(record)
        if self._primary_failed:
            return
        try:
            await self._primary.save(record)
        except Exception as exc:  # pragma: no cover - network failure path
            self._primary_failed = True
            self.last_error = f"{exc.__class__.__name__}: {exc}"

    async def get(self, run_id: str) -> RunRecord | None:
        if not self._primary_failed:
            try:
                record = await self._primary.get(run_id)
                if record is not None:
                    return record
            except Exception as exc:  # pragma: no cover - network failure path
                self._primary_failed = True
                self.last_error = f"{exc.__class__.__name__}: {exc}"
        return await self._fallback.get(run_id)

    async def list_runs(self, limit: int = 50) -> list[RunRecord]:
        if not self._primary_failed:
            try:
                return await self._primary.list_runs(limit)
            except Exception as exc:  # pragma: no cover - network failure path
                self._primary_failed = True
                self.last_error = f"{exc.__class__.__name__}: {exc}"
        return await self._fallback.list_runs(limit)


def get_run_store(config: Any | None = None) -> RunStore:
    """Return MongoDB-backed persistence when configured, otherwise in-memory."""
    if config is None:
        from evolora.config import get_config

        config = get_config()

    if getattr(config, "mongodb_uri", ""):
        mongo = MongoRunStore(
            config.mongodb_uri,
            db_name=getattr(config, "mongodb_db_name", "evolora"),
            collection_name=getattr(config, "mongodb_runs_collection", "runs"),
            server_selection_timeout_ms=getattr(
                config,
                "mongodb_server_selection_timeout_ms",
                3000,
            ),
        )
        return FallbackRunStore(mongo)
    return InMemoryRunStore()
