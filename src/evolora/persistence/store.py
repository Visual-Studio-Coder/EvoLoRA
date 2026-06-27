"""Run store protocol and in-memory implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evolora.models.core import RunRecord


@runtime_checkable
class RunStore(Protocol):
    async def save(self, record: RunRecord) -> None: ...
    async def get(self, run_id: str) -> RunRecord | None: ...
    async def list_runs(self, limit: int = 50) -> list[RunRecord]: ...


class InMemoryRunStore:
    """Thread-safe in-memory store; Mongo-serialization-compatible shape."""

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
        """Serialization shape compatible with MongoDB."""
        doc = record.model_dump(mode="json")
        doc["_id"] = doc.pop("config")["run_id"]
        return doc
