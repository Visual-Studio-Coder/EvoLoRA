"""Per-run logging to the gitignored logs/ dir."""

from __future__ import annotations

import json

import pytest

from evolora.demo.task import ADAPTIVE_EVAL_SET, LOCKED_EVAL_SET
from evolora.models.core import RunConfig
from evolora.models.events import Event, EventKind
from evolora.observability.run_logger import RunLogger
from evolora.orchestration.orchestrator import Orchestrator
from evolora.persistence.store import InMemoryRunStore


def test_run_logger_writes_jsonl_and_readable_log(tmp_path):
    logger = RunLogger("abcd1234ef", log_dir=tmp_path)
    logger.log_event(Event(kind=EventKind.RUN_STARTED, run_id="abcd1234ef", message="start"))
    logger.log_event(
        Event(kind=EventKind.LOG, run_id="abcd1234ef", iteration=1, message="hello", data={"k": 1})
    )

    jsonl = list(tmp_path.glob("*.jsonl"))
    txt = list(tmp_path.glob("*.log"))
    assert len(jsonl) == 1 and len(txt) == 1
    assert jsonl[0].name.endswith("-abcd1234.jsonl")  # run_id is truncated to 8 chars

    rows = [json.loads(line) for line in jsonl[0].read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["run_started", "log"]
    assert rows[1]["data"] == {"k": 1}
    assert "hello" in txt[0].read_text(encoding="utf-8")


def test_run_logger_disabled_writes_nothing(tmp_path):
    logger = RunLogger("x", log_dir=tmp_path, enabled=False)
    logger.log_event(Event(kind=EventKind.LOG, run_id="x", message="nope"))
    assert list(tmp_path.iterdir()) == []
    assert logger.path is None


@pytest.mark.asyncio
async def test_orchestrator_writes_run_log(tmp_path):
    cfg = RunConfig(max_iterations=1, target_score=0.99, patience=5)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        adaptive_eval_set=ADAPTIVE_EVAL_SET,
        run_store=InMemoryRunStore(),
    )
    # Point the run's logger at a temp dir (explicit log_dir forces logging on under pytest).
    orch._run_logger = RunLogger(orch._record.run_id, log_dir=tmp_path)

    async for _ in await orch.run():
        pass

    jsonl = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl) == 1
    kinds = [json.loads(line)["kind"] for line in jsonl[0].read_text(encoding="utf-8").splitlines()]
    assert "run_started" in kinds
    assert "baseline_complete" in kinds
