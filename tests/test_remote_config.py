"""Remote GPU config push tests."""

from __future__ import annotations

import json

import pytest

from evolora.demo.task import LOCKED_EVAL_SET
from evolora.models.core import AgentPlan, LoraHyperparams, RunConfig, TrainingDataSpec
from evolora.models.events import EventKind
from evolora.orchestration.orchestrator import Orchestrator
from evolora.training.backends import MockTrainingBackend
from evolora.training.remote_config import (
    RemoteConfigPushResult,
    build_training_config_payload,
    push_config,
)
from evolora.training.runner import MockModelRunner


def _plan() -> AgentPlan:
    return AgentPlan(
        hyperparams=LoraHyperparams(r=16, lora_alpha=32, learning_rate=1e-4),
        data_spec=TrainingDataSpec(
            examples=[{"prompt": "p", "completion": '{"ok": true}'}],
            rationale="cover strict JSON",
            max_examples=1,
        ),
        focus_areas=["json"],
        target_adapter_name="adapter-1",
    )


def test_build_training_config_payload_includes_gpu_inputs() -> None:
    cfg = RunConfig(run_id="run-1", training_backend="remote", goal="strict JSON")
    payload = build_training_config_payload(
        run_id=cfg.run_id,
        iteration=2,
        run_config=cfg,
        plan=_plan(),
        eval_set=LOCKED_EVAL_SET,
        remote_results_path="~/evolora/results.json",
    )

    assert payload["run_id"] == "run-1"
    assert payload["iteration"] == 2
    assert payload["training_backend"] == "remote"
    assert payload["hyperparameters"]["r"] == 16
    assert payload["training_examples"][0]["prompt"] == "p"
    assert payload["remote_results_path"] == "~/evolora/results.json"
    assert payload["eval_set"]["hash"] == LOCKED_EVAL_SET.hash
    assert payload["eval_set"]["prompt_count"] == len(LOCKED_EVAL_SET)
    assert payload["eval_prompts"]
    assert "expected" not in json.dumps(payload["eval_prompts"])


def test_push_config_dry_run_when_ssh_is_unset() -> None:
    result = push_config(
        {"run_id": "run-1", "hyperparameters": {"r": 16}},
        ssh_host="",
        ssh_user="",
        ssh_key_path="",
        remote_config_path="~/evolora/config.json",
    )

    assert result.dry_run is True
    assert result.pushed is False
    assert result.remote_path == "~/evolora/config.json"
    assert "SSH_HOST" in result.message
    assert '"run_id": "run-1"' in result.config_json


class FakeRemoteFile:
    def __init__(self, sftp: FakeSFTP, path: str) -> None:
        self._sftp = sftp
        self._path = path

    def __enter__(self) -> FakeRemoteFile:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def write(self, data: str) -> None:
        self._sftp.files[self._path] = data


class FakeSFTP:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.dirs = {"/", "/home", "/home/tester"}
        self.closed = False

    def normalize(self, path: str) -> str:
        assert path == "."
        return "/home/tester"

    def stat(self, path: str) -> object:
        if path not in self.dirs:
            raise OSError(path)
        return object()

    def mkdir(self, path: str) -> None:
        self.dirs.add(path)

    def file(self, path: str, mode: str) -> FakeRemoteFile:
        assert mode == "w"
        return FakeRemoteFile(self, path)

    def close(self) -> None:
        self.closed = True


class FakeSSHClient:
    def __init__(self) -> None:
        self.sftp = FakeSFTP()
        self.connect_kwargs: dict = {}
        self.closed = False

    def connect(self, **kwargs) -> None:
        self.connect_kwargs = kwargs

    def open_sftp(self) -> FakeSFTP:
        return self.sftp

    def close(self) -> None:
        self.closed = True


def test_push_config_writes_json_over_sftp() -> None:
    fake_client = FakeSSHClient()

    result = push_config(
        {"run_id": "run-2", "hyperparameters": {"r": 8}},
        ssh_host="gpu.example.com",
        ssh_user="trainer",
        ssh_port=2222,
        ssh_key_path="C:/Users/emmad/.ssh/evolora",
        remote_config_path="~/evolora/config.json",
        ssh_client_factory=lambda: fake_client,
    )

    assert result.dry_run is False
    assert result.pushed is True
    assert result.remote_path == "/home/tester/evolora/config.json"
    assert fake_client.connect_kwargs["hostname"] == "gpu.example.com"
    assert fake_client.connect_kwargs["username"] == "trainer"
    assert fake_client.connect_kwargs["port"] == 2222
    assert fake_client.connect_kwargs["key_filename"] == "C:/Users/emmad/.ssh/evolora"
    assert "/home/tester/evolora" in fake_client.sftp.dirs
    assert '"run_id": "run-2"' in fake_client.sftp.files["/home/tester/evolora/config.json"]
    assert fake_client.sftp.closed is True
    assert fake_client.closed is True


@pytest.mark.asyncio
async def test_orchestrator_pushes_remote_config_before_training(monkeypatch) -> None:
    captured: dict = {}

    def fake_push(config: dict) -> RemoteConfigPushResult:
        captured["config"] = config
        return RemoteConfigPushResult(
            dry_run=True,
            pushed=False,
            remote_path="~/evolora/config.json",
            byte_count=123,
            message="Remote config dry-run: missing SSH_HOST",
            config_json="{}",
        )

    monkeypatch.setattr("evolora.orchestration.orchestrator.push_config", fake_push)
    cfg = RunConfig(max_iterations=1, training_backend="remote", target_score=1.0)
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        training_backend=MockTrainingBackend(),
        model_runner=MockModelRunner(),
    )

    events = []
    async for event in await orch.run():
        events.append(event)

    assert captured["config"]["run_id"] == cfg.run_id
    assert captured["config"]["training_backend"] == "remote"
    assert captured["config"]["training_examples"]
    assert captured["config"]["eval_prompts"]
    assert captured["config"]["eval_set"]["hash"] == LOCKED_EVAL_SET.hash
    assert "expected" not in json.dumps(captured["config"]["eval_prompts"])
    assert any(
        event.kind == EventKind.LOG and "Remote config dry-run" in event.message
        for event in events
    )
