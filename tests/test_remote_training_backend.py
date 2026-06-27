"""RemoteTrainingBackend VM driver tests."""

from __future__ import annotations

import json

import pytest

from evolora.demo.task import LOCKED_EVAL_SET
from evolora.models.core import AgentPlan, LoraHyperparams, RunConfig, TrainingDataSpec
from evolora.training.backends import RemoteTrainingBackend
from evolora.training.remote_config import (
    build_baseline_config_payload,
    build_training_config_payload,
)


def _plan() -> AgentPlan:
    return AgentPlan(
        hyperparams=LoraHyperparams(
            r=8,
            lora_alpha=16,
            learning_rate=2e-4,
            num_epochs=2,
            batch_size=1,
        ),
        data_spec=TrainingDataSpec(
            examples=[{"prompt": "Customers: []", "completion": '{"ok": true}'}],
            max_examples=1,
        ),
    )


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

    def read(self) -> str:
        return self._sftp.files.get(self._path, "")


class FakeSFTP:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.dirs = {"/"}

    def normalize(self, path: str) -> str:
        assert path == "."
        return "/workspace"

    def stat(self, path: str) -> object:
        if path not in self.dirs:
            raise OSError(path)
        return object()

    def mkdir(self, path: str) -> None:
        self.dirs.add(path)

    def file(self, path: str, mode: str) -> FakeRemoteFile:
        assert mode in {"w", "r"}
        return FakeRemoteFile(self, path)

    def close(self) -> None:
        return None


class FakeChannel:
    def __init__(self, status: int = 0) -> None:
        self._status = status

    def recv_exit_status(self) -> int:
        return self._status


class FakeStdout:
    def __init__(self, lines: list[str], status: int = 0) -> None:
        self._lines = lines
        self._index = 0
        self.channel = FakeChannel(status)

    def readlines(self) -> list[str]:
        return self._lines

    def readline(self) -> str:
        if self._index >= len(self._lines):
            return ""
        line = self._lines[self._index]
        self._index += 1
        return line


class FakeStderr:
    def __init__(self, text: str = "") -> None:
        self._text = text

    def read(self) -> bytes:
        return self._text.encode()


class FakeSSHClient:
    def __init__(self) -> None:
        self.sftp = FakeSFTP()
        self.commands: list[str] = []
        self.exec_kwargs: list[dict] = []
        self.connect_kwargs: list[dict] = []
        self.closed_count = 0

    def connect(self, **kwargs) -> None:
        self.connect_kwargs.append(kwargs)

    def open_sftp(self) -> FakeSFTP:
        return self.sftp

    def exec_command(self, command: str, **kwargs):
        self.commands.append(command)
        self.exec_kwargs.append(kwargs)
        if "baseline_evaluate.py" in command:
            return None, FakeStdout(["baseline 1\n", "baseline done\n"]), FakeStderr()
        if "train.py" in command:
            return None, FakeStdout(["train step 1\n", "train done\n"]), FakeStderr()
        return None, FakeStdout(["eval done\n"]), FakeStderr()

    def close(self) -> None:
        self.closed_count += 1


@pytest.mark.asyncio
async def test_remote_training_backend_pushes_files_runs_commands_and_yields_artifact(tmp_path):
    cfg = RunConfig(run_id="run-remote-1", training_backend="remote", goal="strict JSON")
    plan = _plan()
    payload = build_training_config_payload(
        run_id=cfg.run_id,
        iteration=1,
        run_config=cfg,
        plan=plan,
        eval_set=LOCKED_EVAL_SET,
        remote_results_path="/workspace/generations/results.json",
    )
    evaluate_script = tmp_path / "evaluate.py"
    evaluate_script.write_text("print('eval script')\n", encoding="utf-8")
    train_script = tmp_path / "train.py"
    train_script.write_text("print('train script')\n", encoding="utf-8")
    fake_client = FakeSSHClient()
    backend = RemoteTrainingBackend(
        ssh_host="gpu.example.com",
        ssh_user="root",
        ssh_port=2222,
        ssh_key_path="C:/keys/evolora",
        remote_config_path="/workspace/config.json",
        ssh_client_factory=lambda: fake_client,
        evaluate_script_path=evaluate_script,
        train_script_path=train_script,
    )

    stream = await backend.train(
        cfg.run_id,
        1,
        plan,
        cfg.base_model_id,
        remote_payload=payload,
    )
    events = [event async for event in stream]
    final = events[-1]

    assert fake_client.commands == [
        "cd /workspace && python -u train.py",
        "cd /workspace && python -u evaluate.py",
    ]
    assert fake_client.exec_kwargs == [{"get_pty": True}, {"get_pty": True}]
    assert json.loads(fake_client.sftp.files["/workspace/config.json"]) == {
        "learning_rate": 2e-4,
        "lora_alpha": 16,
        "lora_rank": 8,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
    }
    assert "/workspace/data/training_data.jsonl" in fake_client.sftp.files
    assert "/workspace/data/evals.json" in fake_client.sftp.files
    assert fake_client.sftp.files["/workspace/train.py"] == "print('train script')\n"
    assert fake_client.sftp.files["/workspace/evaluate.py"] == "print('eval script')\n"
    assert any(event.get("phase") == "train" for event in events)
    assert any(event.get("phase") == "evaluate" for event in events)
    assert final["done"] is True
    assert final["artifact"].adapter_path == "/workspace/lora_model"
    assert final["artifact"].is_mock is False
    # backend pulls data/evals.json back ([{input, expected}] here; "actual" added on a real VM)
    assert len(final["eval_records"]) == len(LOCKED_EVAL_SET)
    assert "input" in final["eval_records"][0]


@pytest.mark.asyncio
async def test_remote_training_backend_evaluates_base_model_on_vm(tmp_path):
    cfg = RunConfig(run_id="run-remote-1", training_backend="remote", goal="strict JSON")
    payload = build_baseline_config_payload(
        run_id=cfg.run_id,
        run_config=cfg,
        eval_set=LOCKED_EVAL_SET,
        remote_results_path="/workspace/generations/results.json",
    )
    baseline_script = tmp_path / "baseline_evaluate.py"
    baseline_script.write_text("print('baseline script')\n", encoding="utf-8")
    fake_client = FakeSSHClient()
    backend = RemoteTrainingBackend(
        ssh_host="gpu.example.com",
        ssh_user="root",
        ssh_port=2222,
        ssh_key_path="C:/keys/evolora",
        remote_config_path="/workspace/config.json",
        ssh_client_factory=lambda: fake_client,
        baseline_script_path=baseline_script,
    )

    stream = await backend.evaluate_base(payload)
    events = [event async for event in stream]
    final = events[-1]

    assert fake_client.commands == ["cd /workspace && python -u baseline_evaluate.py"]
    assert fake_client.exec_kwargs == [{"get_pty": True}]
    assert json.loads(fake_client.sftp.files["/workspace/config.json"]) == {
        "base_model_id": cfg.base_model_id,
    }
    assert fake_client.sftp.files["/workspace/baseline_evaluate.py"] == "print('baseline script')\n"
    assert any(event.get("message") == "baseline 1" for event in events)
    assert final["done"] is True
    assert len(final["eval_records"]) == len(LOCKED_EVAL_SET)


@pytest.mark.asyncio
async def test_remote_training_backend_requires_ssh_config():
    backend = RemoteTrainingBackend(ssh_host="", ssh_user="", ssh_key_path="")

    stream = await backend.train("run", 1, _plan(), "base", remote_payload={})
    with pytest.raises(RuntimeError, match="SSH_HOST"):
        async for _ in stream:
            pass
