"""Remote GPU config push tests."""

from __future__ import annotations

import json

import pytest

from evolora.demo.task import LOCKED_EVAL_SET
from evolora.models.core import (
    AgentPlan,
    ArtifactMeta,
    LoraHyperparams,
    RunConfig,
    TrainingDataSpec,
)
from evolora.orchestration.orchestrator import Orchestrator
from evolora.training.remote_config import (
    DEFAULT_REMOTE_CONFIG_PATH,
    DEFAULT_REMOTE_EVALS_PATH,
    DEFAULT_REMOTE_TRAINING_DATA_PATH,
    build_baseline_config_payload,
    build_training_config_payload,
    push_config,
    render_remote_files,
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
    assert payload["vm_config"] == {
        "learning_rate": 1e-4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
    }
    assert payload["training_data"][0] == {
        "instruction": "strict JSON",
        "input": "p",
        "output": '{"ok": true}',
    }
    assert payload["remote_results_path"] == "~/evolora/results.json"
    assert payload["eval_set"]["hash"] == LOCKED_EVAL_SET.hash
    assert payload["eval_set"]["prompt_count"] == len(LOCKED_EVAL_SET)
    assert payload["eval_prompts"]
    assert set(payload["eval_prompts"][0]) == {"input", "expected"}


def test_render_remote_files_matches_vm_contract() -> None:
    cfg = RunConfig(run_id="run-1", training_backend="remote", goal="strict JSON")
    payload = build_training_config_payload(
        run_id=cfg.run_id,
        iteration=1,
        run_config=cfg,
        plan=_plan(),
        eval_set=LOCKED_EVAL_SET,
        remote_results_path="/workspace/generations/results.json",
    )

    files = render_remote_files(payload, remote_config_path="/workspace/config.json")
    vm_config = json.loads(files[DEFAULT_REMOTE_CONFIG_PATH])
    training_rows = [
        json.loads(line)
        for line in files[DEFAULT_REMOTE_TRAINING_DATA_PATH].splitlines()
        if line
    ]
    evals = json.loads(files[DEFAULT_REMOTE_EVALS_PATH])

    assert vm_config["lora_rank"] == 16
    assert "run_id" not in vm_config
    assert training_rows == [
        {"instruction": "strict JSON", "input": "p", "output": '{"ok": true}'}
    ]
    assert evals
    assert set(evals[0]) == {"input", "expected"}


def test_build_baseline_config_payload_renders_base_model_eval_files() -> None:
    cfg = RunConfig(
        run_id="run-1",
        training_backend="remote",
        base_model_id="unsloth/Phi-3-mini-4k-instruct",
        goal="strict JSON",
    )
    payload = build_baseline_config_payload(
        run_id=cfg.run_id,
        run_config=cfg,
        eval_set=LOCKED_EVAL_SET,
        remote_results_path="/workspace/generations/results.json",
    )

    files = render_remote_files(payload, remote_config_path="/workspace/config.json")
    vm_config = json.loads(files[DEFAULT_REMOTE_CONFIG_PATH])
    evals = json.loads(files[DEFAULT_REMOTE_EVALS_PATH])

    assert payload["iteration"] == 0
    assert vm_config == {"base_model_id": "unsloth/Phi-3-mini-4k-instruct"}
    assert files[DEFAULT_REMOTE_TRAINING_DATA_PATH] == ""
    assert evals
    assert set(evals[0]) == {"input", "expected"}


def test_push_config_dry_run_when_ssh_is_unset() -> None:
    result = push_config(
        {"vm_config": {"lora_rank": 16}, "training_data": [], "eval_prompts": []},
        ssh_host="",
        ssh_user="",
        ssh_key_path="",
        remote_config_path="/workspace/config.json",
    )

    assert result.dry_run is True
    assert result.pushed is False
    assert result.remote_path == "/workspace/config.json"
    assert "SSH_HOST" in result.message
    assert '"lora_rank": 16' in result.config_json
    assert set(result.files) == {
        DEFAULT_REMOTE_CONFIG_PATH,
        DEFAULT_REMOTE_TRAINING_DATA_PATH,
        DEFAULT_REMOTE_EVALS_PATH,
    }


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
        self.dirs = {"/"}
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
        {"vm_config": {"lora_rank": 8}, "training_data": [], "eval_prompts": []},
        ssh_host="gpu.example.com",
        ssh_user="trainer",
        ssh_port=2222,
        ssh_key_path="C:/Users/emmad/.ssh/evolora",
        remote_config_path="/workspace/config.json",
        ssh_client_factory=lambda: fake_client,
    )

    assert result.dry_run is False
    assert result.pushed is True
    assert result.remote_path == "/workspace/config.json"
    assert fake_client.connect_kwargs["hostname"] == "gpu.example.com"
    assert fake_client.connect_kwargs["username"] == "trainer"
    assert fake_client.connect_kwargs["port"] == 2222
    assert fake_client.connect_kwargs["key_filename"] == "C:/Users/emmad/.ssh/evolora"
    assert "/workspace/data" in fake_client.sftp.dirs
    assert '"lora_rank": 8' in fake_client.sftp.files["/workspace/config.json"]
    assert "/workspace/data/training_data.jsonl" in fake_client.sftp.files
    assert "/workspace/data/evals.json" in fake_client.sftp.files
    assert fake_client.sftp.closed is True
    assert fake_client.closed is True


class CapturingRemoteBackend:
    is_mock = False
    name = "remote"

    def __init__(self) -> None:
        self.payload: dict | None = None

    async def train(self, run_id, iteration, plan, base_model_id, remote_payload=None):
        self.payload = remote_payload

        async def stream():
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
                "cost_usd": 0.0,
                "duration_s": 0.0,
            }

        return stream()

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_orchestrator_passes_remote_payload_to_backend() -> None:
    cfg = RunConfig(max_iterations=1, training_backend="remote", target_score=1.0)
    backend = CapturingRemoteBackend()
    orch = Orchestrator(
        config=cfg,
        eval_set=LOCKED_EVAL_SET,
        training_backend=backend,
        model_runner=MockModelRunner(),
    )

    async for _ in await orch.run():
        pass

    assert backend.payload is not None
    assert backend.payload["run_id"] == cfg.run_id
    assert backend.payload["training_backend"] == "remote"
    assert backend.payload["training_data"]
    assert backend.payload["eval_prompts"]
    assert backend.payload["eval_set"]["hash"] == LOCKED_EVAL_SET.hash
    assert set(backend.payload["eval_prompts"][0]) == {"input", "expected"}
