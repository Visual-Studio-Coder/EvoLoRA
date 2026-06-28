"""Remote training config push over SSH/SFTP."""

from __future__ import annotations

import copy
import json
import posixpath
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from evolora.config import get_config
from evolora.evaluation.locked import LockedEvalSet
from evolora.models.core import AgentPlan, RunConfig

DEFAULT_REMOTE_CONFIG_PATH = "/workspace/config.json"
DEFAULT_REMOTE_TRAINING_DATA_PATH = "/workspace/data/training_data.jsonl"
DEFAULT_REMOTE_EVALS_PATH = "/workspace/data/evals.json"


class RemoteConfigPushError(RuntimeError):
    """Raised when a configured SSH push fails."""


class RemoteConfigPushResult(BaseModel):
    dry_run: bool
    pushed: bool
    remote_path: str
    byte_count: int
    message: str
    config_json: str = ""
    files: dict[str, str] = Field(default_factory=dict)
    remote_paths: dict[str, str] = Field(default_factory=dict)


def build_training_config_payload(
    *,
    run_id: str,
    iteration: int,
    run_config: RunConfig,
    plan: AgentPlan,
    eval_set: LockedEvalSet,
    remote_results_path: str | None = None,
) -> dict[str, Any]:
    """Build the VM-specific files consumed by the remote GPU trainer.

    The current VM expects three concrete files:
    - /workspace/config.json with bare train.py hyperparameter keys
    - /workspace/data/training_data.jsonl in Alpaca JSONL format
    - /workspace/data/evals.json with prompts only
    """
    cfg = get_config()
    instruction = run_config.goal or run_config.task_name
    examples = copy.deepcopy(plan.data_spec.examples)
    hp = plan.hyperparams

    vm_config = {
        # train.py reads base_model_id from config.json; without it, it falls back to its
        # hardcoded default (the wrong model). Must be written so the SELECTED model is trained.
        "base_model_id": run_config.base_model_id,
        "learning_rate": hp.learning_rate,
        "lora_rank": hp.r,
        "lora_alpha": hp.lora_alpha,
        "num_train_epochs": hp.num_epochs,
        "per_device_train_batch_size": hp.batch_size,
    }
    training_data = [
        {
            "instruction": instruction,
            "input": example.get("prompt", ""),
            "output": example.get("completion", ""),
        }
        for example in examples
    ]
    eval_prompts = _eval_records(eval_set)
    results_path = remote_results_path or cfg.remote_results_path

    return {
        "schema_version": 2,
        "source": "evolora",
        "run_id": run_id,
        "iteration": iteration,
        "task_name": run_config.task_name,
        "goal": run_config.goal,
        "base_model_id": run_config.base_model_id,
        "training_backend": run_config.training_backend,
        "target_adapter_name": plan.target_adapter_name,
        "vm_config": vm_config,
        "training_data": training_data,
        "training_example_count": len(training_data),
        "training_data_rationale": plan.data_spec.rationale,
        "focus_areas": list(plan.focus_areas),
        "remote_results_path": results_path,
        "eval_prompts": eval_prompts,
        "eval_set": {
            "hash": eval_set.hash,
            "prompt_count": len(eval_set),
        },
        "paths": {
            "config": cfg.remote_config_path or DEFAULT_REMOTE_CONFIG_PATH,
            "training_data": DEFAULT_REMOTE_TRAINING_DATA_PATH,
            "evals": DEFAULT_REMOTE_EVALS_PATH,
            "results": results_path,
        },
    }


def build_baseline_config_payload(
    *,
    run_id: str,
    run_config: RunConfig,
    eval_set: LockedEvalSet,
    remote_results_path: str | None = None,
) -> dict[str, Any]:
    """Build the VM files for a remote base-model evaluation before training."""
    cfg = get_config()
    results_path = remote_results_path or cfg.remote_results_path
    return {
        "schema_version": 2,
        "source": "evolora",
        "run_id": run_id,
        "iteration": 0,
        "task_name": run_config.task_name,
        "goal": run_config.goal,
        "base_model_id": run_config.base_model_id,
        "training_backend": run_config.training_backend,
        "target_adapter_name": "base-model",
        "vm_config": {
            "base_model_id": run_config.base_model_id,
        },
        "training_data": [],
        "training_example_count": 0,
        "training_data_rationale": "baseline evaluation only",
        "focus_areas": [],
        "remote_results_path": results_path,
        "eval_prompts": _eval_records(eval_set),
        "eval_set": {
            "hash": eval_set.hash,
            "prompt_count": len(eval_set),
        },
        "paths": {
            "config": cfg.remote_config_path or DEFAULT_REMOTE_CONFIG_PATH,
            "training_data": DEFAULT_REMOTE_TRAINING_DATA_PATH,
            "evals": DEFAULT_REMOTE_EVALS_PATH,
            "results": results_path,
        },
    }


def _eval_records(eval_set: LockedEvalSet) -> list[dict[str, str]]:
    # VM guy's evaluate.py reads data/evals.json as [{input, expected}], fills "actual"
    # in place, and we pull it back to score with the LLM-judge. Keep expected as a string.
    return [
        {
            "input": sample.prompt,
            "expected": (
                json.dumps(sample.expected, sort_keys=True)
                if isinstance(sample.expected, (dict, list))
                else str(sample.expected)
            ),
        }
        for sample in eval_set.samples
    ]


def push_config(
    config: dict[str, Any],
    *,
    ssh_host: str | None = None,
    ssh_user: str | None = None,
    ssh_port: int | str | None = None,
    ssh_key_path: str | None = None,
    remote_config_path: str | None = None,
    ssh_client_factory: Callable[[], Any] | None = None,
) -> RemoteConfigPushResult:
    """Serialize VM files and push them to the GPU host.

    Missing SSH host/user/key means dry-run mode. Dry-run returns the file
    contents that would be pushed, but does not attempt network access.
    """
    cfg = get_config()
    host = cfg.ssh_host if ssh_host is None else ssh_host
    user = cfg.ssh_user if ssh_user is None else ssh_user
    port = int(cfg.ssh_port if ssh_port is None else ssh_port)
    key_path = cfg.ssh_key_path if ssh_key_path is None else ssh_key_path
    path = cfg.remote_config_path if remote_config_path is None else remote_config_path
    path = path or DEFAULT_REMOTE_CONFIG_PATH

    files = render_remote_files(config, remote_config_path=path)
    byte_count = sum(len(content.encode("utf-8")) for content in files.values())
    config_json = files.get(path, json.dumps(config, indent=2, sort_keys=True, default=str))

    missing = [
        name
        for name, value in (
            ("SSH_HOST", host),
            ("SSH_USER", user),
            ("SSH_KEY_PATH", key_path),
        )
        if not value
    ]
    if missing:
        return RemoteConfigPushResult(
            dry_run=True,
            pushed=False,
            remote_path=path,
            byte_count=byte_count,
            message=f"Remote config dry-run: missing {', '.join(missing)}",
            config_json=config_json,
            files=files,
            remote_paths=_path_labels(files),
        )

    client = None
    sftp = None
    try:
        client = _make_ssh_client(ssh_client_factory)
        client.connect(
            hostname=host,
            username=user,
            port=port,
            key_filename=key_path,
            timeout=10,
        )
        sftp = client.open_sftp()
        resolved_files: dict[str, str] = {}
        for remote_path, content in files.items():
            resolved_path = _resolve_remote_path(sftp, remote_path)
            resolved_files[remote_path] = resolved_path
            _ensure_remote_parent(sftp, resolved_path)
            with sftp.file(resolved_path, "w") as remote_file:
                remote_file.write(content)
        return RemoteConfigPushResult(
            dry_run=False,
            pushed=True,
            remote_path=resolved_files.get(path, path),
            byte_count=byte_count,
            message=f"Remote VM files pushed to {resolved_files.get(path, path)}",
            config_json=config_json,
            files=files,
            remote_paths=_path_labels(resolved_files),
        )
    except Exception as exc:  # pragma: no cover - exercised with fake clients in tests
        raise RemoteConfigPushError(f"Failed to push remote config to {path}: {exc}") from exc
    finally:
        if sftp is not None:
            sftp.close()
        if client is not None:
            client.close()


def render_remote_files(
    config: dict[str, Any],
    *,
    remote_config_path: str | None = None,
) -> dict[str, str]:
    """Render VM file paths and contents from a training payload."""
    config_path = (
        remote_config_path
        or config.get("paths", {}).get("config")
        or DEFAULT_REMOTE_CONFIG_PATH
    )
    if {"vm_config", "training_data", "eval_prompts"}.issubset(config):
        training_path = config.get("paths", {}).get(
            "training_data", DEFAULT_REMOTE_TRAINING_DATA_PATH
        )
        evals_path = config.get("paths", {}).get("evals", DEFAULT_REMOTE_EVALS_PATH)
        training_jsonl = "\n".join(
            json.dumps(row, sort_keys=True) for row in config["training_data"]
        )
        if training_jsonl:
            training_jsonl += "\n"
        return {
            config_path: json.dumps(config["vm_config"], indent=2, sort_keys=True),
            training_path: training_jsonl,
            evals_path: json.dumps(config["eval_prompts"], indent=2, sort_keys=True),
        }
    return {config_path: json.dumps(config, indent=2, sort_keys=True, default=str)}


def _make_ssh_client(ssh_client_factory: Callable[[], Any] | None) -> Any:
    if ssh_client_factory is not None:
        return ssh_client_factory()

    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return client


def _resolve_remote_path(sftp: Any, remote_path: str) -> str:
    if remote_path == "~":
        return str(sftp.normalize("."))
    if remote_path.startswith("~/"):
        home = str(sftp.normalize("."))
        return posixpath.join(home, remote_path[2:])
    return remote_path


def _ensure_remote_parent(sftp: Any, remote_path: str) -> None:
    parent = posixpath.dirname(remote_path)
    if not parent or parent in {".", "/"}:
        return

    current = "/" if parent.startswith("/") else ""
    for part in [p for p in parent.split("/") if p]:
        current = posixpath.join(current, part) if current else part
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def _path_labels(paths: dict[str, str]) -> dict[str, str]:
    labels = {
        DEFAULT_REMOTE_CONFIG_PATH: "config",
        DEFAULT_REMOTE_TRAINING_DATA_PATH: "training_data",
        DEFAULT_REMOTE_EVALS_PATH: "evals",
    }
    return {labels.get(path, path): value for path, value in paths.items()}
