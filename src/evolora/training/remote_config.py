"""Remote training config push over SSH/SFTP."""

from __future__ import annotations

import copy
import json
import posixpath
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from evolora.config import get_config
from evolora.evaluation.locked import LockedEvalSet
from evolora.models.core import AgentPlan, RunConfig

DEFAULT_REMOTE_CONFIG_PATH = "~/evolora/config.json"


class RemoteConfigPushError(RuntimeError):
    """Raised when a configured SSH push fails."""


class RemoteConfigPushResult(BaseModel):
    dry_run: bool
    pushed: bool
    remote_path: str
    byte_count: int
    message: str
    config_json: str = ""


def build_training_config_payload(
    *,
    run_id: str,
    iteration: int,
    run_config: RunConfig,
    plan: AgentPlan,
    eval_set: LockedEvalSet,
    remote_results_path: str | None = None,
) -> dict[str, Any]:
    """Build the JSON payload consumed by the remote GPU trainer."""
    cfg = get_config()
    examples = copy.deepcopy(plan.data_spec.examples)
    return {
        "schema_version": 1,
        "source": "evolora",
        "run_id": run_id,
        "iteration": iteration,
        "task_name": run_config.task_name,
        "goal": run_config.goal,
        "base_model_id": run_config.base_model_id,
        "training_backend": run_config.training_backend,
        "target_adapter_name": plan.target_adapter_name,
        "hyperparameters": plan.hyperparams.model_dump(mode="json"),
        "training_examples": examples,
        "training_example_count": len(examples),
        "training_data_rationale": plan.data_spec.rationale,
        "focus_areas": list(plan.focus_areas),
        "remote_results_path": remote_results_path or cfg.remote_results_path,
        "eval_prompts": eval_set.prompts_only(),
        "eval_set": {
            "hash": eval_set.hash,
            "prompt_count": len(eval_set),
        },
    }


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
    """Serialize config JSON and push it to a GPU VM.

    Missing SSH host/user/key means dry-run mode. Dry-run returns the JSON that
    would be pushed, but does not attempt network access and does not fail.
    """
    cfg = get_config()
    host = cfg.ssh_host if ssh_host is None else ssh_host
    user = cfg.ssh_user if ssh_user is None else ssh_user
    port = int(cfg.ssh_port if ssh_port is None else ssh_port)
    key_path = cfg.ssh_key_path if ssh_key_path is None else ssh_key_path
    path = cfg.remote_config_path if remote_config_path is None else remote_config_path
    path = path or DEFAULT_REMOTE_CONFIG_PATH

    config_json = json.dumps(config, indent=2, sort_keys=True, default=str)
    encoded = config_json.encode("utf-8")

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
            byte_count=len(encoded),
            message=f"Remote config dry-run: missing {', '.join(missing)}",
            config_json=config_json,
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
        resolved_path = _resolve_remote_path(sftp, path)
        _ensure_remote_parent(sftp, resolved_path)
        with sftp.file(resolved_path, "w") as remote_file:
            remote_file.write(config_json)
        return RemoteConfigPushResult(
            dry_run=False,
            pushed=True,
            remote_path=resolved_path,
            byte_count=len(encoded),
            message=f"Remote config pushed to {resolved_path}",
            config_json=config_json,
        )
    except Exception as exc:  # pragma: no cover - exercised with fake clients in tests
        raise RemoteConfigPushError(f"Failed to push remote config to {path}: {exc}") from exc
    finally:
        if sftp is not None:
            sftp.close()
        if client is not None:
            client.close()


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
