"""Pluggable training backend protocol and implementations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import shlex
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from evolora.config import get_config
from evolora.models.core import AgentPlan, ArtifactMeta
from evolora.training.remote_config import push_config


@runtime_checkable
class TrainingBackend(Protocol):
    is_mock: bool
    name: str

    async def train(
        self,
        run_id: str,
        iteration: int,
        plan: AgentPlan,
        base_model_id: str,
        remote_payload: dict | None = None,
    ) -> AsyncIterator[dict]:
        """Yield progress dicts; final dict contains 'artifact' key."""
        ...

    async def health_check(self) -> bool: ...


class MockTrainingBackend:
    """Deterministic fake backend for demos and tests — no GPU required."""

    is_mock = True
    name = "mock"

    async def train(
        self,
        run_id: str,
        iteration: int,
        plan: AgentPlan,
        base_model_id: str,
        remote_payload: dict | None = None,
    ) -> AsyncIterator[dict]:
        return self._stream(run_id, iteration, plan, base_model_id)

    async def _stream(
        self,
        run_id: str,
        iteration: int,
        plan: AgentPlan,
        base_model_id: str,
    ) -> AsyncIterator[dict]:
        hp = plan.hyperparams
        steps = hp.num_epochs * 10
        for step in range(1, steps + 1):
            loss = max(0.05, 2.0 * (0.85 ** step) + random.uniform(-0.02, 0.02))
            yield {"step": step, "total_steps": steps, "loss": round(loss, 4), "done": False}
            await asyncio.sleep(0.05)

        import hashlib
        import json

        artifact_id = f"mock-adapter-run{iteration}-{run_id[:8]}"
        checksum = hashlib.sha256(
            json.dumps({"run_id": run_id, "iteration": iteration}).encode()
        ).hexdigest()

        yield {
            "done": True,
            "artifact": ArtifactMeta(
                run_id=run_id,
                iteration=iteration,
                adapter_path=f"./artifacts/{artifact_id}",
                score=0.0,  # filled by orchestrator after eval
                checksum=checksum,
                is_mock=True,
            ),
            "cost_usd": 0.0,
            "duration_s": steps * 0.05,
        }

    async def health_check(self) -> bool:
        return True


class RemoteTrainingBackend:
    """Run the VM training scripts over SSH/SFTP."""

    is_mock = False
    name = "remote"

    def __init__(
        self,
        base_url: str = "",
        token: str = "",
        *,
        ssh_host: str | None = None,
        ssh_user: str | None = None,
        ssh_port: int | None = None,
        ssh_key_path: str | None = None,
        remote_config_path: str | None = None,
        remote_workspace: str = "/workspace",
        ssh_client_factory=None,
        evaluate_script_path: str | Path | None = None,
        train_script_path: str | Path | None = None,
        baseline_script_path: str | Path | None = None,
        chat_script_path: str | Path | None = None,
    ) -> None:
        cfg = get_config()
        self._base_url = base_url
        self._token = token
        self._ssh_host = cfg.ssh_host if ssh_host is None else ssh_host
        self._ssh_user = cfg.ssh_user if ssh_user is None else ssh_user
        self._ssh_port = cfg.ssh_port if ssh_port is None else ssh_port
        self._ssh_key_path = cfg.ssh_key_path if ssh_key_path is None else ssh_key_path
        self._remote_config_path = (
            cfg.remote_config_path if remote_config_path is None else remote_config_path
        )
        self._remote_workspace = remote_workspace.rstrip("/")
        self._ssh_client_factory = ssh_client_factory
        self._evaluate_script_path = Path(evaluate_script_path) if evaluate_script_path else _default_evaluate_script_path()
        self._train_script_path = Path(train_script_path) if train_script_path else _default_train_script_path()
        self._baseline_script_path = Path(baseline_script_path) if baseline_script_path else _default_baseline_script_path()
        self._chat_script_path = Path(chat_script_path) if chat_script_path else _default_chat_script_path()

    async def train(
        self,
        run_id: str,
        iteration: int,
        plan: AgentPlan,
        base_model_id: str,
        remote_payload: dict | None = None,
    ) -> AsyncIterator[dict]:
        return self._stream(run_id, iteration, remote_payload)

    async def evaluate_base(self, remote_payload: dict | None = None) -> AsyncIterator[dict]:
        """Run the base model on the locked evals on the VM and pull filled records."""
        return self._baseline_stream(remote_payload)

    async def _baseline_stream(self, remote_payload: dict | None) -> AsyncIterator[dict]:
        self._assert_configured(remote_payload)

        yield {"phase": "baseline", "message": "Uploading VM baseline config/evals", "done": False}
        push_result = await asyncio.to_thread(
            push_config,
            remote_payload,
            ssh_host=self._ssh_host,
            ssh_user=self._ssh_user,
            ssh_port=self._ssh_port,
            ssh_key_path=self._ssh_key_path,
            remote_config_path=self._remote_config_path,
            ssh_client_factory=self._ssh_client_factory,
        )
        yield {
            "phase": "baseline",
            "message": push_result.message,
            "byte_count": push_result.byte_count,
            "done": False,
        }

        yield {"phase": "baseline", "message": "Uploading VM baseline_evaluate.py", "done": False}
        await asyncio.to_thread(
            self._push_script,
            self._baseline_script_path,
            "baseline_evaluate.py",
        )

        baseline_command = f"cd {self._remote_workspace} && python -u baseline_evaluate.py"
        async for line in self._exec_remote_command(baseline_command):
            yield {"phase": "baseline", "message": line, "done": False}

        evals_remote = f"{self._remote_workspace}/data/evals.json"
        eval_records = await asyncio.to_thread(self._pull_json, evals_remote)
        yield {
            "phase": "baseline",
            "message": f"Pulled {len(eval_records)} baseline eval records",
            "eval_records": eval_records,
            "done": True,
        }

    async def _stream(
        self,
        run_id: str,
        iteration: int,
        remote_payload: dict | None,
    ) -> AsyncIterator[dict]:
        self._assert_configured(remote_payload)

        yield {"phase": "upload", "message": "Uploading VM config/data/evals", "done": False}
        push_result = await asyncio.to_thread(
            push_config,
            remote_payload,
            ssh_host=self._ssh_host,
            ssh_user=self._ssh_user,
            ssh_port=self._ssh_port,
            ssh_key_path=self._ssh_key_path,
            remote_config_path=self._remote_config_path,
            ssh_client_factory=self._ssh_client_factory,
        )
        yield {
            "phase": "upload",
            "message": push_result.message,
            "byte_count": push_result.byte_count,
            "done": False,
        }

        yield {"phase": "upload", "message": "Uploading VM train.py + evaluate.py", "done": False}
        await asyncio.to_thread(self._push_script, self._train_script_path, "train.py")
        await asyncio.to_thread(self._push_script, self._evaluate_script_path, "evaluate.py")

        train_command = f"cd {self._remote_workspace} && python -u train.py"
        async for line in self._exec_remote_command(train_command):
            yield {"phase": "train", "message": line, "done": False}

        eval_command = f"cd {self._remote_workspace} && python -u evaluate.py"
        async for line in self._exec_remote_command(eval_command):
            yield {"phase": "evaluate", "message": line, "done": False}

        # The VM's evaluate.py fills "actual" in place in data/evals.json; pull it back so
        # the orchestrator can score it with the LLM-judge.
        evals_remote = f"{self._remote_workspace}/data/evals.json"
        eval_records = await asyncio.to_thread(self._pull_json, evals_remote)
        yield {
            "phase": "evaluate",
            "message": f"Pulled {len(eval_records)} eval records",
            "done": False,
        }

        # Archive this run's adapter so it persists and is selectable for chat.
        try:
            archived = await self._archive_adapter(run_id, remote_payload)
            yield {"phase": "archive", "message": f"Saved model as {archived}", "done": False}
        except Exception as exc:  # best-effort; never fail a run on archiving
            yield {"phase": "archive", "message": f"Adapter archive skipped: {exc}", "done": False}

        adapter_path = f"{self._remote_workspace}/lora_model"
        checksum = hashlib.sha256(
            json.dumps(
                {"run_id": run_id, "iteration": iteration, "adapter_path": adapter_path},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        yield {
            "done": True,
            "artifact": ArtifactMeta(
                run_id=run_id,
                iteration=iteration,
                adapter_path=adapter_path,
                score=0.0,
                checksum=checksum,
                is_mock=False,
            ),
            "eval_records": eval_records,
            "cost_usd": 0.0,
            "duration_s": 0.0,
        }

    async def health_check(self) -> bool:
        return bool(self._ssh_host and self._ssh_user and self._ssh_key_path)

    def _assert_ssh_configured(self) -> None:
        missing = [
            name
            for name, value in (
                ("SSH_HOST", self._ssh_host),
                ("SSH_USER", self._ssh_user),
                ("SSH_KEY_PATH", self._ssh_key_path),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "RemoteTrainingBackend requires SSH configuration; missing "
                + ", ".join(missing)
            )

    def _assert_configured(self, remote_payload: dict | None) -> None:
        self._assert_ssh_configured()
        if remote_payload is None:
            raise RuntimeError("RemoteTrainingBackend requires a VM config payload")

    async def chat(self, prompt: str, model_dir: str = "lora_model") -> str:
        """One-shot inference against a chosen model on the VM.

        ``model_dir`` is a trained adapter dir (e.g. ``lora_model`` or
        ``adapters/<name>``) or a base model name. Pushes chat.py, runs it, and
        returns just the model's response (extracted via the <<<EVOLORA_RESPONSE>>>
        marker). Loads the model per call, so the first reply is slow. Raises if SSH
        is unconfigured or the chosen model is missing.
        """
        self._assert_ssh_configured()
        await asyncio.to_thread(self._push_script, self._chat_script_path, "chat.py")
        command = (
            f"cd {self._remote_workspace} && "
            f"python chat.py {shlex.quote(prompt)} {shlex.quote(model_dir)}"
        )
        lines: list[str] = []
        async for line in self._exec_remote_command(command):
            lines.append(line)
        text = "\n".join(lines)
        marker = "<<<EVOLORA_RESPONSE>>>"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return lines[-1].strip() if lines else ""

    async def list_adapters(self) -> list[str]:
        """List selectable trained models on the VM: archived adapters/<name> dirs
        plus the latest lora_model (if present). Returns model_dir paths for chat()."""
        self._assert_ssh_configured()
        command = (
            f"cd {self._remote_workspace}; "
            f"ls -1d adapters/*/ 2>/dev/null | sed 's:/*$::'; "
            f"[ -d lora_model ] && echo lora_model; true"
        )
        names: list[str] = []
        async for line in self._exec_remote_command(command):
            name = line.strip()
            if name and name not in names:
                names.append(name)
        return names

    async def _archive_adapter(self, run_id: str, remote_payload: dict | None) -> str:
        """Copy the just-trained lora_model to adapters/<slug> so it persists and
        becomes selectable for chat. Best-effort; returns the archive label."""
        goal = str((remote_payload or {}).get("goal") or "model")
        label = f"{_slugify(goal)[:32] or 'model'}-{run_id[:6]}"
        target = f"adapters/{label}"
        command = (
            f"cd {self._remote_workspace} && mkdir -p adapters && "
            f"rm -rf {target} && cp -r lora_model {target}"
        )
        async for _ in self._exec_remote_command(command):
            pass
        return target

    def _make_client(self):
        if self._ssh_client_factory is not None:
            return self._ssh_client_factory()

        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return client

    def _connect_client(self):
        client = self._make_client()
        client.connect(
            hostname=self._ssh_host,
            username=self._ssh_user,
            port=self._ssh_port,
            key_filename=self._ssh_key_path,
            timeout=15,
        )
        return client

    def _push_script(self, local_path: Path, remote_name: str) -> None:
        if not local_path.exists():
            raise RuntimeError(f"Missing VM script: {local_path}")

        content = local_path.read_text(encoding="utf-8")
        client = self._connect_client()
        sftp = None
        try:
            sftp = client.open_sftp()
            remote_path = f"{self._remote_workspace}/{remote_name}"
            with sftp.file(remote_path, "w") as remote_file:
                remote_file.write(content)
        finally:
            if sftp is not None:
                sftp.close()
            client.close()

    def _pull_json(self, remote_path: str):
        client = self._connect_client()
        sftp = None
        try:
            sftp = client.open_sftp()
            with sftp.file(remote_path, "r") as remote_file:
                raw = remote_file.read()
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            return json.loads(text)
        finally:
            if sftp is not None:
                sftp.close()
            client.close()

    async def _exec_remote_command(self, command: str) -> AsyncIterator[str]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        worker = asyncio.create_task(
            asyncio.to_thread(self._exec_streaming_blocking, command, loop, queue)
        )
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        exit_status, stderr = await worker
        if exit_status != 0:
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(f"Remote command failed ({exit_status}) {command}{detail}")

    def _exec_streaming_blocking(
        self,
        command: str,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[str | None],
    ) -> tuple[int, str]:
        client = self._connect_client()
        try:
            _, stdout, stderr = client.exec_command(command, get_pty=True)
            while True:
                raw = stdout.readline()
                if not raw:
                    break
                text = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
                text = text.rstrip()
                if text:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
            exit_status = stdout.channel.recv_exit_status()
            err = stderr.read()
            if isinstance(err, bytes):
                err_text = err.decode(errors="replace")
            else:
                err_text = str(err or "")
            return exit_status, err_text.strip()
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            client.close()


def _slugify(text: str) -> str:
    """Filesystem-safe slug for adapter archive names."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "model"


def _try_unsloth_backend():
    """Return UnslothTrainingBackend if deps are available, else raise clearly."""
    try:
        import torch  # noqa: F401
        import unsloth  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"UnslothTrainingBackend requires CUDA + torch + unsloth. Missing: {exc}. "
            "Install with: pip install evolora[unsloth]"
        ) from exc

    class UnslothTrainingBackend:
        is_mock = False
        name = "unsloth"

        async def train(self, run_id, iteration, plan, base_model_id, remote_payload=None):
            raise NotImplementedError("Unsloth backend skeleton — not yet implemented.")

        async def health_check(self) -> bool:
            return False

    return UnslothTrainingBackend()


def get_backend(name: str, **kwargs) -> TrainingBackend:
    if name == "mock":
        return MockTrainingBackend()
    if name == "remote":
        return RemoteTrainingBackend(
            base_url=kwargs.get("base_url", ""),
            token=kwargs.get("token", ""),
        )
    if name == "unsloth":
        return _try_unsloth_backend()
    raise ValueError(f"Unknown training backend: {name!r}")


def _default_evaluate_script_path() -> Path:
    # Use the VM guy's committed evaluate.py (reads data/evals.json as [{input, expected}]
    # and fills "actual" in place) so the VM-side eval matches the agreed format.
    return Path(__file__).resolve().parents[3] / "src" / "virtual_machine_code" / "evaluate.py"


def _default_train_script_path() -> Path:
    # The VM guy's committed train.py (reads config.json + data/training_data.jsonl, saves the
    # adapter to lora_model/). Pushed so a run is self-contained even if /workspace was wiped.
    return Path(__file__).resolve().parents[3] / "src" / "virtual_machine_code" / "train.py"


def _default_baseline_script_path() -> Path:
    # Separate baseline evaluator: same eval format as VM evaluate.py, but loads the base model.
    return Path(__file__).resolve().parents[3] / "scripts" / "vm" / "baseline_evaluate.py"


def _default_chat_script_path() -> Path:
    # Single-prompt inference against the trained adapter (lora_model).
    return Path(__file__).resolve().parents[3] / "src" / "virtual_machine_code" / "chat.py"
