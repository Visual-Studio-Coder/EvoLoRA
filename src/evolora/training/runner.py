"""Model runner protocol — runs the model on eval prompts and returns its outputs."""

from __future__ import annotations

import os
import random
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelRunner(Protocol):
    is_mock: bool

    async def run_batch(
        self, prompts: list[dict], adapter_path: str | None = None
    ) -> dict[str, str]:
        """Return {sample_id: raw_response} for each prompt dict."""
        ...


class MockModelRunner:
    """Returns deterministic fake responses for testing without a real model."""

    is_mock = True

    _MOCK_CUSTOMERS = ["Alice", "Bob", "Carol", "Dave", "Eve"]

    async def run_batch(
        self, prompts: list[dict], adapter_path: str | None = None
    ) -> dict[str, str]:
        import json

        results: dict[str, str] = {}
        for p in prompts:
            sid = p["sample_id"]
            customer = random.choice(self._MOCK_CUSTOMERS)
            total = round(random.uniform(1000, 5000), 2)
            results[sid] = json.dumps({
                "top_customer": customer,
                "top_customer_total": total,
                "customer_count": random.randint(3, 10),
                "total_revenue": round(total * random.uniform(1.5, 4.0), 2),
                "summary": f"Mock summary for adapter={adapter_path or 'base'}.",
            })
        return results


class RemoteModelRunner:
    """Reads eval outputs the GPU VM produced by running the trained adapter.

    Contract (implemented on the VM side): after the VM finishes training and
    evaluating an iteration, it writes the results file as a JSON object
    ``{"<sample_id>": "<raw model output>"}`` (or ``{"responses": {...}}``) at
    ``REMOTE_RESULTS_PATH``. This runner SFTP-reads that file and returns the
    responses for the requested prompts; the evaluator then scores them against
    the expected answers EvoLoRA kept locally — so only prompts ever leave the
    box and scoring stays honest.

    When SSH is not configured (no host/key) it falls back to MockModelRunner so
    the mock loop still runs.
    """

    is_mock = False

    def __init__(
        self,
        *,
        host: str = "",
        user: str = "",
        port: int = 22,
        key_path: str = "",
        results_path: str = "~/evolora/results.json",
        poll_timeout: float = 600.0,
        poll_interval: float = 5.0,
    ) -> None:
        self._host = host
        self._user = user
        self._port = port
        self._key_path = key_path
        self._results_path = results_path
        self._poll_timeout = poll_timeout
        self._poll_interval = poll_interval

    @property
    def configured(self) -> bool:
        return bool(self._host and self._key_path)

    async def run_batch(
        self, prompts: list[dict], adapter_path: str | None = None
    ) -> dict[str, str]:
        if not self.configured:
            # No VM wired yet -> degrade to mock so the loop still runs end to end.
            return await MockModelRunner().run_batch(prompts, adapter_path)
        results = await self._read_results()
        return {p["sample_id"]: str(results.get(p["sample_id"], "")) for p in prompts}

    async def _read_results(self) -> dict:
        """Poll the VM for the results file until it appears (or timeout)."""
        import asyncio

        attempts = max(1, int(self._poll_timeout / max(self._poll_interval, 0.1)))
        for i in range(attempts):
            try:
                data = await asyncio.to_thread(self._sftp_read_json)
                if data:
                    return data
            except Exception:
                pass  # transient SSH/SFTP error or file not ready yet — retry
            if i < attempts - 1:
                await asyncio.sleep(self._poll_interval)
        return {}

    def _sftp_read_json(self) -> dict:
        import json

        import paramiko  # lazy import: only needed when a VM is actually configured

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self._host,
            port=self._port,
            username=self._user or None,
            key_filename=self._key_path or None,
            timeout=15,
        )
        try:
            sftp = client.open_sftp()
            with sftp.open(self._results_path, "r") as fh:
                raw = fh.read()
            sftp.close()
        finally:
            client.close()
        text = raw.decode() if isinstance(raw, bytes) else raw
        return self._extract_responses(json.loads(text))

    @staticmethod
    def _extract_responses(data) -> dict:
        """Accept either {sample_id: response} or {"responses": {sample_id: response}}."""
        if isinstance(data, dict) and isinstance(data.get("responses"), dict):
            return data["responses"]
        return data if isinstance(data, dict) else {}


def get_runner(name: str) -> ModelRunner:
    if name == "mock":
        return MockModelRunner()
    if name in {"vm", "remote"}:
        return RemoteModelRunner(
            host=os.getenv("SSH_HOST", ""),
            user=os.getenv("SSH_USER", ""),
            port=int(os.getenv("SSH_PORT", "22")),
            key_path=os.getenv("SSH_KEY_PATH", ""),
            results_path=os.getenv("REMOTE_RESULTS_PATH", "~/evolora/results.json"),
        )
    raise ValueError(f"Unknown model runner: {name!r}")
