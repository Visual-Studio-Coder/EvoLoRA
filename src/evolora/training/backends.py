"""Pluggable training backend protocol and implementations."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from evolora.models.core import AgentPlan, ArtifactMeta, TrainingDataSpec


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

        import hashlib, json, os

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
    """Placeholder for a remote HTTP training backend (e.g. DigitalOcean GPU)."""

    is_mock = False
    name = "remote"

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url
        self._token = token

    async def train(
        self,
        run_id: str,
        iteration: int,
        plan: AgentPlan,
        base_model_id: str,
    ) -> AsyncIterator[dict]:
        raise NotImplementedError(
            "RemoteTrainingBackend is not yet verified. "
            "Set TRAINING_BACKEND=mock until a real GPU endpoint is confirmed on-site."
        )

    async def health_check(self) -> bool:
        return False


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

        async def train(self, run_id, iteration, plan, base_model_id):
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
