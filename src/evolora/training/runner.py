"""Model runner protocol — runs the model on eval prompts."""

from __future__ import annotations

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


def get_runner(name: str) -> ModelRunner:
    if name == "mock":
        return MockModelRunner()
    raise ValueError(f"Unknown model runner: {name!r}")
