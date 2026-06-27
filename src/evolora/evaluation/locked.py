"""Locked evaluation set with tamper detection."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from evolora.models.core import EvalSample


class LockedEvalSet:
    """Immutable eval set identified by a canonical hash.

    The hash is computed once at construction and checked before every use.
    Any mutation raises RuntimeError — the orchestrator must halt the run.
    """

    def __init__(self, samples: list[EvalSample]) -> None:
        self._samples = list(samples)
        self._hash = self._compute_hash(self._samples)

    @staticmethod
    def _compute_hash(samples: list[EvalSample]) -> str:
        canonical = json.dumps(
            [s.model_dump() for s in sorted(samples, key=lambda s: s.sample_id)],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def hash(self) -> str:
        return self._hash

    @property
    def samples(self) -> list[EvalSample]:
        self._assert_integrity()
        return list(self._samples)

    def _assert_integrity(self) -> None:
        current = self._compute_hash(self._samples)
        if current != self._hash:
            raise RuntimeError(
                f"Eval set integrity violation: expected {self._hash}, got {current}"
            )

    def prompts_only(self) -> list[dict[str, Any]]:
        """Return prompts without expected answers — safe to pass to the model."""
        self._assert_integrity()
        return [{"sample_id": s.sample_id, "prompt": s.prompt} for s in self._samples]

    def __len__(self) -> int:
        return len(self._samples)
