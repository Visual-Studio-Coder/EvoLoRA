"""Artifact store protocol and local filesystem implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from evolora.models.core import ArtifactMeta


@runtime_checkable
class ArtifactStore(Protocol):
    async def save(self, meta: ArtifactMeta, data: bytes) -> ArtifactMeta: ...
    async def load(self, artifact_id: str) -> bytes | None: ...


class LocalArtifactStore:
    """Saves artifacts to a local directory with checksum verification."""

    def __init__(self, base_dir: str = "./artifacts") -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    async def save(self, meta: ArtifactMeta, data: bytes) -> ArtifactMeta:
        path = self._base / meta.artifact_id
        path.write_bytes(data)
        checksum = hashlib.sha256(data).hexdigest()
        meta = meta.model_copy(update={"adapter_path": str(path), "checksum": checksum})
        # Write sidecar metadata
        (self._base / f"{meta.artifact_id}.json").write_text(
            json.dumps(meta.model_dump(mode="json"), indent=2)
        )
        return meta

    async def load(self, artifact_id: str) -> bytes | None:
        path = self._base / artifact_id
        if path.exists():
            return path.read_bytes()
        return None

    async def save_mock(self, meta: ArtifactMeta) -> ArtifactMeta:
        """Create a clearly-labeled mock artifact file."""
        content = json.dumps(
            {"mock": True, "run_id": meta.run_id, "iteration": meta.iteration}, indent=2
        ).encode()
        return await self.save(meta, content)
