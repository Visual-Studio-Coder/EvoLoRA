"""Per-run detailed logging to a gitignored directory for offline diagnosis.

Every event the orchestrator emits is appended to two files under ``logs/``:

* ``run-<ts>-<id>.jsonl`` — one JSON object per event (full ``data`` payload),
  machine-readable for tooling/agents.
* ``run-<ts>-<id>.log`` — a human-readable line per event.

The directory is gitignored. Logging is best-effort: any failure is swallowed so
it can never break a run. It is auto-disabled under pytest (so the suite doesn't
litter ``logs/``) unless an explicit ``log_dir`` is passed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from evolora.models.events import Event


def _is_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").exists() or (path / ".git").exists()


def _default_log_dir() -> Path:
    override = os.getenv("EVOLORA_LOG_DIR")
    if override:
        return Path(override)

    cwd = Path.cwd()
    for candidate in (cwd, cwd / "evolora"):
        if _is_project_root(candidate):
            return candidate / "logs"

    for parent in Path(__file__).resolve().parents:
        if _is_project_root(parent):
            return parent / "logs"

    return cwd / "logs"


def default_log_dir() -> Path:
    """Return the directory where EvoLoRA writes local run logs."""
    return _default_log_dir()


def _auto_enabled() -> bool:
    """Default on, but off inside the test suite and when explicitly disabled."""
    if os.getenv("EVOLORA_RUN_LOG", "").strip().lower() in {"0", "false", "off", "no"}:
        return False
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    return True


class RunLogger:
    """Append orchestrator events to a per-run JSONL + readable log under ``logs/``."""

    def __init__(
        self,
        run_id: str,
        *,
        log_dir: str | Path | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._run_id = run_id
        # An explicit log_dir means a caller (e.g. a test) wants logging on.
        if enabled is None:
            enabled = True if log_dir is not None else _auto_enabled()
        self._enabled = enabled
        self._dir = Path(log_dir) if log_dir is not None else _default_log_dir()
        self._jsonl: Path | None = None
        self._txt: Path | None = None
        self._initialized = False

    def _ensure_files(self, first_event: Event) -> None:
        if self._initialized:
            return
        self._initialized = True
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            stamp = first_event.ts.strftime("%Y%m%d-%H%M%S")
            base = f"run-{stamp}-{self._run_id[:8]}"
            self._jsonl = self._dir / f"{base}.jsonl"
            self._txt = self._dir / f"{base}.log"
        except Exception:
            self._jsonl = None
            self._txt = None

    def log_event(self, event: Event) -> None:
        if not self._enabled:
            return
        self._ensure_files(event)
        if self._jsonl is None:
            return
        kind = getattr(event.kind, "value", str(event.kind))
        try:
            row = {
                "ts": event.ts.isoformat(),
                "kind": kind,
                "iteration": event.iteration,
                "message": event.message,
                "data": event.data,
            }
            with self._jsonl.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            it = f"i{event.iteration}" if event.iteration is not None else "--"
            with self._txt.open("a", encoding="utf-8") as fh:
                fh.write(f"{event.ts.isoformat()}  {kind:<24} {it:>4}  {event.message}\n")
        except Exception:
            # Logging must never break a run.
            pass

    @property
    def path(self) -> Path | None:
        """The JSONL log path (None until the first event is logged)."""
        return self._jsonl

    @property
    def readable_path(self) -> Path | None:
        """The human-readable log path (None until the first event is logged)."""
        return self._txt
