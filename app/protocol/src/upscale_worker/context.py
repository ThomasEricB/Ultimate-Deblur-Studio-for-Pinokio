from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PathEscapeError(ValueError):
    pass


def anchored(root: Path, relative: str) -> Path:
    """Resolve `relative` under `root`, refusing absolute paths, `..` escapes
    and symlinks that land outside the root."""
    if not relative or Path(relative).is_absolute() or Path(relative).drive:
        raise PathEscapeError(f"path must be relative: {relative!r}")
    root = root.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise PathEscapeError(f"path escapes its root: {relative!r}")
    return target


@dataclass
class WorkerContext:
    backend: str
    plugin_dir: Path
    models_dir: Path
    spool_dir: Path

    def model_path(self, relative: str) -> Path:
        return anchored(self.models_dir, relative)

    def spool_path(self, relative: str) -> Path:
        return anchored(self.spool_dir, relative)


class JobCancelled(Exception):
    """Raised by adapters when the cancellation token is set."""
