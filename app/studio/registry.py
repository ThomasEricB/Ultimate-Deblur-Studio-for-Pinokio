"""Backend registry: strict manifest validation, model profiles and custom models.

Backends are discovered from plugins/*/manifest.toml only - never by scanning
arbitrary venvs - and nothing in a plugin folder is imported by the master.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from upscale_protocol import PROTOCOL_VERSION, protocol_compatible

from .paths import PLUGINS_DIR, STATE_DIR, backend_models_dir

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$", re.IGNORECASE)
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ ()+-]{0,200}$")
Task = Literal["upscale", "deblur", "denoise", "jpeg", "restore"]
TASK_LABELS = {
    "upscale": "Upscale",
    "deblur": "Deblur",
    "denoise": "Denoise",
    "jpeg": "JPEG cleanup",
    "restore": "Restore",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelFile(_Strict):
    url: str
    filename: str
    size: int | None = None
    sha256: str | None = None

    @field_validator("filename")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not SAFE_FILENAME.match(v) or "/" in v or "\\" in v or ".." in v:
            raise ValueError(f"unsafe filename {v!r}")
        return v

    @field_validator("url")
    @classmethod
    def _https(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("model URLs must be https")
        return v


class ModelEntry(_Strict):
    id: str
    name: str
    description: str = ""
    task: Task = "upscale"
    scale: int = Field(4, ge=1, le=16)
    group: str = ""
    license: str = ""
    commercial: bool | None = None
    recommended: bool = False
    preview: str | None = None
    files: list[ModelFile] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _safe_id(cls, v: str) -> str:
        if not SAFE_ID.match(v):
            raise ValueError(f"unsafe model id {v!r}")
        return v


class Resources(_Strict):
    gpu: bool = True
    estimated_vram_mb: int = 0


class Capabilities(_Strict):
    tta: bool = False
    tiling: bool = False
    gpu_id: bool = False
    precision: bool = False


class CustomModels(_Strict):
    extensions: list[str] = Field(default_factory=list)
    pair: bool = False


class Manifest(_Strict):
    schema_version: Literal[1]
    id: str
    display_name: str
    description: str = ""
    protocol_version: str
    adapter: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
    python: str = Field(pattern=r"^3\.\d{1,2}$")
    requirements: str = "requirements.txt"
    torch: bool = False
    resources: Resources = Field(default_factory=Resources)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    custom_models: CustomModels = Field(default_factory=CustomModels)
    models: list[ModelEntry] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _safe_id(cls, v: str) -> str:
        if not SAFE_ID.match(v):
            raise ValueError(f"unsafe backend id {v!r}")
        return v

    @field_validator("protocol_version")
    @classmethod
    def _compatible(cls, v: str) -> str:
        if not protocol_compatible(v):
            raise ValueError(f"protocol {v} is incompatible with master protocol {PROTOCOL_VERSION}")
        return v


@dataclass(frozen=True)
class Backend:
    manifest: Manifest
    dir: Path

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def name(self) -> str:
        return self.manifest.display_name

    @property
    def venv_dir(self) -> Path:
        return self.dir / ".venv"

    @property
    def requirements_file(self) -> Path:
        return self.dir / self.manifest.requirements

    @property
    def models_dir(self) -> Path:
        return backend_models_dir(self.id)


@dataclass(frozen=True)
class Profile:
    """A selectable model. Many profiles share one backend environment."""

    backend: Backend
    entry: ModelEntry
    custom: bool = False

    @property
    def key(self) -> str:
        return f"{self.backend.id}/{self.entry.id}"

    @property
    def label(self) -> str:
        return self.entry.name

    def file_paths(self) -> list[Path]:
        return [self.backend.models_dir / f.filename for f in self.entry.files]

    def relative_files(self) -> list[str]:
        return [f.filename for f in self.entry.files]

    @property
    def installed(self) -> bool:
        for f, path in zip(self.entry.files, self.file_paths()):
            if not path.is_file():
                return False
            if f.size and path.stat().st_size != f.size:
                return False
        return True

    @property
    def size_bytes(self) -> int:
        return sum(f.size or 0 for f in self.entry.files)


# ---------------------------------------------------------------------------
_lock = threading.Lock()
CUSTOM_FILE = STATE_DIR / "custom_models.json"


@lru_cache(maxsize=1)
def _load_backends() -> tuple[tuple[Backend, ...], tuple[str, ...]]:
    backends: list[Backend] = []
    errors: list[str] = []
    for manifest_path in sorted(PLUGINS_DIR.glob("*/manifest.toml")):
        try:
            data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = Manifest.model_validate(data)
            if manifest.id != manifest_path.parent.name:
                raise ValueError(f"manifest id {manifest.id!r} must match its folder name")
            ids = [m.id for m in manifest.models]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate model ids")
            backends.append(Backend(manifest, manifest_path.parent.resolve()))
        except Exception as exc:  # a broken plugin must not take the app down
            errors.append(f"{manifest_path.parent.name}: {exc}")
    return tuple(backends), tuple(errors)


def backends() -> list[Backend]:
    return list(_load_backends()[0])


def manifest_errors() -> list[str]:
    return list(_load_backends()[1])


def backend(backend_id: str) -> Backend:
    for b in backends():
        if b.id == backend_id:
            return b
    raise KeyError(f"unknown backend {backend_id!r}")


def _custom_entries() -> list[dict]:
    try:
        return json.loads(CUSTOM_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def profiles() -> list[Profile]:
    out: list[Profile] = []
    by_id = {b.id: b for b in backends()}
    for b in by_id.values():
        out.extend(Profile(b, m) for m in b.manifest.models)
    for raw in _custom_entries():
        b = by_id.get(raw.get("backend", ""))
        if b is None:
            continue
        try:
            out.append(Profile(b, ModelEntry.model_validate(raw["entry"]), custom=True))
        except Exception:
            continue
    return out


def profile(key: str) -> Profile:
    for p in profiles():
        if p.key == key:
            return p
    raise KeyError(f"unknown model {key!r}")


def installed_profiles() -> list[Profile]:
    return [p for p in profiles() if p.installed]


def guess_scale(name: str) -> int:
    """Upscayl's rule (common/check-model-scale.ts), extended to 1x/8x."""
    lowered = name.lower()
    for s in (1, 2, 3, 8, 16):
        if re.search(rf"(?<![0-9])(x{s}|{s}x)(?![0-9])", lowered):
            return s
    return 4


def add_custom(backend_id: str, model_id: str, name: str, filenames: list[str], scale: int, task: str = "upscale") -> Profile:
    b = backend(backend_id)
    entry = ModelEntry(
        id=model_id,
        name=name,
        description="Imported custom model",
        task=task,
        scale=scale,
        group="Custom",
        files=[
            ModelFile(url="https://local.invalid/custom", filename=f, size=(b.models_dir / f).stat().st_size)
            for f in filenames
        ],
    )
    with _lock:
        items = [i for i in _custom_entries() if not (i.get("backend") == backend_id and i["entry"]["id"] == model_id)]
        if any(p.backend.id == backend_id and p.entry.id == model_id and not p.custom for p in profiles()):
            raise ValueError(f"{model_id} clashes with a catalog model")
        items.append({"backend": backend_id, "entry": entry.model_dump()})
        CUSTOM_FILE.write_text(json.dumps(items, indent=2))
    return Profile(b, entry, custom=True)


def remove_custom(key: str) -> None:
    backend_id, _, model_id = key.partition("/")
    with _lock:
        items = [i for i in _custom_entries() if not (i.get("backend") == backend_id and i["entry"]["id"] == model_id)]
        CUSTOM_FILE.write_text(json.dumps(items, indent=2))
