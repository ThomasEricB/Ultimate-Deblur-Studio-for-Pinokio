from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = "1.0"

TOKEN_HEADER = "X-Worker-Token"
TOKEN_ENV = "DS_WORKER_TOKEN"

JobState = Literal["queued", "running", "done", "failed", "cancelled"]
TERMINAL_STATES = {"done", "failed", "cancelled"}


def protocol_compatible(version: str) -> bool:
    """Major versions must match; minor versions are additive."""
    try:
        return version.split(".")[0] == PROTOCOL_VERSION.split(".")[0]
    except Exception:
        return False


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Health(_Strict):
    status: Literal["ok", "loading", "busy"]
    protocol_version: str
    backend: str
    pid: int
    device: str = ""
    loaded_model: str | None = None


class Metadata(_Strict):
    backend: str
    protocol_version: str
    adapter: str
    capabilities: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)


class LoadRequest(_Strict):
    model_id: str
    # Weight files, relative to the worker's models directory. The worker
    # resolves and anchors them; absolute paths are rejected.
    files: list[str]
    scale: int = 4
    options: dict[str, Any] = Field(default_factory=dict)


class LoadResponse(_Strict):
    model_id: str
    seconds: float
    info: dict[str, Any] = Field(default_factory=dict)


class JobRequest(_Strict):
    job_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # Both paths are relative to the shared job spool root.
    input: str
    output: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class JobStatus(_Strict):
    job_id: str
    state: JobState
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
