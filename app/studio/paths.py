"""Filesystem layout. Everything lives under app/ so the Pinokio launcher can
reset/clean it, but weights, venvs, jobs and state are kept apart: deleting a
venv never touches weights or history."""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

PLUGINS_DIR = APP_DIR / "plugins"          # trusted backend manifests + adapters (+ .venv each)
PROTOCOL_DIR = APP_DIR / "protocol"        # shared worker protocol package
ASSETS_DIR = APP_DIR / "assets"            # logo, css/js, model comparison images
MODELS_DIR = Path(os.environ.get("DS_MODELS_DIR", APP_DIR / "models")).resolve()  # downloaded weights
RUNTIME_DIR = APP_DIR / "runtime"
JOBS_DIR = RUNTIME_DIR / "jobs"            # shared job spool (master <-> worker)
LOGS_DIR = RUNTIME_DIR / "logs"
STATE_DIR = APP_DIR / "state"              # settings.json, registry.sqlite3, custom models
DEFAULT_OUTPUT_DIR = APP_DIR / "outputs"

for _d in (MODELS_DIR, JOBS_DIR, LOGS_DIR, STATE_DIR, DEFAULT_OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def backend_models_dir(backend_id: str) -> Path:
    path = MODELS_DIR / backend_id
    path.mkdir(parents=True, exist_ok=True)
    return path
