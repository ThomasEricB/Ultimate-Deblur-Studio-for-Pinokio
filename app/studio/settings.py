"""User settings - the Gradio equivalent of Upscayl's jotai `atomWithStorage`
atoms (renderer/atoms/user-settings-atom.ts). Stored server-side because this
is a single-user local app and the worker supervisor needs them too."""

from __future__ import annotations

import json
import threading
from typing import Any

from .paths import DEFAULT_OUTPUT_DIR, STATE_DIR

SETTINGS_FILE = STATE_DIR / "settings.json"

DEFAULTS: dict[str, Any] = {
    # Upscayl atoms
    "selected_model": "ncnn-vulkan/upscayl-standard-4x",
    "double_upscayl": False,
    "gpu_id": "",
    "save_image_as": "png",
    "scale": 4,
    "saved_output_path": str(DEFAULT_OUTPUT_DIR),
    "remember_output_folder": False,
    "compression": 0,
    "overwrite": False,
    "turn_off_notifications": False,
    "tta_mode": False,
    "view_type": "slider",
    "zoom_amount": 100,
    "custom_width": 0,
    "use_custom_width": False,
    "tile_size": 0,
    "copy_metadata": False,
    "theme": "system",
    # Deblur Studio additions
    "restore_model": "",          # optional pre-pass (deblur/denoise) before the main model
    "precision": "auto",          # auto | fp16 | bf16 | fp32
    "idle_timeout_min": 15,       # stop the worker (free VRAM) after this many idle minutes
    "torch_variant": "auto",      # auto | cu130 | cu128 | rocm7.2 | mps | cpu
    "onboarded": False,
}

_lock = threading.Lock()


def load() -> dict[str, Any]:
    with _lock:
        data = dict(DEFAULTS)
        try:
            stored = json.loads(SETTINGS_FILE.read_text())
            data.update({k: v for k, v in stored.items() if k in DEFAULTS})
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return data


def save(values: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        try:
            current = json.loads(SETTINGS_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        current.update({k: v for k, v in values.items() if k in DEFAULTS})
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2))
        tmp.replace(SETTINGS_FILE)
    return load()


def set(key: str, value: Any) -> dict[str, Any]:  # noqa: A001 - mirrors atom setters
    return save({key: value})


def reset() -> dict[str, Any]:
    with _lock:
        SETTINGS_FILE.unlink(missing_ok=True)
    return load()
