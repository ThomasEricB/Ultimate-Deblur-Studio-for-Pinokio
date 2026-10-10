"""Model weight download / delete / import.

Weights are fetched by the master (plain HTTPS, no ML imports) into
models/<backend>/, verified against the catalog's size and sha256 when given,
written to a .part file and atomically renamed. Weights are never stored in a
venv, so rebuilding an environment keeps them.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
from pathlib import Path
from typing import Callable

import httpx

from . import registry
from .paths import STATE_DIR
from .registry import Profile

HASHES_FILE = STATE_DIR / "model_hashes.json"   # trust-on-first-use record when no sha256 is published
_lock = threading.Lock()
ProgressFn = Callable[[float, str], None]


class DownloadCancelled(Exception):
    pass


def _record_hash(path: Path, digest: str) -> None:
    with _lock:
        try:
            data = json.loads(HASHES_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        data[str(path.name)] = digest
        HASHES_FILE.write_text(json.dumps(data, indent=2))


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download(p: Profile, progress: ProgressFn, cancelled: Callable[[], bool] = lambda: False) -> None:
    if p.custom:
        raise ValueError("custom models are imported, not downloaded")
    total = p.size_bytes or 0
    done = 0
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=15.0),
                      headers={"User-Agent": "deblur-studio/1.0"}) as client:
        for f, target in zip(p.entry.files, p.file_paths()):
            if target.is_file() and (not f.size or target.stat().st_size == f.size):
                done += f.size or target.stat().st_size
                continue
            part = target.with_name(target.name + ".part")
            digest = hashlib.sha256()
            with client.stream("GET", f.url) as r:
                r.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        if cancelled():
                            fh.close()
                            part.unlink(missing_ok=True)
                            raise DownloadCancelled()
                        fh.write(chunk)
                        digest.update(chunk)
                        done += len(chunk)
                        frac = done / total if total else 0.0
                        progress(min(frac, 1.0), f"{p.label}: {human(done)} / {human(total) if total else '?'}")
            size = part.stat().st_size
            if f.size and size != f.size:
                part.unlink(missing_ok=True)
                raise RuntimeError(f"{f.filename}: expected {f.size} bytes, got {size}")
            hexdigest = digest.hexdigest()
            if f.sha256 and hexdigest.lower() != f.sha256.lower():
                part.unlink(missing_ok=True)
                raise RuntimeError(f"{f.filename}: sha256 mismatch (got {hexdigest[:12]}…)")
            part.replace(target)
            _record_hash(target, hexdigest)


def delete(p: Profile) -> None:
    for path in p.file_paths():
        anchored = path.resolve()
        if p.backend.models_dir.resolve() not in anchored.parents:
            raise RuntimeError(f"refusing to delete {anchored}")
        anchored.unlink(missing_ok=True)
        anchored.with_name(anchored.name + ".part").unlink(missing_ok=True)
    if p.custom:
        registry.remove_custom(p.key)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower()[:60] or "custom"


def import_files(paths: list[str]) -> list[str]:
    """Import uploaded weight files. .pth/.safetensors/.ckpt/.pt -> spandrel;
    .param + .bin pairs -> ncnn-vulkan. Returns human-readable results."""
    results: list[str] = []
    by_ext: dict[str, list[Path]] = {}
    for raw in paths or []:
        src = Path(raw)
        by_ext.setdefault(src.suffix.lower(), []).append(src)

    def backend_for(ext: str) -> registry.Backend | None:
        for b in registry.backends():
            if ext in b.manifest.custom_models.extensions:
                return b
        return None

    # NCNN pairs
    params = {p.stem: p for p in by_ext.pop(".param", [])}
    bins = {p.stem: p for p in by_ext.pop(".bin", [])}
    for stem in sorted(set(params) | set(bins)):
        if stem not in params or stem not in bins:
            results.append(f"⚠️ {stem}: NCNN models need both {stem}.param and {stem}.bin")
            continue
        b = backend_for(".param")
        if b is None:
            results.append("⚠️ no backend accepts NCNN models")
            continue
        model_id = _slug(stem)
        names = []
        for src in (params[stem], bins[stem]):
            name = f"{model_id}{src.suffix.lower()}"
            shutil.copyfile(src, b.models_dir / name)
            names.append(name)
        registry.add_custom(b.id, model_id, stem, names, registry.guess_scale(stem))
        results.append(f"✅ {stem} → {b.name}")

    for ext, files in by_ext.items():
        b = backend_for(ext)
        for src in files:
            if b is None:
                results.append(f"⚠️ {src.name}: unsupported file type")
                continue
            model_id = _slug(src.stem)
            name = f"custom-{model_id}{ext}"
            shutil.copyfile(src, b.models_dir / name)
            registry.add_custom(b.id, model_id, src.stem, [name], registry.guess_scale(src.stem))
            results.append(f"✅ {src.name} → {b.name} (scale is detected from the weights at load time)")
    return results
