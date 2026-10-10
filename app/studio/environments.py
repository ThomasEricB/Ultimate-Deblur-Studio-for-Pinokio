"""Backend environment administration (create / verify / update / repair / remove).

Each backend gets `plugins/<id>/.venv`, built with uv from the backend's
pinned requirements.txt plus the shared protocol package. Venvs are treated as
disposable outputs: the manifest + requirements are authoritative, so Repair
recreates instead of patching. Weights live elsewhere and are never touched.

All commands are fixed argument vectors (shell=False) built from allowlisted
manifests - the UI cannot inject packages, URLs or paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from upscale_protocol import PROTOCOL_VERSION

from . import registry
from .paths import LOGS_DIR, PROTOCOL_DIR
from .registry import Backend

MARKER = "deblur-studio.json"
_installer_lock = threading.Lock()          # global: one install at a time
_busy: dict[str, str] = {}                  # backend id -> operation in progress

# PyTorch builds per accelerator. Kept in one table so updating is a one-line
# change; each variant pins torch + the matching torchvision.
TORCH_VARIANTS: dict[str, dict] = {
    "cu130": {"label": "NVIDIA CUDA 13.0 (driver >= 580)", "index": "https://download.pytorch.org/whl/cu130",
              "packages": ["torch==2.14.1", "torchvision==0.29.1"]},
    "cu128": {"label": "NVIDIA CUDA 12.8 (driver >= 570)", "index": "https://download.pytorch.org/whl/cu128",
              "packages": ["torch==2.11.0", "torchvision==0.26.0"]},
    "rocm7.2": {"label": "AMD ROCm 7.2 (Linux)", "index": "https://download.pytorch.org/whl/rocm7.2",
                "packages": ["torch==2.14.1", "torchvision==0.29.1"]},
    "mps": {"label": "Apple Silicon (MPS)", "index": None,
            "packages": ["torch==2.14.1", "torchvision==0.29.1"]},
    "cpu": {"label": "CPU only", "index": "https://download.pytorch.org/whl/cpu",
            "packages": ["torch==2.14.1", "torchvision==0.29.1"]},
}


def _nvidia_driver_major() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return int(out[0].split(".")[0]) if out else None
    except Exception:
        return None


def detect_torch_variant() -> str:
    system = platform.system()
    if system == "Darwin":
        return "mps" if platform.machine() == "arm64" else "cpu"
    major = _nvidia_driver_major()
    if major is not None:
        return "cu130" if major >= 580 else "cu128"
    if system == "Linux" and (shutil.which("rocminfo") or Path("/opt/rocm").exists()):
        return "rocm7.2"
    return "cpu"


def resolve_torch_variant(choice: str | None) -> str:
    return choice if choice in TORCH_VARIANTS else detect_torch_variant()


def find_uv() -> str:
    exe = shutil.which("uv")
    if exe:
        return exe
    try:
        from uv import find_uv_bin  # pip package fallback (see requirements.txt)

        return find_uv_bin()
    except Exception as exc:
        raise RuntimeError("uv was not found on PATH and the uv package is not installed") from exc


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _sanitized_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    env["UV_NO_PROGRESS"] = "1"
    return env


def requirements_hash(b: Backend, torch_variant: str | None) -> str:
    h = hashlib.sha256()
    h.update(b.requirements_file.read_bytes())
    h.update(f"python={b.manifest.python};protocol={PROTOCOL_VERSION}".encode())
    if b.manifest.torch and torch_variant:
        h.update(json.dumps(TORCH_VARIANTS[torch_variant], sort_keys=True).encode())
    for p in sorted((PROTOCOL_DIR / "src").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


@dataclass
class EnvStatus:
    backend: Backend
    state: str            # absent | creating | ready | outdated | broken | running
    python: str = ""
    torch_variant: str = ""
    created: str = ""
    detail: str = ""


def read_marker(b: Backend) -> dict | None:
    try:
        return json.loads((b.venv_dir / MARKER).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def status(b: Backend, running_backend: str | None = None) -> EnvStatus:
    if b.id in _busy:
        return EnvStatus(b, "creating", detail=_busy[b.id])
    if not b.venv_dir.exists():
        return EnvStatus(b, "absent")
    marker = read_marker(b)
    if marker is None or not venv_python(b.venv_dir).exists():
        return EnvStatus(b, "broken", detail="incomplete install - use Repair")
    variant = marker.get("torch_variant") or ""
    st = EnvStatus(b, "ready", python=marker.get("python", ""), torch_variant=variant, created=marker.get("created", ""))
    if marker.get("hash") != requirements_hash(b, variant or None):
        st.state, st.detail = "outdated", "requirements changed since install - use Update"
    if running_backend == b.id:
        st.state = "running"
    return st


def is_ready(b: Backend) -> bool:
    return status(b).state in ("ready", "outdated", "running")


# ---------------------------------------------------------------------------
LogFn = Callable[[str], None]


def _run(argv: list[str], log: LogFn, cwd: Path | None = None, timeout: float | None = None) -> None:
    log("$ " + " ".join(argv))
    proc = subprocess.Popen(
        argv, cwd=cwd, env=_sanitized_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, shell=False,
    )
    started = time.monotonic()
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
        if timeout and time.monotonic() - started > timeout:
            proc.kill()
            raise TimeoutError(f"command timed out after {timeout}s")
    if proc.wait() != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {argv[0]} {argv[1] if len(argv) > 1 else ''}")


def selftest(b: Backend, log: LogFn | None = None) -> dict:
    py = venv_python(b.venv_dir)
    argv = [str(py), "-m", "upscale_worker", "--backend", b.id, "--adapter", b.manifest.adapter,
            "--plugin-dir", str(b.dir), "--models-dir", str(b.models_dir), "--selftest"]
    proc = subprocess.run(argv, cwd=b.dir, env=_sanitized_env(), capture_output=True, text=True, timeout=300)
    report: dict = {"ok": False, "error": "no selftest output"}
    for line in (proc.stdout or "").splitlines():
        if line.startswith("SELFTEST "):
            report = json.loads(line[len("SELFTEST "):])
    if log:
        for line in (proc.stderr or "").splitlines()[-40:]:
            log(line)
    return report


def _install(b: Backend, variant: str | None, log: LogFn) -> None:
    uv = find_uv()
    py = venv_python(b.venv_dir)
    if b.manifest.torch and variant:
        spec = TORCH_VARIANTS[variant]
        log(f"Installing PyTorch ({spec['label']})")
        argv = [uv, "pip", "install", "--python", str(py), *spec["packages"]]
        if spec["index"]:
            argv += ["--index-url", spec["index"]]
        _run(argv, log)
    log("Installing backend requirements")
    _run([uv, "pip", "install", "--python", str(py), "-r", str(b.requirements_file)], log, cwd=b.dir)
    log("Installing the worker protocol package")
    _run([uv, "pip", "install", "--python", str(py), str(PROTOCOL_DIR)], log)


def _write_marker(b: Backend, variant: str | None) -> None:
    py = venv_python(b.venv_dir)
    version = subprocess.run([str(py), "-c", "import sys;print(sys.version.split()[0])"],
                             capture_output=True, text=True).stdout.strip()
    marker = {
        "backend": b.id,
        "hash": requirements_hash(b, variant),
        "python": version,
        "torch_variant": variant or "",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (b.venv_dir / MARKER).write_text(json.dumps(marker, indent=2))


def _safe_venv_path(b: Backend) -> Path:
    venv = b.venv_dir.resolve() if b.venv_dir.exists() else b.venv_dir
    if venv.parent != b.dir or venv.name != ".venv":
        raise RuntimeError(f"refusing to touch unexpected path {venv}")
    return venv


class EnvLog:
    """Collects install output for streaming into the UI and a log file."""

    def __init__(self, backend_id: str, op: str):
        self.lines: list[str] = []
        self.path = LOGS_DIR / f"env-{backend_id}.log"
        self.fh = self.path.open("a", encoding="utf-8")
        self.fh.write(f"\n===== {op} {backend_id} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    def __call__(self, line: str) -> None:
        self.lines.append(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def text(self, tail: int = 400) -> str:
        return "\n".join(self.lines[-tail:])

    def close(self) -> None:
        self.fh.close()


def run_operation(op: str, backend_id: str, torch_choice: str | None = None,
                  stop_worker: Callable[[str], None] | None = None) -> Iterator[str]:
    """Generator yielding the accumulated log while a mutation runs on a
    background thread. Ops: create, update, repair, remove, verify."""
    b = registry.backend(backend_id)
    log = EnvLog(b.id, op)
    result: dict = {}

    def work() -> None:
        try:
            if not _installer_lock.acquire(timeout=1):
                raise RuntimeError("another environment operation is running; wait for it to finish")
            try:
                _busy[b.id] = op
                if op != "verify" and stop_worker:
                    stop_worker(b.id)  # never mutate an env under a running worker
                marker = read_marker(b) or {}
                variant = None
                if b.manifest.torch:
                    variant = resolve_torch_variant(torch_choice or marker.get("torch_variant"))
                if op in ("create", "repair"):
                    venv = _safe_venv_path(b)
                    if venv.exists():
                        log(f"Removing {venv}")
                        shutil.rmtree(venv)
                    log(f"Creating Python {b.manifest.python} environment for {b.name}")
                    _run([find_uv(), "venv", str(venv), "--python", b.manifest.python, "--no-project"], log)
                    _install(b, variant, log)
                elif op == "update":
                    if not venv_python(b.venv_dir).exists():
                        raise RuntimeError("environment does not exist - use Install")
                    _install(b, variant, log)
                elif op == "remove":
                    venv = _safe_venv_path(b)
                    if venv.exists():
                        log(f"Removing {venv} (downloaded weights are kept)")
                        shutil.rmtree(venv)
                    log("Removed.")
                    return
                if op in ("create", "repair", "update", "verify"):
                    if not venv_python(b.venv_dir).exists():
                        raise RuntimeError("environment does not exist")
                    log("Running worker self-test")
                    report = selftest(b, log)
                    log(json.dumps(report, indent=2))
                    if not report.get("ok"):
                        raise RuntimeError(f"self-test failed: {report.get('error')}")
                    if op != "verify":
                        _write_marker(b, variant)
                    log("✅ Environment ready." if op != "verify" else "✅ Verification passed.")
            finally:
                _busy.pop(b.id, None)
                _installer_lock.release()
        except Exception as exc:
            log(f"❌ {type(exc).__name__}: {exc}")
            result["error"] = str(exc)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    shown = 0
    while thread.is_alive():
        thread.join(0.5)
        if len(log.lines) != shown:
            shown = len(log.lines)
            yield log.text()
    log.close()
    yield log.text()
    if "error" in result:
        raise RuntimeError(result["error"])


def disk_usage(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total
