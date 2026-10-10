"""Worker supervisor: one active backend worker at a time, started on demand.

State machine:  STOPPED -> STARTING -> READY -> LOADING -> IDLE <-> BUSY
                any -> STOPPING -> STOPPED,  any -> CRASHED

Workers are launched with their own venv interpreter (no activation), in a new
process group/session so the whole tree can be signalled. Process exit is the
authoritative way to release VRAM, so switching to a different backend always
stops the old worker first.
"""

from __future__ import annotations

import collections
import os
import secrets
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import httpx

from upscale_protocol import PROTOCOL_VERSION, protocol_compatible
from upscale_protocol.models import TOKEN_ENV, TOKEN_HEADER

from . import environments
from .paths import JOBS_DIR, LOGS_DIR
from .registry import Backend, Profile


class State(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    LOADING = "loading"
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    CRASHED = "crashed"


class WorkerError(RuntimeError):
    pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class WorkerHandle:
    backend: Backend
    process: subprocess.Popen
    port: int
    token: str
    log_path: Path
    started: float = field(default_factory=time.time)
    loaded_key: str | None = None
    load_signature: tuple | None = None
    load_info: dict = field(default_factory=dict)
    client: httpx.Client | None = None

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    @property
    def pid(self) -> int:
        return self.process.pid


class Supervisor:
    MAX_LOG_BYTES = 5 * 1024 * 1024

    def __init__(self) -> None:
        self.lock = threading.RLock()          # lifecycle lock
        self.active: WorkerHandle | None = None
        self.state = State.STOPPED
        self.last_error = ""
        self.last_activity = time.time()
        self.log_tail: collections.deque[str] = collections.deque(maxlen=2000)
        self.idle_timeout_min = 15
        threading.Thread(target=self._idle_watchdog, daemon=True).start()

    # -- logging ---------------------------------------------------------------
    def _drain(self, handle: WorkerHandle) -> None:
        assert handle.process.stdout is not None
        with handle.log_path.open("a", encoding="utf-8", errors="replace") as fh:
            for line in handle.process.stdout:
                line = line.rstrip()
                self.log_tail.append(f"[{handle.backend.id}] {line}")
                fh.write(line + "\n")
                fh.flush()
                if fh.tell() > self.MAX_LOG_BYTES:
                    fh.truncate(0)
                    fh.seek(0)

    # -- lifecycle -------------------------------------------------------------
    def _start(self, b: Backend) -> WorkerHandle:
        if not environments.is_ready(b):
            raise WorkerError(f"The {b.name} environment is not installed. Open the Environments tab and click Install.")
        python = environments.venv_python(b.venv_dir)
        port = _free_port()
        token = secrets.token_urlsafe(32)
        argv = [
            str(python), "-m", "upscale_worker",
            "--backend", b.id,
            "--adapter", b.manifest.adapter,
            "--plugin-dir", str(b.dir),
            "--models-dir", str(b.models_dir),
            "--spool", str(JOBS_DIR),
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        env = environments._sanitized_env()
        env[TOKEN_ENV] = token
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.state = State.STARTING
        log_path = LOGS_DIR / f"worker-{b.id}.log"
        process = subprocess.Popen(
            argv, cwd=b.dir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, shell=False, **kwargs,
        )
        handle = WorkerHandle(b, process, port, token, log_path)
        handle.client = httpx.Client(
            base_url=f"http://127.0.0.1:{port}", headers={TOKEN_HEADER: token}, timeout=httpx.Timeout(30.0, connect=2.0)
        )
        threading.Thread(target=self._drain, args=(handle,), daemon=True).start()
        self.log_tail.append(f"[supervisor] started {b.id} pid={process.pid} port={port}")

        deadline = time.time() + 180
        while time.time() < deadline:
            if not handle.alive:
                self.state = State.CRASHED
                tail = "\n".join(list(self.log_tail)[-25:])
                raise WorkerError(f"{b.name} worker exited during startup (code {process.returncode}).\n{tail}")
            try:
                health = handle.client.get("/v1/health", timeout=2).json()
                if not protocol_compatible(health.get("protocol_version", "")):
                    self._kill(handle)
                    raise WorkerError(
                        f"{b.name} speaks protocol {health.get('protocol_version')}, master needs {PROTOCOL_VERSION}"
                    )
                self.state = State.READY
                return handle
            except (httpx.TransportError, ValueError):
                time.sleep(0.3)
        self._kill(handle)
        self.state = State.CRASHED
        raise WorkerError(f"{b.name} worker did not become healthy within 180s")

    def _kill(self, handle: WorkerHandle, grace: float = 5.0) -> None:
        if handle.alive:
            try:
                handle.client.post("/v1/shutdown", timeout=2)
            except Exception:
                pass
            try:
                handle.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self._signal(handle, terminate=True)
                try:
                    handle.process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    self._signal(handle, terminate=False)
                    handle.process.wait(timeout=10)
        if handle.client:
            handle.client.close()

    @staticmethod
    def _signal(handle: WorkerHandle, terminate: bool) -> None:
        try:
            if os.name == "nt":
                if terminate:
                    handle.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    handle.process.kill()
            else:
                os.killpg(handle.process.pid, signal.SIGTERM if terminate else signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    def stop(self, reason: str = "") -> None:
        with self.lock:
            if self.active is None:
                self.state = State.STOPPED
                return
            self.state = State.STOPPING
            handle = self.active
            self.active = None
            self.log_tail.append(f"[supervisor] stopping {handle.backend.id} {reason}".rstrip())
            self._kill(handle)
            self.state = State.STOPPED

    def stop_backend(self, backend_id: str) -> None:
        if self.active and self.active.backend.id == backend_id:
            self.stop("(environment change)")

    def ensure_model(self, p: Profile, options: dict) -> WorkerHandle:
        """Make sure the worker for `p.backend` runs and has `p` loaded."""
        signature = (p.key, str(options.get("gpu_id") or ""), str(options.get("precision") or "auto"))
        with self.lock:
            self.last_activity = time.time()
            if self.active and (self.active.backend.id != p.backend.id or not self.active.alive):
                self.stop("(switching backend)" if self.active.alive else "(worker died)")
            if self.active is None:
                self.active = self._start(p.backend)
            handle = self.active
            if handle.load_signature != signature:
                if not p.installed:
                    raise WorkerError(f"{p.label} is not downloaded. Open the Model Manager tab to download it.")
                self.state = State.LOADING
                try:
                    r = handle.client.post(
                        "/v1/load",
                        json={"model_id": p.entry.id, "files": p.relative_files(), "scale": p.entry.scale, "options": options},
                        timeout=600,
                    )
                except httpx.TransportError as exc:
                    self.state = State.CRASHED
                    raise WorkerError(f"worker stopped responding while loading {p.label}: {exc}") from exc
                if r.status_code != 200:
                    self.state = State.READY
                    raise WorkerError(f"Could not load {p.label}: {r.json().get('detail', r.text)}")
                handle.loaded_key = p.key
                handle.load_signature = signature
                handle.load_info = r.json().get("info", {})
                self.log_tail.append(f"[supervisor] loaded {p.key} {handle.load_info}")
            self.state = State.IDLE
            return handle

    def unload_if_loaded(self, key: str) -> None:
        with self.lock:
            if self.active and self.active.loaded_key == key:
                self.stop("(model deleted)")

    # -- jobs ------------------------------------------------------------------
    def submit(self, handle: WorkerHandle, job_id: str, input_rel: str, output_rel: str, params: dict) -> None:
        self.state = State.BUSY
        self.last_activity = time.time()
        r = handle.client.post(
            "/v1/jobs", json={"job_id": job_id, "input": input_rel, "output": output_rel, "parameters": params}
        )
        if r.status_code != 200:
            self.state = State.IDLE
            raise WorkerError(r.json().get("detail", r.text))

    def poll(self, handle: WorkerHandle, job_id: str) -> dict:
        self.last_activity = time.time()
        if not handle.alive:
            self.state = State.CRASHED
            raise WorkerError(f"{handle.backend.name} worker crashed (exit code {handle.process.returncode}). See Diagnostics.")
        r = handle.client.get(f"/v1/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        status = r.json()
        if status["state"] in ("done", "failed", "cancelled"):
            self.state = State.IDLE
        return status

    def cancel(self, job_id: str) -> None:
        handle = self.active
        if handle and handle.alive:
            try:
                handle.client.post(f"/v1/jobs/{job_id}/cancel", timeout=5)
            except Exception:
                pass

    def hard_restart(self) -> str:
        with self.lock:
            backend = self.active.backend if self.active else None
            self.stop("(hard restart)")
            if backend is None:
                return "No worker was running."
            self.active = self._start(backend)
            return f"Restarted {backend.name} (pid {self.active.pid}). The model will reload on the next job."

    # -- introspection ---------------------------------------------------------
    def health(self) -> dict:
        handle = self.active
        if not handle or not handle.alive:
            return {"state": self.state.value}
        try:
            data = handle.client.get("/v1/health", timeout=2).json()
            data["metadata"] = handle.client.get("/v1/metadata", timeout=2).json()
            return data
        except Exception as exc:
            return {"state": self.state.value, "error": str(exc)}

    def status_line(self) -> dict:
        handle = self.active
        if handle and not handle.alive and self.state not in (State.STOPPED, State.STOPPING):
            self.state = State.CRASHED
        return {
            "state": self.state.value,
            "backend": handle.backend.name if handle else "—",
            "pid": handle.pid if handle and handle.alive else None,
            "model": handle.loaded_key if handle else None,
            "device": (handle.load_info or {}).get("device", "") if handle else "",
        }

    def _idle_watchdog(self) -> None:
        while True:
            time.sleep(30)
            try:
                timeout = float(self.idle_timeout_min or 0)
                if (
                    timeout > 0
                    and self.active is not None
                    and self.state in (State.IDLE, State.READY)
                    and time.time() - self.last_activity > timeout * 60
                    and self.lock.acquire(blocking=False)
                ):
                    try:
                        self.stop(f"(idle for {timeout:g} min - VRAM released)")
                    finally:
                        self.lock.release()
            except Exception:
                pass


supervisor = Supervisor()
