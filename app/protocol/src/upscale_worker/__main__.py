import argparse
import hmac
import importlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from upscale_protocol.models import (
    PROTOCOL_VERSION,
    TERMINAL_STATES,
    TOKEN_ENV,
    TOKEN_HEADER,
    Health,
    JobRequest,
    JobStatus,
    LoadRequest,
    LoadResponse,
    Metadata,
)

from .context import JobCancelled, PathEscapeError, WorkerContext


def log(*parts: Any) -> None:
    print("[worker]", *parts, flush=True)


def load_adapter_class(plugin_dir: Path, spec: str):
    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise SystemExit(f"bad --adapter spec {spec!r}; expected module:Class")
    sys.path.insert(0, str(plugin_dir))
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class _Job:
    def __init__(self, request: JobRequest):
        self.request = request
        self.cancel = threading.Event()
        self.status = JobStatus(job_id=request.job_id, state="queued")


class WorkerRuntime:
    """Owns the adapter and runs one job at a time on a background thread."""

    def __init__(self, ctx: WorkerContext, adapter):
        self.ctx = ctx
        self.adapter = adapter
        self.lock = threading.Lock()
        self.jobs: dict[str, _Job] = {}
        self.current: "_Job | None" = None
        self.loaded_model: "str | None" = None
        self.loading = False

    # -- lifecycle -----------------------------------------------------------
    def load(self, req: LoadRequest) -> LoadResponse:
        with self.lock:
            if self.current and self.current.status.state not in TERMINAL_STATES:
                raise RuntimeError("worker is busy")
            files = [str(self.ctx.model_path(f)) for f in req.files]
            for f in files:
                if not Path(f).exists():
                    raise FileNotFoundError(f"weight file missing: {f}")
            self.loading = True
            started = time.perf_counter()
            try:
                info = self.adapter.load(req.model_id, files, req.scale, req.options) or {}
            finally:
                self.loading = False
            self.loaded_model = req.model_id
            return LoadResponse(
                model_id=req.model_id,
                seconds=round(time.perf_counter() - started, 3),
                info=info,
            )

    def unload(self) -> None:
        with self.lock:
            self.adapter.unload()
            self.loaded_model = None

    # -- jobs ----------------------------------------------------------------
    def submit(self, req: JobRequest) -> JobStatus:
        with self.lock:
            if self.loaded_model is None:
                raise RuntimeError("no model loaded")
            if self.current and self.current.status.state not in TERMINAL_STATES:
                raise RuntimeError("worker is busy")
            input_path = self.ctx.spool_path(req.input)
            output_path = self.ctx.spool_path(req.output)
            if not input_path.is_file():
                raise FileNotFoundError(f"input missing: {req.input}")
            job = _Job(req)
            self.jobs[req.job_id] = job
            self.current = job
            # Bound memory: keep only the most recent statuses.
            for old in list(self.jobs)[:-32]:
                self.jobs.pop(old, None)
        threading.Thread(
            target=self._run, args=(job, input_path, output_path), daemon=True
        ).start()
        return job.status

    def _run(self, job: _Job, input_path: Path, output_path: Path) -> None:
        status = job.status
        status.state = "running"
        status.message = "Starting"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)

        def progress(fraction: float, message: str = "") -> None:
            status.progress = max(0.0, min(1.0, float(fraction)))
            if message:
                status.message = message

        def cancelled() -> bool:
            return job.cancel.is_set()

        started = time.perf_counter()
        try:
            result = self.adapter.upscale(
                str(input_path), str(tmp), dict(job.request.parameters), progress, cancelled
            ) or {}
            if job.cancel.is_set():
                raise JobCancelled()
            if not tmp.is_file() or tmp.stat().st_size == 0:
                raise RuntimeError("adapter produced no output")
            os.replace(tmp, output_path)
            result["seconds"] = round(time.perf_counter() - started, 3)
            status.result = result
            status.progress = 1.0
            status.message = "Done"
            status.state = "done"
        except JobCancelled:
            status.state = "cancelled"
            status.message = "Cancelled"
        except Exception as exc:  # report, never crash the worker
            traceback.print_exc()
            status.state = "failed"
            status.error = f"{type(exc).__name__}: {exc}"
            status.message = "Failed"
            if _is_oom(exc):
                status.result = {"oom": True}
        finally:
            tmp.unlink(missing_ok=True)

    def cancel(self, job_id: str) -> JobStatus:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.cancel.set()
        return job.status


def _is_oom(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "out of memory" in text or "outofmemory" in text


def build_app(runtime: WorkerRuntime, token: str):
    def auth(request: Request) -> None:
        supplied = request.headers.get(TOKEN_HEADER, "")
        if not token or not hmac.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="bad worker token")

    app = FastAPI(title=f"upscale worker: {runtime.ctx.backend}", dependencies=[Depends(auth)])

    def fail(exc: Exception, code: int = 400):
        raise HTTPException(status_code=code, detail=f"{type(exc).__name__}: {exc}")

    @app.get("/v1/health", response_model=Health)
    def health():
        busy = runtime.current is not None and runtime.current.status.state not in TERMINAL_STATES
        return Health(
            status="loading" if runtime.loading else ("busy" if busy else "ok"),
            protocol_version=PROTOCOL_VERSION,
            backend=runtime.ctx.backend,
            pid=os.getpid(),
            device=str(getattr(runtime.adapter, "device_name", "")),
            loaded_model=runtime.loaded_model,
        )

    @app.get("/v1/metadata", response_model=Metadata)
    def metadata():
        meta = runtime.adapter.metadata() if hasattr(runtime.adapter, "metadata") else {}
        return Metadata(
            backend=runtime.ctx.backend,
            protocol_version=PROTOCOL_VERSION,
            adapter=type(runtime.adapter).__name__,
            capabilities=meta.get("capabilities", {}),
            runtime=meta.get("runtime", {}),
        )

    @app.post("/v1/load", response_model=LoadResponse)
    def load(req: LoadRequest):
        try:
            return runtime.load(req)
        except (PathEscapeError, FileNotFoundError, RuntimeError, ValueError) as exc:
            fail(exc)
        except Exception as exc:
            traceback.print_exc()
            fail(exc, 500)

    @app.post("/v1/jobs", response_model=JobStatus)
    def submit(req: JobRequest):
        try:
            return runtime.submit(req)
        except (PathEscapeError, FileNotFoundError, RuntimeError) as exc:
            fail(exc)

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus)
    def job_status(job_id: str):
        job = runtime.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.status

    @app.get("/v1/jobs/{job_id}/result", response_model=JobStatus)
    def job_result(job_id: str):
        job = runtime.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if job.status.state not in TERMINAL_STATES:
            raise HTTPException(status_code=409, detail="job not finished")
        return job.status

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobStatus)
    def cancel(job_id: str):
        try:
            return runtime.cancel(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown job")

    @app.post("/v1/unload")
    def unload():
        runtime.unload()
        return {"unloaded": True}

    @app.post("/v1/shutdown")
    def shutdown():
        if runtime.current:
            runtime.current.cancel.set()
        # Exit shortly after the response is flushed. Process exit is the
        # authoritative way to hand VRAM back to the driver.
        threading.Timer(0.3, lambda: os._exit(0)).start()
        return {"shutting_down": True}

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="upscale_worker")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--adapter", required=True, help="module:Class inside --plugin-dir")
    parser.add_argument("--plugin-dir", required=True, type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--spool", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--selftest", action="store_true", help="import the adapter, print a JSON report and exit")
    args = parser.parse_args(argv)

    plugin_dir = args.plugin_dir.resolve()
    ctx = WorkerContext(
        backend=args.backend,
        plugin_dir=plugin_dir,
        models_dir=(args.models_dir or plugin_dir / "models").resolve(),
        spool_dir=(args.spool or plugin_dir / "spool").resolve(),
    )
    adapter_cls = load_adapter_class(plugin_dir, args.adapter)

    if args.selftest:
        report: dict[str, Any] = {"ok": True, "python": sys.version.split()[0], "protocol": PROTOCOL_VERSION}
        try:
            adapter = adapter_cls(ctx)
            if hasattr(adapter, "selftest"):
                report.update(adapter.selftest() or {})
        except Exception as exc:
            report["ok"] = False
            report["error"] = f"{type(exc).__name__}: {exc}"
            report["traceback"] = traceback.format_exc()
        print("SELFTEST " + json.dumps(report), flush=True)
        raise SystemExit(0 if report["ok"] else 1)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("workers only bind to loopback")
    token = os.environ.pop(TOKEN_ENV, "")
    if not token:
        raise SystemExit(f"{TOKEN_ENV} is not set; refusing to start an unauthenticated worker")

    ctx.spool_dir.mkdir(parents=True, exist_ok=True)
    adapter = adapter_cls(ctx)
    runtime = WorkerRuntime(ctx, adapter)
    app = build_app(runtime, token)

    import uvicorn

    log(f"backend={args.backend} pid={os.getpid()} listening on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
