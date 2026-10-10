"""Upscale orchestration: the Gradio equivalent of electron/commands/
image-upscayl.ts, double-upscayl.ts and batch-upscayl.ts.

Workers only produce the raw model output at native scale. The master does
everything Upscayl's binary did around it - final scale / custom width,
format, compression, metadata - so it behaves identically for every backend.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from . import jobs, registry
from .paths import JOBS_DIR
from .supervisor import WorkerError, supervisor

VALID_EXTENSIONS = {"png", "jpg", "jpeg", "jfif", "webp", "bmp", "tif", "tiff"}
MAX_INPUT_PIXELS = 100_000_000
MAX_OUTPUT_PIXELS = 1_000_000_000
Image.MAX_IMAGE_PIXELS = MAX_OUTPUT_PIXELS

ProgressFn = Callable[[float, str], None]


class Cancelled(Exception):
    pass


@dataclass
class UpscaleRequest:
    model: str                      # profile key "<backend>/<model id>"
    restore_model: str = ""         # optional pre-pass profile key
    double: bool = False
    scale: int = 4
    use_custom_width: bool = False
    custom_width: int = 0
    save_as: str = "png"
    compression: int = 0
    tile_size: int = 0
    tta: bool = False
    gpu_id: str = ""
    precision: str = "auto"
    copy_metadata: bool = False
    overwrite: bool = False
    output_dir: str = ""


@dataclass
class ImageResult:
    source: str
    display_input: str
    display_output: str
    saved_path: str
    in_size: tuple[int, int]
    out_size: tuple[int, int]
    seconds: float
    reused: bool = False
    notes: list[str] = field(default_factory=list)


def is_valid_image(path: str) -> bool:
    return Path(path).suffix.lower().lstrip(".") in VALID_EXTENSIONS


def target_size(w: int, h: int, req: UpscaleRequest) -> tuple[int, int]:
    """Same math as Upscayl's upscaylResolution (upscayl-steps.tsx)."""
    if req.use_custom_width and req.custom_width > 0:
        return req.custom_width, max(1, round(req.custom_width * h / w))
    factor = req.scale * req.scale if req.double else req.scale
    return w * factor, h * factor


def _scale_tag(req: UpscaleRequest) -> str:
    if req.use_custom_width and req.custom_width > 0:
        return f"{req.custom_width}px"
    # Upscayl names a Double Upscayl like a single pass, so the two overwrite
    # each other; use the effective factor instead.
    return f"{req.scale * req.scale}x" if req.double else f"{req.scale}x"


def output_name(source: Path, req: UpscaleRequest) -> str:
    tag = _scale_tag(req)
    model_id = req.model.split("/", 1)[-1]
    restore = f"_{req.restore_model.split('/', 1)[-1]}" if req.restore_model else ""
    return f"{source.stem}_upscaled_{tag}{restore}_{model_id}.{req.save_as}"


def _normalize_input(src: Path, dst: Path) -> tuple[Image.Image, tuple[int, int]]:
    """Decode once, apply EXIF orientation, reduce to 8-bit RGB/RGBA PNG for
    the workers. Returns the opened original (for metadata) and its size."""
    original = Image.open(src)
    if original.width * original.height > MAX_INPUT_PIXELS:
        raise ValueError(f"image is {original.width}x{original.height}; the limit is {MAX_INPUT_PIXELS // 1_000_000} MP")
    img = ImageOps.exif_transpose(original.copy())
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        img = img.point(lambda v: v / 256).convert("L")
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA" if has_alpha else "RGB")
    img.save(dst, format="PNG", compress_level=1)
    return original, img.size


def _save_final(img: Image.Image, path: Path, req: UpscaleRequest, original: Image.Image) -> list[str]:
    notes: list[str] = []
    fmt = req.save_as.lower()
    kwargs: dict = {}
    if req.copy_metadata:
        exif = original.getexif()
        if exif:
            exif[0x0112] = 1  # orientation already applied
            kwargs["exif"] = exif.tobytes()
        if original.info.get("icc_profile"):
            kwargs["icc_profile"] = original.info["icc_profile"]
        if fmt != "jpg":
            notes.append("On some systems metadata may not be visible in formats other than JPG.")
    c = max(0, min(100, int(req.compression)))
    if fmt == "png":
        img.save(path, format="PNG", compress_level=round(c / 100 * 9), **kwargs)
    elif fmt == "jpg":
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(path, format="JPEG", quality=max(1, 100 - c), subsampling=0 if c < 10 else 2, **kwargs)
    elif fmt == "webp":
        if c == 0:
            img.save(path, format="WEBP", lossless=True, **kwargs)
        else:
            img.save(path, format="WEBP", quality=max(1, 100 - c), **kwargs)
    else:
        raise ValueError(f"unsupported format {fmt}")
    return notes


def _run_passes(job_id: str, work: Path, source_png: Path, req: UpscaleRequest,
                progress: ProgressFn, cancel: threading.Event, base: float, span: float) -> Path:
    keys = ([req.restore_model] if req.restore_model else []) + [req.model] + ([req.model] if req.double else [])
    current = source_png
    options = {"gpu_id": req.gpu_id, "precision": req.precision}
    params = {"tile_size": req.tile_size, "tta": req.tta, "gpu_id": req.gpu_id}
    for i, key in enumerate(keys):
        if cancel.is_set():
            raise Cancelled()
        profile = registry.profile(key)
        stage = f"Pass {i + 1}/{len(keys)} · {profile.label}" if len(keys) > 1 else profile.label
        progress(base + span * i / len(keys), f"{stage}: loading model…")
        handle = supervisor.ensure_model(profile, options)
        out = work / "output" / f"pass{i + 1}.png"
        worker_job = f"{job_id}-{work.name}-p{i + 1}" if work.name != job_id else f"{job_id}-p{i + 1}"
        supervisor.submit(
            handle, worker_job,
            str(current.relative_to(JOBS_DIR)), str(out.relative_to(JOBS_DIR)), params,
        )
        while True:
            if cancel.is_set():
                supervisor.cancel(worker_job)
            st = supervisor.poll(handle, worker_job)
            frac = base + span * (i + st.get("progress", 0.0)) / len(keys)
            progress(frac, f"{stage}: {st.get('message') or ''}")
            if st["state"] == "done":
                break
            if st["state"] == "cancelled":
                raise Cancelled()
            if st["state"] == "failed":
                if st.get("result", {}).get("oom"):
                    supervisor.stop("(out of memory - worker restarted cleanly)")
                    raise WorkerError(
                        f"Out of GPU memory with {profile.label}. Set a smaller tile size in Settings "
                        f"(e.g. {max(64, (req.tile_size or 512) // 2)}), or 0 for automatic."
                    )
                raise WorkerError(st.get("error") or "the worker reported a failure")
            time.sleep(0.15)
        current = out
    return current


def _process(job_id: str, work: Path, src: Path, req: UpscaleRequest, out_path: Path,
             progress: ProgressFn, cancel: threading.Event, base: float = 0.0, span: float = 1.0) -> ImageResult:
    started = time.perf_counter()
    source_png = work / "input" / "source.png"
    original, (w, h) = _normalize_input(src, source_png)
    tw, th = target_size(w, h, req)
    if tw * th > MAX_OUTPUT_PIXELS:
        raise ValueError(f"output would be {tw}x{th} - over {MAX_OUTPUT_PIXELS // 1_000_000} MP. Lower the scale.")
    raw = _run_passes(job_id, work, source_png, req, progress, cancel, base, span)
    progress(base + span * 0.99, "Scaling and converting image...")
    img = Image.open(raw)
    img.load()
    if img.size != (tw, th):
        img = img.resize((tw, th), Image.Resampling.LANCZOS)
    display = work / "output" / f"final.{req.save_as}"
    notes = _save_final(img, display, req, original)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(".tmp-" + out_path.name)
    shutil.copyfile(display, tmp)
    tmp.replace(out_path)
    for f in (work / "output").glob("pass*.png"):
        f.unlink(missing_ok=True)
    return ImageResult(
        source=str(src), display_input=str(source_png), display_output=str(display), saved_path=str(out_path),
        in_size=(w, h), out_size=(tw, th), seconds=round(time.perf_counter() - started, 2), notes=notes,
    )


def run_image(src_path: str, req: UpscaleRequest, progress: ProgressFn, cancel: threading.Event) -> ImageResult:
    src = Path(src_path)
    if not is_valid_image(src_path):
        raise ValueError("Please select an image with a valid extension like PNG, JPG, JPEG, JFIF or WEBP.")
    out_dir = Path(req.output_dir).expanduser()
    out_path = out_dir / output_name(src, req)
    if len(str(out_path)) >= 255:
        progress(0, "Warning: the output path exceeds 255 characters.")

    job_id = jobs.new_id()
    work = jobs.job_dir(job_id)

    if out_path.exists() and not req.overwrite:
        # Upscayl behaviour: reuse an existing result unless overwrite is on.
        source_png = work / "input" / "source.png"
        _normalize_input(src, source_png)
        display = work / "output" / f"final{out_path.suffix}"
        shutil.copyfile(out_path, display)
        with Image.open(display) as im:
            size = im.size
        with Image.open(source_png) as im:
            in_size = im.size
        return ImageResult(str(src), str(source_png), str(display), str(out_path), in_size, size, 0.0, reused=True,
                           notes=["Already upscaled - loaded the existing file. Enable 'Overwrite previous upscale' to process again."])

    mode = "double" if req.double else "image"
    jobs.create(job_id, mode, req.model, req.restore_model or None, asdict(req), src.name)
    try:
        result = _process(job_id, work, src, req, out_path, progress, cancel)
    except Cancelled:
        jobs.finish(job_id, "cancelled")
        raise
    except Exception as exc:
        jobs.finish(job_id, "failed", error=str(exc))
        raise
    jobs.finish(job_id, "done", output_path=result.saved_path, in_w=result.in_size[0], in_h=result.in_size[1],
                out_w=result.out_size[0], out_h=result.out_size[1], seconds=result.seconds)
    return result


def collect_folder(folder: str) -> list[Path]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"not a folder: {folder}")
    return sorted(p for p in root.iterdir() if p.is_file() and is_valid_image(str(p)))


def batch_output_dir(folder_name: str, req: UpscaleRequest) -> Path:
    """Upscayl's batch folder name: upscayl_<format>_<model>_<scale>x (batch-upscayl.ts)."""
    model_id = req.model.split("/", 1)[-1]
    restore = f"_{req.restore_model.split('/', 1)[-1]}" if req.restore_model else ""
    return Path(req.output_dir).expanduser() / f"upscaled_{req.save_as}{restore}_{model_id}_{_scale_tag(req)}"


def run_batch(files: list[Path], folder_name: str, req: UpscaleRequest, progress: ProgressFn,
              cancel: threading.Event) -> tuple[Path, list[ImageResult], list[str]]:
    req.double = False  # Upscayl disables Double Upscayl in batch mode
    if not files:
        raise ValueError("Make sure that the folder contains PNG, JPG, JPEG or WEBP images.")
    out_dir = batch_output_dir(folder_name, req)
    job_id = jobs.new_id()
    jobs.job_dir(job_id)
    jobs.create(job_id, "batch", req.model, req.restore_model or None, {**asdict(req), "files": len(files)}, folder_name)
    results: list[ImageResult] = []
    errors: list[str] = []
    started = time.perf_counter()
    try:
        for i, src in enumerate(files):
            if cancel.is_set():
                raise Cancelled()
            out_path = out_dir / f"{src.stem}.{req.save_as}"
            if out_path.exists() and not req.overwrite:
                continue
            work = JOBS_DIR / job_id / f"{i:05d}"
            (work / "input").mkdir(parents=True, exist_ok=True)
            (work / "output").mkdir(parents=True, exist_ok=True)
            n = len(files)

            def prog(frac: float, msg: str, i=i) -> None:
                progress(frac, f"Batch Upscale In Progress: {i + 1}/{n} · {msg}")

            try:
                results.append(_process(job_id, work, src, req, out_path, prog, cancel, base=i / n, span=1 / n))
            except Cancelled:
                raise
            except WorkerError as exc:
                if "Out of GPU memory" in str(exc) or "crashed" in str(exc):
                    raise
                errors.append(f"{src.name}: {exc}")
            except Exception as exc:
                errors.append(f"{src.name}: {exc}")
    except Cancelled:
        jobs.finish(job_id, "cancelled", output_path=str(out_dir))
        raise
    except Exception as exc:
        jobs.finish(job_id, "failed", error=str(exc), output_path=str(out_dir))
        raise
    jobs.finish(job_id, "done" if not errors else "failed", output_path=str(out_dir),
                error="; ".join(errors[:5]) or None, seconds=round(time.perf_counter() - started, 2))
    return out_dir, results, errors
