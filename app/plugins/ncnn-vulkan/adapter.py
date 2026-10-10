"""NCNN Vulkan adapter on the upstream Tencent `ncnn` wheel.

Upscayl's desktop app drives a prebuilt `upscayl-bin` whose ncnn snapshot is
from 2022. This adapter reimplements the same Real-ESRGAN inference loop
(realesrgan-ncnn-vulkan: border pre-padding, tiling, optional 8x TTA, alpha
handled separately) in Python against the current ncnn release, so the same
.param/.bin models benefit from newer Vulkan kernels (fp16/bf16 cooperative
matrix, newer driver workarounds).

The master hands us an 8-bit RGB/RGBA PNG and expects a PNG at the model's
native scale; final resizing, format conversion and metadata happen there.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import ncnn
import numpy as np
from PIL import Image

from upscale_worker.context import JobCancelled, WorkerContext

PREPADDING = 10  # same as realesrgan-ncnn-vulkan
TTA_TRANSFORMS = [(k, f) for f in (False, True) for k in range(4)]


def _blob_names(param_path: Path) -> tuple[str, str]:
    """Find the input blob (top of the Input layer) and the output blob
    ("output" when present, else the last layer's top) from a .param file."""
    lines = param_path.read_text(encoding="utf-8", errors="replace").split("\n")
    input_blob, last_top = "data", None
    names: set[str] = set()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        layer_type, n_in, n_out = parts[0], int(parts[2]), int(parts[3])
        tops = parts[4 + n_in : 4 + n_in + n_out]
        names.update(tops)
        if layer_type == "Input" and tops:
            input_blob = tops[0]
        if tops:
            last_top = tops[-1]
    output_blob = "output" if "output" in names else (last_top or "output")
    return input_blob, output_blob


def _auto_tile(heap_budget_mb: int) -> int:
    # realesrgan-ncnn-vulkan's table, extended for cards with lots of VRAM.
    if heap_budget_mb > 16000:
        return 512
    if heap_budget_mb > 8000:
        return 400
    if heap_budget_mb > 1900:
        return 200
    if heap_budget_mb > 550:
        return 100
    if heap_budget_mb > 190:
        return 64
    return 32


class NcnnVulkanAdapter:
    def __init__(self, ctx: WorkerContext):
        self.ctx = ctx
        self.net: ncnn.Net | None = None
        self.model_id: str | None = None
        self.scale = 4
        self.gpu_index = -1
        self.blobs = ("data", "output")
        self.device_name = "cpu"
        self.fp16 = True

    # -- protocol ------------------------------------------------------------
    def selftest(self) -> dict:
        count = ncnn.get_gpu_count()
        return {
            "ncnn": getattr(ncnn, "__version__", "?"),
            "vulkan_devices": [ncnn.get_gpu_info(i).device_name() for i in range(count)],
        }

    def metadata(self) -> dict:
        return {
            "capabilities": {"tta": True, "tiling": True, "gpu_id": True, "precision": True},
            "runtime": {
                "ncnn": getattr(ncnn, "__version__", "?"),
                "device": self.device_name,
                "fp16": self.fp16,
            },
        }

    def load(self, model_id: str, files: list[str], scale: int, options: dict) -> dict:
        params = [Path(f) for f in files if f.endswith(".param")]
        bins = [Path(f) for f in files if f.endswith(".bin")]
        if len(params) != 1 or len(bins) != 1:
            raise ValueError("an NCNN model needs exactly one .param and one .bin")
        self.unload()

        gpu_count = ncnn.get_gpu_count()
        requested = str(options.get("gpu_id") or "").strip()
        if gpu_count == 0:
            self.gpu_index = -1
        elif requested:
            try:
                self.gpu_index = int(requested)
            except ValueError:
                raise ValueError(f"Invalid GPU id {requested!r}") from None
            if not 0 <= self.gpu_index < gpu_count:
                raise ValueError(f"Invalid GPU id {self.gpu_index}: {gpu_count} Vulkan device(s) found")
        else:
            self.gpu_index = ncnn.get_default_gpu_index()

        precision = str(options.get("precision") or "auto")
        self.fp16 = self.gpu_index >= 0 and precision != "fp32"

        net = ncnn.Net()
        net.opt.use_vulkan_compute = self.gpu_index >= 0
        net.opt.use_fp16_packed = self.fp16
        net.opt.use_fp16_storage = self.fp16
        net.opt.use_fp16_arithmetic = False  # realesrgan keeps arithmetic in fp32
        net.opt.num_threads = max(1, (os.cpu_count() or 2) // 2)
        if self.gpu_index >= 0:
            net.set_vulkan_device(self.gpu_index)
        if net.load_param(str(params[0])) != 0:
            raise RuntimeError(f"ncnn could not parse {params[0].name}")
        if net.load_model(str(bins[0])) != 0:
            raise RuntimeError(f"ncnn could not load {bins[0].name}")

        self.net = net
        self.model_id = model_id
        self.scale = int(scale) if scale else 4
        self.blobs = _blob_names(params[0])
        self.device_name = (
            ncnn.get_gpu_info(self.gpu_index).device_name() if self.gpu_index >= 0 else "cpu"
        )
        return {
            "engine": f"ncnn {getattr(ncnn, '__version__', '')}".strip(),
            "device": self.device_name,
            "fp16": self.fp16,
            "blobs": list(self.blobs),
        }

    def unload(self) -> None:
        if self.net is not None:
            self.net.clear()
        self.net = None
        self.model_id = None

    # -- inference -----------------------------------------------------------
    def _infer(self, chw: np.ndarray) -> np.ndarray:
        # ncnn.Mat wraps the numpy buffer without copying, so the contiguous
        # array must stay referenced until extraction finishes.
        data = np.ascontiguousarray(chw, dtype=np.float32)
        ex = self.net.create_extractor()
        ex.input(self.blobs[0], ncnn.Mat(data))
        ret, out = ex.extract(self.blobs[1])
        del data
        if ret != 0:
            raise RuntimeError(f"ncnn extract failed ({ret}) - try a smaller tile size")
        return np.array(out, copy=True)

    def _infer_tta(self, chw: np.ndarray, tta: bool) -> np.ndarray:
        if not tta:
            return self._infer(chw)
        acc = None
        for k, flip in TTA_TRANSFORMS:
            t = chw[:, :, ::-1] if flip else chw
            t = np.rot90(t, k, axes=(1, 2))
            o = self._infer(t)
            o = np.rot90(o, -k, axes=(1, 2))
            o = o[:, :, ::-1] if flip else o
            acc = o if acc is None else acc + o
        return acc / len(TTA_TRANSFORMS)

    def upscale(self, input_path, output_path, parameters, progress, cancelled) -> dict:
        if self.net is None:
            raise RuntimeError("no model loaded")
        started = time.perf_counter()
        image = Image.open(input_path)
        image.load()
        alpha = image.getchannel("A") if image.mode == "RGBA" else None
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
        _, h, w = rgb.shape

        tile = int(parameters.get("tile_size") or 0)
        if tile <= 0:
            budget = (
                ncnn.get_gpu_device(self.gpu_index).get_heap_budget() if self.gpu_index >= 0 else 4000
            )
            tile = _auto_tile(budget)
        tile = max(32, tile)
        tta = bool(parameters.get("tta"))

        p = PREPADDING
        padded = np.pad(rgb, ((0, 0), (p, p), (p, p)), mode="reflect" if min(h, w) > p else "edge")
        ys, xs = range(0, h, tile), range(0, w, tile)
        total = len(ys) * len(xs)
        out = None
        s = self.scale
        done = 0
        progress(0.0, "0.00%")
        for y in ys:
            for x in xs:
                if cancelled():
                    raise JobCancelled()
                th, tw = min(tile, h - y), min(tile, w - x)
                region = padded[:, y : y + th + 2 * p, x : x + tw + 2 * p]
                result = self._infer_tta(region, tta)
                if out is None:
                    # Trust the network over the catalog: custom models may be
                    # named without their scale.
                    s = max(1, round(result.shape[1] / region.shape[1]))
                    self.scale = s
                    out = np.zeros((3, h * s, w * s), dtype=np.float32)
                out[:, y * s : (y + th) * s, x * s : (x + tw) * s] = result[
                    :3, p * s : (p + th) * s, p * s : (p + tw) * s
                ]
                done += 1
                progress(done / total, f"{100 * done / total:.2f}%")

        progress(1.0, "Scaling and converting image...")
        arr = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8).transpose(1, 2, 0)
        result_img = Image.fromarray(arr, "RGB")
        if alpha is not None:
            result_img.putalpha(alpha.resize(result_img.size, Image.Resampling.LANCZOS))
        result_img.save(output_path, format="PNG", compress_level=1)
        return {
            "engine": "ncnn",
            "device": self.device_name,
            "tile": tile,
            "tta": tta,
            "scale": s,
            "inference_seconds": round(time.perf_counter() - started, 3),
        }
