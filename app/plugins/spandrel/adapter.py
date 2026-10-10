"""PyTorch adapter built on spandrel.

spandrel detects the architecture from a checkpoint's state dict, so one
environment serves Real-ESRGAN, SwinIR, HAT, DAT, NAFNet, Restormer, SCUNet,
FBCNN and the rest. Each model in the manifest is a *profile* pointing at
this shared backend.

The master hands us an 8-bit RGB/RGBA PNG (it normalises EXIF orientation,
16-bit and palette images) and expects a PNG at the model's native scale; it
does final resizing, format conversion and metadata itself.
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import spandrel
import spandrel_extra_arches
from upscale_worker.context import JobCancelled, WorkerContext

# Register Restormer, MPRNet, MIRNet2, ... with spandrel's loader.
spandrel_extra_arches.install()

# Dihedral group: (rot90 k, flip) pairs used for test-time augmentation.
TTA_TRANSFORMS = [(k, f) for f in (False, True) for k in range(4)]
AUTO_TILES = [0, 1024, 512, 256, 128]


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


class SpandrelAdapter:
    def __init__(self, ctx: WorkerContext):
        self.ctx = ctx
        self.descriptor = None
        self.model_id: str | None = None
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.device_name = "cpu"
        self.scale = 1

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _pick_device(gpu_id: str | int | None) -> torch.device:
        if torch.cuda.is_available():
            index = 0
            if gpu_id not in (None, ""):
                try:
                    index = int(gpu_id)
                except ValueError:
                    index = 0
            if index >= torch.cuda.device_count():
                raise ValueError(f"Invalid GPU id {index}: only {torch.cuda.device_count()} CUDA device(s)")
            return torch.device(f"cuda:{index}")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _pick_dtype(self, precision: str) -> torch.dtype:
        d = self.descriptor
        if self.device.type == "cpu":
            return torch.float32
        if precision == "fp32":
            return torch.float32
        if precision == "bf16" and d.supports_bfloat16:
            return torch.bfloat16
        if precision in ("fp16", "auto") and d.supports_half:
            return torch.float16
        if precision == "auto" and d.supports_bfloat16 and self.device.type == "cuda":
            return torch.bfloat16
        return torch.float32

    # -- protocol ------------------------------------------------------------
    def selftest(self) -> dict:
        info = {
            "torch": torch.__version__,
            "spandrel": getattr(spandrel, "__version__", "?"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_build": torch.version.cuda,
            "hip_build": getattr(torch.version, "hip", None),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
        if torch.cuda.is_available():
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            # A real kernel launch catches "no kernel image for this GPU" errors
            # (wrong wheel for the card) at verify time instead of first upscale.
            x = torch.ones(8, device="cuda")
            info["cuda_kernel_ok"] = bool((x * 2).sum().item() == 16)
        return info

    def metadata(self) -> dict:
        runtime = {"torch": torch.__version__, "device": str(self.device), "dtype": str(self.dtype)}
        if self.descriptor is not None:
            d = self.descriptor
            runtime.update(
                architecture=d.architecture.name,
                scale=d.scale,
                input_channels=d.input_channels,
                output_channels=d.output_channels,
                purpose=getattr(d, "purpose", ""),
            )
        return {"capabilities": {"tta": True, "tiling": True, "gpu_id": True, "precision": True}, "runtime": runtime}

    def load(self, model_id: str, files: list[str], scale: int, options: dict) -> dict:
        self.unload()
        weights = [f for f in files if Path(f).suffix.lower() in (".pth", ".safetensors", ".ckpt", ".pt")]
        if len(weights) != 1:
            raise ValueError("spandrel models are a single weight file")
        descriptor = spandrel.ModelLoader().load_from_file(weights[0])
        if not isinstance(descriptor, spandrel.ImageModelDescriptor):
            raise ValueError(
                f"{descriptor.architecture.name} is not a single-image model (inpainting/other models are unsupported)"
            )
        self.descriptor = descriptor
        self.device = self._pick_device(options.get("gpu_id"))
        self.dtype = self._pick_dtype(str(options.get("precision") or "auto"))
        descriptor.to(self.device)
        descriptor.eval()
        if self.dtype == torch.float16:
            descriptor.half()
        elif self.dtype == torch.bfloat16:
            descriptor.bfloat16()
        self.model_id = model_id
        self.scale = descriptor.scale
        self.device_name = (
            torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else str(self.device)
        )
        return {
            "architecture": descriptor.architecture.name,
            "scale": descriptor.scale,
            "device": self.device_name,
            "dtype": str(self.dtype).replace("torch.", ""),
        }

    def unload(self) -> None:
        self.descriptor = None
        self.model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- inference -----------------------------------------------------------
    def _forward(self, tile: torch.Tensor) -> torch.Tensor:
        """Run the model on one BCHW float tile, padding to its size rules."""
        d = self.descriptor
        req = d.size_requirements
        _, _, h, w = tile.shape
        target_h, target_w = max(h, req.minimum), max(w, req.minimum)
        if req.multiple_of > 1:
            target_h = math.ceil(target_h / req.multiple_of) * req.multiple_of
            target_w = math.ceil(target_w / req.multiple_of) * req.multiple_of
        if req.square:
            target_h = target_w = max(target_h, target_w)
        pad_h, pad_w = target_h - h, target_w - w
        if pad_h or pad_w:
            mode = "reflect" if pad_h < h and pad_w < w else "replicate"
            tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode=mode)
        out = d(tile.to(self.device, self.dtype))
        return out[:, :, : h * d.scale, : w * d.scale].float()

    def _forward_tta(self, tile: torch.Tensor, tta: bool) -> torch.Tensor:
        if not tta:
            return self._forward(tile)
        acc = None
        for k, flip in TTA_TRANSFORMS:
            t = tile.flip(-1) if flip else tile
            t = torch.rot90(t, k, dims=(-2, -1))
            o = self._forward(t)
            o = torch.rot90(o, -k, dims=(-2, -1))
            o = o.flip(-1) if flip else o
            acc = o if acc is None else acc + o
        return acc / len(TTA_TRANSFORMS)

    def _run_tiled(self, img: torch.Tensor, tile_size: int, overlap: int, tta: bool, progress, cancelled) -> torch.Tensor:
        _, c, h, w = img.shape
        s = self.descriptor.scale
        if tile_size <= 0 or (tile_size >= h and tile_size >= w):
            progress(0.05, "Processing the image...")
            return self._forward_tta(img, tta).cpu()
        out = torch.zeros((1, self.descriptor.output_channels, h * s, w * s), dtype=torch.float32)
        ys = list(range(0, h, tile_size))
        xs = list(range(0, w, tile_size))
        total = len(ys) * len(xs)
        done = 0
        for y in ys:
            for x in xs:
                if cancelled():
                    raise JobCancelled()
                y0, y1 = max(y - overlap, 0), min(y + tile_size + overlap, h)
                x0, x1 = max(x - overlap, 0), min(x + tile_size + overlap, w)
                result = self._forward_tta(img[:, :, y0:y1, x0:x1], tta).cpu()
                ty, tx = (y - y0) * s, (x - x0) * s
                th, tw = (min(y + tile_size, h) - y) * s, (min(x + tile_size, w) - x) * s
                out[:, :, y * s : y * s + th, x * s : x * s + tw] = result[:, :, ty : ty + th, tx : tx + tw]
                done += 1
                progress(done / total, f"{100 * done / total:.2f}%")
        return out

    @torch.inference_mode()
    def upscale(self, input_path, output_path, parameters, progress, cancelled) -> dict:
        if self.descriptor is None:
            raise RuntimeError("no model loaded")
        d = self.descriptor
        started = time.perf_counter()
        image = Image.open(input_path)
        image.load()
        alpha = image.getchannel("A") if image.mode == "RGBA" else None
        if d.input_channels == 1:
            rgb = np.asarray(image.convert("L"), dtype=np.float32)[..., None] / 255.0
        else:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]

        requested = int(parameters.get("tile_size") or 0)
        tta = bool(parameters.get("tta"))
        overlap = int(parameters.get("tile_pad") or 16)
        candidates = [requested] if requested > 0 else AUTO_TILES
        result = None
        used_tile = None
        last_exc: BaseException | None = None
        for tile in candidates:
            try:
                result = self._run_tiled(tensor, tile, overlap, tta, progress, cancelled)
                used_tile = tile
                break
            except JobCancelled:
                raise
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                last_exc = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if requested > 0:
                    raise RuntimeError(
                        f"Out of memory with tile size {requested}. Lower the tile size in Settings or set it to 0 (auto)."
                    ) from exc
                print(f"[spandrel] OOM at tile={tile or 'full'}, retrying smaller", flush=True)
        if result is None:
            raise RuntimeError(f"Out of memory even at the smallest tile size: {last_exc}")

        progress(1.0, "Scaling and converting image...")
        arr = result[0].clamp_(0, 1).mul_(255).round_().byte().permute(1, 2, 0).numpy()
        if arr.shape[2] == 1:
            out = Image.fromarray(arr[..., 0], "L").convert("RGB")
        else:
            out = Image.fromarray(arr[..., :3], "RGB")
        if alpha is not None:
            out.putalpha(alpha.resize(out.size, Image.Resampling.LANCZOS))
        out.save(output_path, format="PNG", compress_level=1)
        return {
            "architecture": d.architecture.name,
            "tile": used_tile or "full",
            "tta": tta,
            "inference_seconds": round(time.perf_counter() - started, 3),
        }
