# Deblur Studio

**A Gradio image studio for Pinokio** - AI image upscaling, deblurring, denoising and JPEG cleanup, with a
built-in model manager. Every AI backend runs in its own isolated Python environment, managed from the app.

<p align="center"><img src="icon.png" width="128" alt="Deblur Studio"></p>

## What it does

- **A complete upscaler workflow** (UI ported 1:1 from Upscayl) - a step-by-step panel (Batch Upscale, Steps 1-4, Double Upscale, image scale, output
  folder), the same Settings tab (format, compression, metadata copy, custom width, tile size, GPU ID, TTA, overwrite,
  remember output folder, notifications), the before/after **slider** and **lens** views with zoom, the
  *More options* drawer with usage stats, and predictable output names (`<name>_upscaled_<scale>x_<model>.<ext>`).
- **Deblur & restore** - a *Restore first* step runs a deblur / denoise / JPEG-cleanup model (NAFNet, Restormer,
  SCUNet, FBCNN, ...) before the upscaler, or use the combined 4x deblur+upscale models.
- **Model Manager** - browse 44 curated models, download / delete them with checksum verification, or import your
  own (NCNN `.param` + `.bin`, PyTorch `.pth` / `.safetensors`).
- **Environment manager** - install, verify (self-test), update, repair or remove each backend's venv.
- **Jobs & Diagnostics** - job history (SQLite), scratch cleanup, worker logs, health, `nvidia-smi`, hard restart.

### Selection tools: process only part of the image

The floating one-column palette on the right of the canvas holds PhotoCraft's selection tools. Select the
part of the image you want changed and **only that part is fed to the model**: the input is cropped to the
selection's tight bounding box, and any unselected pixels inside that box are replaced by a fill made only
from the selected colours, so nothing you left out reaches the model. The result is blended back through the
selection's edge, so **unselected pixels stay exactly the original**, just resized to the output scale.
The image info under the left panel reports what the model got (`model fed 80×80 (5,140 selected px)`), and a
selection result opens in the slider view with the divider just left of the selection.

| Tool | Key | Use |
|---|---|---|
| Move | V | Pan (also hold Space with any tool) |
| Rectangular / Elliptical / Single Row / Single Column Marquee | M | Drag a box or ellipse (Shift = square, Alt = from centre while dragging) |
| Lasso / Polygonal Lasso / Magnetic Lasso | L | Freehand; click points (Enter or double-click closes); edge-snapping live wire (Width, Contrast, Frequency) |
| Object Selection | W | Drag a rectangle around an object (GrabCut) |
| Quick Selection | W | Paint over an area; it grows to the edges (`[` `]` brush size) |
| Magic Wand | W | Click a colour (Tolerance, Contiguous, Anti-alias) |
| Crop | C | Choose the area that is output; Enter commits, Esc cancels |
| Eyedropper | I | Selects every pixel similar to the clicked colour (Color Range, Fuzziness) |

Shift adds to the selection, Alt subtracts, Shift+Alt intersects (or use the mode buttons in the options bar).
Ctrl+A selects all, Ctrl+D deselects, Ctrl+Shift+I inverts. Shift+key cycles the tools sharing a key.
The algorithms are a JavaScript port of PhotoCraft's (`app/studio/js/selection_algo.js`): Boykov–Kolmogorov
max-flow, GrabCut with GMMs, the Quick Selection min-cut fitted to Photoshop, and the Mortensen–Barrett live
wire. The heavy cuts run in a Web Worker.

### Backends

| Backend | Runs on | Engine | Models |
|---|---|---|---|
| **NCNN Vulkan** (`ncnn-vulkan`) | any Vulkan GPU (NVIDIA, AMD, Intel, Apple) | upstream Tencent **ncnn 1.0.20260526** | Upscayl's 7 built-in models, the upscayl/custom-models collection, Real-ESRGAN anime video |
| **PyTorch** (`spandrel`) | CUDA 13.0 / 12.8, ROCm 7.2, Apple MPS, CPU | **torch 2.14.1** + spandrel 0.4.2 (+ extra arches) | Real-ESRGAN, UltraSharp V1/V2, SwinIR, Swin2SR, HAT, DAT2, RealPLKSR, NAFNet, Restormer, SCUNet, FBCNN |

Upscayl's desktop app ships `upscayl-bin`, which vendors an **ncnn snapshot from April 2022**. The NCNN backend here
reimplements the same Real-ESRGAN inference loop (border pre-padding, tiling, 8x TTA, separate alpha) on the current
upstream ncnn wheel, so the same models run with newer Vulkan kernels (fp16/bf16 cooperative matrix).

## How to use

1. **Install** in Pinokio, then **Start** and open the Web UI.
2. On first launch click **⚡ Quick setup**: it installs the small NCNN backend (~300 MB) and the *Upscayl Standard*
   model. For PyTorch models open **Model Manager**, select models (or **⭐ Select recommended**) and click
   **Download selected**; the backend environment is installed automatically (PyTorch + CUDA is ~5 GB).
3. In the **🚀 Upscale** tab, the panel floating over the canvas walks through four steps: **1 Source**
   (**Select Image**, or drop / paste it on the canvas), **2 AI model** (model, optional *Restore first*, Double
   Upscale, scale), **3 Output** (folder) and **4 Upscale**. Use **Batch Upscale** for a folder. The worker status
   sits above the panel; the file name and size sit below it.
4. Optionally select part of the image with the tools on the right (see above), then click **Upscale 🚀**.
5. Compare with the **Slider** or **Lens** buttons under the canvas; scroll to zoom, drag to pan.

Results are written to the output folder (default `app/outputs`). Weights live in `app/models/<backend>/` and are
never stored inside a venv, so **Repair**/**Remove** on an environment keeps them.

## Architecture

```
Browser ── one stable URL ──► Master (app/env)          Gradio UI, registry, env manager,
                              never imports torch/ncnn   supervisor, SQLite jobs, downloads
                                   │ loopback HTTP + per-launch token, shared job spool
                                   ▼
                              One worker at a time:  plugins/<id>/.venv/bin/python -m upscale_worker
                              (FastAPI, headless)    → adapter.py → model → runtime/jobs/<id>/output
```

- `app/plugins/<id>/manifest.toml` - strictly validated backend manifest + model catalog (URLs, sizes, sha256).
- `app/plugins/<id>/adapter.py` - translates protocol jobs into the backend's API.
- `app/protocol/` - the shared `upscale_protocol` schemas and `upscale_worker` shell (no ML imports).
- Switching to a model on another backend stops the old worker first; process exit is what frees VRAM.
  Workers also stop after the idle timeout (Settings, default 15 min).
- Adding a backend = a new folder with `manifest.toml`, `requirements.txt` and `adapter.py`.

## API

The app exposes two documented Gradio endpoints. Output format, compression, tile size, TTA, precision and GPU follow the saved Settings.

| Endpoint | Inputs | Output |
|---|---|---|
| `/models` | – | list of `{key, name, task, scale, backend, installed}` |
| `/upscale` | `image` (file), `model` (key), `scale` (int, 0 = model's native), `restore_model` (key or `""`), `double` (bool) | upscaled image file |

### Python

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")
models = client.predict(api_name="/models")
print([m["key"] for m in models if m["installed"]])

result = client.predict(
    handle_file("photo.jpg"),
    "ncnn-vulkan/upscayl-standard-4x",   # model key
    4,                                   # output scale
    "spandrel/nafnet-gopro-w64",         # restore first ("" for none)
    False,                               # double upscale
    api_name="/upscale",
)
print(result)  # path to the upscaled image
```

### JavaScript

```javascript
import { Client, handle_file } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860/");
const models = await client.predict("/models", {});
const result = await client.predict("/upscale", {
  image: handle_file(await (await fetch("photo.jpg")).blob()),
  model: "ncnn-vulkan/upscayl-standard-4x",
  scale: 4,
  restore_model: "",
  double: false,
});
console.log(result.data[0].url);
```

### curl

```bash
URL=http://127.0.0.1:7860
# 1. upload the image
FILE=$(curl -s -F "files=@photo.jpg" $URL/gradio_api/upload | python -c 'import json,sys;print(json.load(sys.stdin)[0])')
# 2. start the job
EVENT=$(curl -s -X POST $URL/gradio_api/call/upscale -H "Content-Type: application/json" \
  -d "{\"data\":[{\"path\":\"$FILE\",\"meta\":{\"_type\":\"gradio.FileData\"}},\"ncnn-vulkan/upscayl-standard-4x\",4,\"\",false]}" \
  | python -c 'import json,sys;print(json.load(sys.stdin)["event_id"])')
# 3. stream the result (the last "data:" line contains the output file URL)
curl -s -N $URL/gradio_api/call/upscale/$EVENT
```

## Credits & licenses

- UI and behaviour ported from [Upscayl](https://github.com/upscayl/upscayl) (AGPL-3.0) - this project is AGPL-3.0 too.
- The selection tools copy [PhotoCraft](https://github.com/storytold/photocraft)'s toolbar and port its selection
  algorithms to JavaScript (MIT OR Apache-2.0), with its icons (Lucide, ISC; PhotoCraft glyphs, MIT; one CC BY 3.0
  glyph) - see `app/assets/photocraft-icons/ATTRIBUTION.md`.
- [Tencent ncnn](https://github.com/Tencent/ncnn), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3, see
  `Real-ESRGAN_LICENSE.txt`), [spandrel](https://github.com/chaiNNer-org/spandrel), and every model author listed in
  the Model Manager. Several models are **non-commercial** (CC BY-NC-SA 4.0 or research licenses); the Model Manager
  shows each model's license.
