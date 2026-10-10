"""Gradio UI: a 1:1 port of Upscayl's renderer plus the model/environment
manager tabs.

Upscayl component            ->  here
---------------------------------------------------------------------------
sidebar (collapsible)        ->  gr.Sidebar(position="left")
  Header / Tabs / Footer     ->  HTML header, gr.Tabs("Upscayl", "Settings"), HTML footer
  UpscaylSteps (Step 1-4)    ->  batch toggle, Select Image/Folder, model + double
                                 upscayl + scale, output folder, Upscayl button
  SelectModelDialog          ->  "Browse models" gallery panel with before/after cards
  SettingsTab                ->  Settings tab (same controls, same order)
main content                 ->  Studio tab: image drop zone / slider+lens viewer
  ProgressBar (+ STOP)       ->  progress card + Stop button
  MoreOptionsDrawer          ->  gr.Sidebar(position="right"): reset, lens/slider, zoom, stats
  OnboardingDialog           ->  first-run welcome card with one-click quick setup
new                          ->  Model Manager, Environments, Jobs, Diagnostics tabs
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from . import downloads, environments, jobs, native, registry, settings, sysinfo, viewer
from .logs import logit, text as log_text
from .paths import ASSETS_DIR, DEFAULT_OUTPUT_DIR, MODELS_DIR, RUNTIME_DIR
from .pipeline import Cancelled, UpscaleRequest, collect_folder, run_batch, run_image
from .supervisor import WorkerError, supervisor

APP_NAME = "Deblur Studio"
VERSION = "1.0.0"
PREVIEW_DIR = RUNTIME_DIR / "previews"
RESTORE_TASKS = {"deblur", "denoise", "jpeg", "restore"}

_cancel: dict[str, threading.Event] = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _session(request: gr.Request | None) -> str:
    return getattr(request, "session_hash", None) or "default"


def _toast_ok(msg: str) -> None:
    if not settings.load().get("turn_off_notifications"):
        gr.Info(msg)


def _backend_short(b: registry.Backend) -> str:
    return "NCNN" if b.id == "ncnn-vulkan" else ("PyTorch" if b.id == "spandrel" else b.id)


def model_choices() -> list[tuple[str, str]]:
    items = []
    for p in registry.installed_profiles():
        task = "" if p.entry.task == "upscale" else f" [{registry.TASK_LABELS[p.entry.task]}]"
        items.append((f"{p.label}{task} · {_backend_short(p.backend)}", p.key))
    return items


def restore_choices() -> list[tuple[str, str]]:
    items = [("None", "")]
    for p in registry.installed_profiles():
        if p.entry.task in RESTORE_TASKS:
            items.append((f"{p.label} ({p.entry.scale}x) · {_backend_short(p.backend)}", p.key))
    return items


def _valid_selection(key: str, choices: list[tuple[str, str]]) -> str | None:
    keys = [k for _, k in choices]
    if key in keys:
        return key
    return keys[0] if keys else None


def _scale_label(scale: int, use_custom_width: bool) -> str:
    warn = " ⚠️" if int(scale) >= 6 else ""
    return f"Image Scale ({int(scale)}X){warn}" + (" — DISABLED (custom width)" if use_custom_width else "")


def _resolution_text(dims: dict | None, scale: int, double: bool, use_cw: bool, cw: int) -> str:
    if not dims or not dims.get("w"):
        return ""
    w, h = dims["w"], dims["h"]
    if use_cw and cw and cw > 0:
        nw, nh = int(cw), round(int(cw) * h / w)
    else:
        f = int(scale) * (int(scale) if double else 1)
        nw, nh = w * f, h * f
    return f"Upscale from **{w}x{h}** to **{nw}x{nh}**"


def _fmt_duration(seconds: float) -> str:
    seconds = float(seconds or 0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {seconds % 60:.0f}s"
    return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60)}m"


def stats_html() -> str:
    s = jobs.stats()
    last = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["last_used"])) if s["last_used"] else "—"
    rows = [
        ("Total Upscales", s["total"]),
        ("Total Batch Upscales", s["batch"]),
        ("Total Image Upscales", s["image"]),
        ("Total Double Upscales", s["double"]),
        ("Average Upscale Time", _fmt_duration(s["avg_seconds"])),
        ("Last Upscale Duration", _fmt_duration(s["last_seconds"])),
        ("Last Used At", last),
    ]
    cells = "".join(f'<div class="ds-stat"><div class="ds-stat-title">{t}</div><div class="ds-stat-value">{v}</div></div>'
                    for t, v in rows)
    return f'<div class="ds-stats"><p class="ds-stats-head">STATS</p>{cells}</div>'


def status_strip() -> str:
    st = supervisor.status_line()
    color = {"busy": "#f59e0b", "loading": "#f59e0b", "starting": "#f59e0b", "idle": "#22c55e", "ready": "#22c55e",
             "crashed": "#ef4444"}.get(st["state"], "#64748b")
    pid = f" · pid {st['pid']}" if st["pid"] else ""
    model = st["model"].split("/", 1)[-1] if st["model"] else "no model loaded"
    device = f" · {st['device']}" if st["device"] else ""
    return (f'<div class="ds-strip"><span class="ds-dot" style="background:{color}"></span>'
            f'<b>{st["backend"]}</b> · {st["state"]}{pid} · {model}{device}</div>')


def header_html() -> str:
    logo = viewer.file_url(str(ASSETS_DIR / "logo.png")).split("?")[0]
    return f"""
<a class="ds-header" href="https://github.com/ThomasEricB/deblur-studio-for-pinokio" target="_blank" title="Star us on GitHub 😁">
  <img src="{logo}" alt="logo"/>
  <div><h1>{APP_NAME} <span>{VERSION}</span></h1><p>AI Image Upscaler &amp; Deblur</p></div>
</a>"""


FOOTER_HTML = f"""
<div class="ds-footer">
  <p>Copyright © {time.strftime('%Y')} - <b>{APP_NAME}</b></p>
  <p>AI upscaling, deblurring &amp; restoration - runs locally</p>
</div>"""


def instructions_md(batch: bool) -> str:
    if batch:
        return ("### Select a Folder\nMake sure that the folder doesn't contain anything except PNG, JPG, JPEG & WEBP images.")
    return ("Select an Image\n\nSelect or drag and drop a PNG, JPG, JPEG or WEBP image.\n\n"
            "Hit Ctrl+V or ⌘+V to Paste image from Clipboard")


def progress_html(message: str, batch_count: str = "") -> str:
    return f"""
<div class="ds-progress">
  <img class="ds-spin" src="{viewer.file_url(str(ASSETS_DIR / 'logo.png')).split('?')[0]}"/>
  {f'<p class="ds-progress-batch">{batch_count}</p>' if batch_count else ''}
  <p class="ds-progress-msg">{message}</p>
  <p class="ds-progress-sub">Working its magic...</p>
</div>"""


def _preview_gallery() -> list[tuple[str, str]]:
    """Upscayl's model-selection cards: before | after composites."""
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for p in registry.profiles():
        if not p.entry.preview:
            continue
        src = ASSETS_DIR / p.entry.preview
        before, after = src / "before.webp", src / "after.webp"
        if not (before.exists() and after.exists()):
            continue
        out = PREVIEW_DIR / f"{p.backend.id}--{p.entry.id}.webp"
        if not out.exists():
            b, a = Image.open(before).convert("RGB"), Image.open(after).convert("RGB")
            h = 360
            b = b.resize((round(b.width * h / b.height), h))
            a = a.resize((round(a.width * h / a.height), h))
            half = min(b.width, a.width) // 2
            canvas = Image.new("RGB", (half * 2 + 2, h), "white")
            canvas.paste(b.crop((0, 0, half, h)), (0, 0))
            canvas.paste(a.crop((a.width - half, 0, a.width, h)), (half + 2, 0))
            canvas.save(out, quality=88)
        badge = "" if p.installed else "  ⬇ not downloaded"
        items.append((str(out), f"{p.label} — {p.entry.description}{badge}"))
    return items


def _gallery_keys() -> list[str]:
    return [p.key for p in registry.profiles() if p.entry.preview and (ASSETS_DIR / p.entry.preview / "before.webp").exists()]


# ---- model manager table ----------------------------------------------------
def models_table(backend_filter: str, task_filter: str, status_filter: str) -> list[list]:
    rows = []
    for p in registry.profiles():
        if backend_filter not in ("All", p.backend.id):
            continue
        if task_filter != "All" and registry.TASK_LABELS[p.entry.task] != task_filter:
            continue
        inst = p.installed
        if status_filter == "Installed" and not inst:
            continue
        if status_filter == "Not installed" and inst:
            continue
        lic = p.entry.license + ("" if p.entry.commercial in (None, True) else " (non-commercial)")
        rows.append([
            "✅" if inst else ("⭐" if p.entry.recommended else ""),
            p.label + (" (custom)" if p.custom else ""),
            registry.TASK_LABELS[p.entry.task],
            f"{p.entry.scale}x",
            _backend_short(p.backend),
            downloads.human(p.size_bytes) if p.size_bytes else "—",
            lic,
            p.entry.description,
            p.key,
        ])
    return rows


def model_select_choices() -> list[tuple[str, str]]:
    return [(f"{'✅ ' if p.installed else ''}{p.label} · {_backend_short(p.backend)}", p.key) for p in registry.profiles()]


def models_summary() -> str:
    lines = []
    for b in registry.backends():
        ps = [p for p in registry.profiles() if p.backend.id == b.id]
        n = sum(p.installed for p in ps)
        used = environments.disk_usage(b.models_dir)
        env = environments.status(b, supervisor.active.backend.id if supervisor.active else None)
        lines.append(f"**{b.name}** — {n}/{len(ps)} models downloaded ({downloads.human(used)}) · environment: `{env.state}`")
    return "\n\n".join(lines) + f"\n\nWeights folder: `{MODELS_DIR}`"


# ---- environments table -----------------------------------------------------
def env_rows() -> list[list]:
    running = supervisor.active.backend.id if supervisor.active and supervisor.active.alive else None
    rows = []
    for b in registry.backends():
        st = environments.status(b, running)
        ps = [p for p in registry.profiles() if p.backend.id == b.id]
        rows.append([
            b.name, b.id, st.state, st.python or f"({b.manifest.python})",
            st.torch_variant or ("—" if not b.manifest.torch else f"auto → {environments.detect_torch_variant()}"),
            downloads.human(environments.disk_usage(b.venv_dir)) if b.venv_dir.exists() else "—",
            f"{sum(p.installed for p in ps)}/{len(ps)}",
            st.detail or st.created,
        ])
    return rows


def jobs_rows() -> list[list]:
    rows = []
    for j in jobs.recent():
        dims = f"{j['in_w']}x{j['in_h']} → {j['out_w']}x{j['out_h']}" if j.get("out_w") else ""
        rows.append([
            j["id"], time.strftime("%m-%d %H:%M:%S", time.localtime(j["created"])), j["mode"],
            j["model"].split("/", 1)[-1] + (f" ← {j['restore_model'].split('/', 1)[-1]}" if j.get("restore_model") else ""),
            j["status"], dims, _fmt_duration(j["seconds"]) if j.get("seconds") else "",
            j.get("output_path") or "", (j.get("error") or "")[:200],
        ])
    return rows


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build() -> gr.Blocks:
    s = settings.load()
    supervisor.idle_timeout_min = s["idle_timeout_min"]
    initial_models = model_choices()
    initial_model = _valid_selection(s["selected_model"], initial_models)
    restore_opts = restore_choices()
    initial_restore = _valid_selection(s["restore_model"], restore_opts) or ""
    output_default = s["saved_output_path"] if s["remember_output_folder"] else str(DEFAULT_OUTPUT_DIR)

    with gr.Blocks(title=APP_NAME, fill_width=True) as demo:
        dims_state = gr.State({})
        batch_files_state = gr.State([])
        last_result_state = gr.State({})

        # =================================================================
        # RIGHT DRAWER - MoreOptionsDrawer (only shown on the Upscayl tab)
        # =================================================================
        with gr.Sidebar(position="right", open=False, width=380, label="More options", elem_id="ds-drawer") as drawer:
            reset_image_btn = gr.Button("Reset Image", variant="primary")
            view_type = gr.Radio([("Lens View", "lens"), ("Slider View", "slider")], value=s["view_type"], label="View")
            zoom_amount = gr.Slider(100, 1000, value=s["zoom_amount"], step=10, label="Zoom Amount (%)",
                                    visible=s["view_type"] == "slider")
            stats_box = gr.HTML(stats_html())

        # =================================================================
        # ONE TAB STRIP ON TOP OF EVERYTHING (binder-style, see style.css)
        # Upscayl's sidebar tabs (Upscayl / Settings) and the manager tabs are
        # merged; the left panel lives inside the first tab only.
        # =================================================================
        status_box = gr.HTML(status_strip(), elem_id="ds-status")  # floats at the right end of the strip
        with gr.Tabs(elem_id="ds-main-tabs") as main_tabs:
            # -------------------------- Upscayl --------------------------
            with gr.Tab("🚀 Upscale", id="studio"):
                with gr.Row(equal_height=False, elem_id="ds-studio"):
                    # ---------- left panel: Upscayl's sidebar (UpscaylSteps) ----------
                    with gr.Column(scale=0, min_width=340, elem_id="ds-sidebar") as sidebar_col:
                        with gr.Row(elem_id="ds-side-top"):
                            gr.HTML(header_html(), padding=False)
                            collapse_btn = gr.Button("◀", size="sm", scale=0, min_width=34, elem_id="ds-collapse")
                        batch_mode = gr.Checkbox(False, label="Batch Upscale",
                                                 info="Upscale all images in a folder at once")

                        gr.Markdown("STEP 1", elem_classes="ds-step")
                        select_image_btn = gr.UploadButton("Select Image", file_types=["image"], variant="primary",
                                                           elem_classes="ds-btn")
                        with gr.Group(visible=False) as folder_group:
                            folder_path = gr.Textbox(label="Folder", placeholder="Select or paste a folder path",
                                                     max_lines=1)
                            with gr.Row():
                                select_folder_btn = gr.Button("Select Folder", variant="primary", size="sm")
                                upload_folder_btn = gr.UploadButton("Upload Folder", file_count="directory", size="sm")

                        gr.Markdown("STEP 2", elem_classes="ds-step")
                        gr.Markdown("Select AI Model", elem_classes="ds-sub")
                        model_dd = gr.Dropdown(initial_models, value=initial_model, show_label=False, filterable=True,
                                               elem_id="ds-model")
                        browse_models_btn = gr.Button("🎨 Browse models", size="sm")
                        restore_dd = gr.Dropdown(restore_opts, value=initial_restore, label="Restore first (optional)",
                                                 info="Run a deblur / denoise / JPEG-cleanup model before upscaling")
                        double_upscayl = gr.Checkbox(s["double_upscayl"], label="Double Upscale",
                                                     info="Run the model twice on an image. Significantly slower; scales above 4X may cause performance issues.")
                        scale_step = gr.Slider(1, 16, value=s["scale"], step=1,
                                               label=_scale_label(s["scale"], s["use_custom_width"]),
                                               interactive=not s["use_custom_width"])

                        gr.Markdown("STEP 3", elem_classes="ds-step")
                        output_folder = gr.Textbox(output_default, label="Output folder",
                                                   info="Defaults to the app's outputs folder", max_lines=1)
                        with gr.Row():
                            set_output_btn = gr.Button("Set Output Folder", variant="primary", size="sm")
                            open_output_btn = gr.Button("📂 Open", size="sm")

                        gr.Markdown("STEP 4", elem_classes="ds-step")
                        resolution_md = gr.Markdown("", elem_classes="ds-sub")
                        upscayl_btn = gr.Button("Upscale 🚀", variant="primary", size="lg", elem_id="ds-go")
                        stop_btn_side = gr.Button("STOP", variant="stop", visible=False)
                        gr.HTML(FOOTER_HTML, padding=False)

                    # ---------- main content ----------
                    with gr.Column(scale=1, elem_id="ds-main"):
                        expand_btn = gr.Button("▶ Show panel", size="sm", visible=False, elem_id="ds-expand")
                        with gr.Column(visible=not s["onboarded"], elem_classes="ds-card") as onboarding:
                            gr.Markdown(
                                f"## Welcome to {APP_NAME} 🎉\n"
                                "Each AI backend runs in its own isolated environment, and you choose which models to keep.\n\n"
                                "**Quick setup** installs the small NCNN Vulkan backend (works on any GPU) and the "
                                "*Upscayl Standard* model (~35 MB). Install PyTorch backends and deblur models later from "
                                "**Model Manager** and **Environments**."
                            )
                            with gr.Row():
                                quick_setup_btn = gr.Button("⚡ Quick setup", variant="primary")
                                skip_onboarding_btn = gr.Button("Get Started")
                            quick_log = gr.Textbox(label="Setup log", lines=8, visible=False, autoscroll=True)

                        with gr.Column(visible=False, elem_classes="ds-card") as model_dialog:
                            gr.Markdown("### Select AI Model\nClick a card to select it. Models marked *not downloaded* "
                                        "can be fetched in one click.")
                            model_gallery = gr.Gallery(_preview_gallery(), columns=2, height=520, object_fit="cover",
                                                       allow_preview=False, show_label=False)
                            with gr.Row():
                                gallery_download_btn = gr.Button("⬇ Download selected", visible=False, variant="primary")
                                close_dialog_btn = gr.Button("Close")
                            gallery_pick = gr.State("")

                        image_in = gr.Image(type="filepath", sources=["upload", "clipboard"], show_label=False,
                                            height="70vh", elem_id="ds-input", placeholder=instructions_md(False),
                                            buttons=["fullscreen"])
                        batch_info = gr.Markdown("", visible=False, elem_classes="ds-card")
                        progress_box = gr.HTML("", visible=False)
                        stop_btn = gr.Button("STOP", variant="stop", visible=False, elem_id="ds-stop")
                        compare = viewer.component(visible=False)
                        with gr.Row(visible=False) as result_row:
                            result_md = gr.Markdown("")
                            download_btn = gr.DownloadButton("⬇ Download", size="sm")
                            open_result_btn = gr.Button("📂 Open Output Folder", size="sm")
                        batch_gallery = gr.Gallery(visible=False, columns=4, height=480, label="Upscaled",
                                                   object_fit="contain")

            # -------------------------- Settings (Upscayl's SettingsTab) --------------------------
            with gr.Tab("⚙️ Settings", id="settings"):
                with gr.Row(equal_height=False):
                    with gr.Column(elem_classes="ds-card"):
                        gr.Markdown("OUTPUT", elem_classes="ds-step")
                        save_as = gr.Radio([("PNG", "png"), ("JPG", "jpg"), ("WEBP", "webp")], value=s["save_image_as"],
                                           label="SAVE IMAGE AS")
                        copy_metadata = gr.Checkbox(s["copy_metadata"], label="COPY IMAGE METADATA",
                                                    info="Keep original image metadata (EXIF) in the upscaled image.")
                        scale_settings = gr.Slider(1, 16, value=s["scale"], step=1,
                                                   label=_scale_label(s["scale"], s["use_custom_width"]),
                                                   info="Anything above the model's own scale only resizes the image and does not use AI upscaling.",
                                                   interactive=not s["use_custom_width"])
                        use_custom_width = gr.Checkbox(s["use_custom_width"], label="CUSTOM OUTPUT WIDTH",
                                                       info="Use a custom width for the output images. The height is adjusted automatically. Overrides the scale setting.")
                        custom_width = gr.Number(s["custom_width"] or None, label="Width (px)", precision=0, minimum=1,
                                                 interactive=s["use_custom_width"])
                        compression = gr.Slider(0, 100, value=s["compression"], step=1, label="IMAGE COMPRESSION",
                                                info="PNG compression is lossless (higher = smaller, slower). JPG and WebP compression is lossy.")
                        remember_output = gr.Checkbox(s["remember_output_folder"], label="SAVE OUTPUT FOLDER",
                                                      info="If enabled, the output folder will be remembered between sessions.")
                        overwrite = gr.Checkbox(s["overwrite"], label="OVERWRITE PREVIOUS UPSCALE",
                                                info="If enabled, the image is processed again instead of loading the existing result.")
                    with gr.Column(elem_classes="ds-card"):
                        gr.Markdown("PROCESSING", elem_classes="ds-step")
                        gpu_id = gr.Textbox(s["gpu_id"], label="GPU ID", max_lines=1, placeholder="auto",
                                            info="Index of the GPU to use (blank = default). See System Info.")
                        tile_size = gr.Number(s["tile_size"], label="CUSTOM TILE SIZE", precision=0, minimum=0,
                                              info="0 = Auto. Smaller tiles use less VRAM; larger tiles are faster.")
                        precision = gr.Radio([("Auto", "auto"), ("FP16", "fp16"), ("BF16", "bf16"), ("FP32", "fp32")],
                                             value=s["precision"], label="PRECISION",
                                             info="Auto uses half precision where the model supports it.")
                        tta_mode = gr.Checkbox(s["tta_mode"], label="TTA MODE",
                                               info="Test Time Augmentation for better results (fewer artifacts) BUT ~8x the processing time!")
                        idle_timeout = gr.Number(s["idle_timeout_min"], label="FREE VRAM AFTER (MINUTES IDLE)", precision=0,
                                                 minimum=0, info="Stops the model worker when idle. 0 = keep it loaded.")
                        gr.Markdown("ADD CUSTOM MODELS", elem_classes="ds-step")
                        gr.Markdown("Import NCNN (.param + .bin) or PyTorch (.pth / .safetensors) models from a folder. "
                                    "More in the Model Manager tab.", elem_classes="ds-sub")
                        custom_models_folder = gr.Textbox(label="Custom models folder", max_lines=1)
                        with gr.Row():
                            pick_custom_btn = gr.Button("Select Folder", size="sm")
                            import_custom_btn = gr.Button("Import", size="sm", variant="primary")
                    with gr.Column(elem_classes="ds-card"):
                        gr.Markdown("APP", elem_classes="ds-step")
                        gr.Button("🙏 GET HELP", link="https://github.com/ThomasEricB/deblur-studio-for-pinokio#readme",
                                  link_target="_blank", variant="primary", size="sm")
                        theme_radio = gr.Radio(["system", "light", "dark"], value=s["theme"], label="THEME")
                        turn_off_notifications = gr.Checkbox(s["turn_off_notifications"], label="TURN OFF NOTIFICATIONS",
                                                             info="If enabled, no notifications are shown on success.")
                        logs_box = gr.Textbox(log_text(), label="Logs", lines=10, max_lines=10, interactive=False,
                                              buttons=["copy"], elem_id="ds-logs")
                        refresh_logs_btn = gr.Button("Refresh logs", size="sm")
                        reset_settings_btn = gr.Button("RESET SETTINGS", variant="stop", size="sm")
                        gr.JSON(sysinfo.summary(), label="System Info")


            # -------------------------- Model Manager --------------------------
            with gr.Tab("🧩 Model Manager", id="models"):
                gr.Markdown("### Model Manager\nDownload, delete and import upscaling / deblurring models. "
                            "Each model runs on a backend environment (see **Environments**). ⭐ = recommended.")
                models_summary_md = gr.Markdown(models_summary())
                with gr.Row():
                    f_backend = gr.Dropdown(["All"] + [b.id for b in registry.backends()], value="All", label="Backend")
                    f_task = gr.Dropdown(["All", *registry.TASK_LABELS.values()], value="All", label="Task")
                    f_status = gr.Dropdown(["All", "Installed", "Not installed"], value="All", label="Status")
                models_df = gr.Dataframe(
                    models_table("All", "All", "All"),
                    headers=["", "Model", "Task", "Scale", "Backend", "Size", "License", "Description", "Key"],
                    interactive=False, wrap=True, max_height=460, show_search="filter",
                    column_widths=["4%", "18%", "8%", "6%", "8%", "8%", "14%", "26%", "8%"],
                )
                model_sel = gr.Dropdown(model_select_choices(), multiselect=True, label="Selected models",
                                        info="Click rows in the table or pick here")
                with gr.Row():
                    auto_env = gr.Checkbox(True, label="Also install the backend environment if it is missing")
                    torch_choice_mm = gr.Dropdown(
                        [("Auto-detect", "auto")] + [(v["label"], k) for k, v in environments.TORCH_VARIANTS.items()],
                        value=s["torch_variant"], label="PyTorch build (for PyTorch backends)")
                with gr.Row():
                    download_sel_btn = gr.Button("⬇ Download selected", variant="primary")
                    recommended_btn = gr.Button("⭐ Select recommended")
                    delete_sel_btn = gr.Button("🗑 Delete selected", variant="stop")
                    confirm_delete_btn = gr.Button("Click again to confirm delete", variant="stop", visible=False)
                mm_log = gr.Textbox(label="Activity", lines=8, autoscroll=True)
                gr.Markdown("#### Import custom models")
                custom_upload = gr.File(file_count="multiple", file_types=[".pth", ".safetensors", ".ckpt", ".pt", ".param", ".bin"],
                                        label="NCNN .param + .bin pairs, or PyTorch .pth / .safetensors")
                import_upload_btn = gr.Button("Import uploaded files")

            # -------------------------- Environments --------------------------
            with gr.Tab("🧰 Environments", id="envs"):
                gr.Markdown(
                    "### Backend environments\nEvery backend has its own isolated Python environment "
                    "(`plugins/<id>/.venv`), built with **uv** from pinned requirements. The UI never imports "
                    "model code - it launches one headless worker at a time with that environment's interpreter, "
                    "so incompatible stacks never collide and stopping a worker returns all of its VRAM."
                )
                env_df = gr.Dataframe(env_rows(), headers=["Backend", "ID", "State", "Python", "Torch build", "Venv size",
                                                           "Models", "Details"], interactive=False, wrap=True)
                with gr.Row():
                    env_backend = gr.Dropdown([(b.name, b.id) for b in registry.backends()],
                                              value=registry.backends()[0].id if registry.backends() else None,
                                              label="Backend")
                    env_torch = gr.Dropdown(
                        [("Auto-detect", "auto")] + [(v["label"], k) for k, v in environments.TORCH_VARIANTS.items()],
                        value=s["torch_variant"], label=f"PyTorch build (detected: {environments.detect_torch_variant()})")
                with gr.Row():
                    env_install_btn = gr.Button("Install", variant="primary")
                    env_verify_btn = gr.Button("Verify / self-test")
                    env_update_btn = gr.Button("Update")
                    env_repair_btn = gr.Button("Repair (recreate)")
                    env_remove_btn = gr.Button("Remove", variant="stop")
                    env_confirm_remove_btn = gr.Button("Click again to remove", variant="stop", visible=False)
                env_log = gr.Textbox(label="Install log", lines=18, max_lines=18, autoscroll=True, buttons=["copy"])
                for err in registry.manifest_errors():
                    gr.Markdown(f"⚠️ Plugin manifest error: `{err}`")

            # -------------------------- Jobs --------------------------
            with gr.Tab("🗂️ Jobs", id="jobs"):
                gr.Markdown("### Job history")
                jobs_df = gr.Dataframe(jobs_rows(), headers=["ID", "Created", "Mode", "Model", "Status", "Size",
                                                             "Time", "Output", "Error"],
                                       interactive=False, wrap=True, max_height=520)
                with gr.Row():
                    refresh_jobs_btn = gr.Button("Refresh")
                    cleanup_days = gr.Number(7, label="Delete job scratch files older than (days)", precision=1, minimum=0)
                    cleanup_btn = gr.Button("Clean up scratch files")
                    clear_history_btn = gr.Button("Clear history", variant="stop")
                jobs_msg = gr.Markdown("")

            # -------------------------- Diagnostics --------------------------
            with gr.Tab("🩺 Diagnostics", id="diag"):
                with gr.Row():
                    diag_refresh_btn = gr.Button("Refresh")
                    stop_worker_btn = gr.Button("Stop worker (free VRAM)")
                    restart_worker_btn = gr.Button("Hard restart worker", variant="stop")
                diag_msg = gr.Markdown("")
                health_json = gr.JSON(label="Worker health / metadata")
                worker_log = gr.Textbox(label="Worker log", lines=16, max_lines=16, autoscroll=True, buttons=["copy"])
                gpu_box = gr.Code(label="nvidia-smi", language=None)

        status_timer = gr.Timer(2.0)

        # =================================================================
        # EVENTS
        # =================================================================
        def persist(key):
            def _save(value):
                settings.set(key, value)
            return _save

        for comp, key in [
            (save_as, "save_image_as"), (copy_metadata, "copy_metadata"), (compression, "compression"),
            (overwrite, "overwrite"), (turn_off_notifications, "turn_off_notifications"),
            (gpu_id, "gpu_id"), (precision, "precision"), (tta_mode, "tta_mode"), (double_upscayl, "double_upscayl"),
            (model_dd, "selected_model"), (restore_dd, "restore_model"), (view_type, "view_type"),
            (zoom_amount, "zoom_amount"), (theme_radio, "theme"),
        ]:
            comp.change(persist(key), comp, None, show_progress="hidden", queue=False)

        tile_size.change(lambda v: settings.set("tile_size", int(v or 0)), tile_size, None, queue=False)
        custom_width.change(lambda v: settings.set("custom_width", int(v or 0)), custom_width, None, queue=False)
        torch_choice_mm.change(persist("torch_variant"), torch_choice_mm, None, queue=False)
        env_torch.change(persist("torch_variant"), env_torch, None, queue=False)

        def on_idle_timeout(v):
            v = int(v or 0)
            supervisor.idle_timeout_min = v
            settings.set("idle_timeout_min", v)

        idle_timeout.change(on_idle_timeout, idle_timeout, None, queue=False)

        def on_remember(v, folder):
            settings.save({"remember_output_folder": v, "saved_output_path": folder})

        remember_output.change(on_remember, [remember_output, output_folder], None, queue=False)

        def on_output_folder(folder, remember):
            if remember:
                settings.set("saved_output_path", folder)

        output_folder.change(on_output_folder, [output_folder, remember_output], None, queue=False)

        # theme (client side)
        theme_js = """(t) => {
            const dark = t === 'dark' || (t === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
            document.body.classList.toggle('dark', dark); return t; }"""
        theme_radio.change(None, theme_radio, None, js=theme_js)
        demo.load(None, theme_radio, None, js=theme_js)

        # scale sliders + resolution line (two sliders kept in sync like Upscayl's shared atom)
        def on_scale(v, dims, double, use_cw, cw):
            v = int(v)
            settings.set("scale", v)
            label = _scale_label(v, use_cw)
            return (gr.update(value=v, label=label), gr.update(value=v, label=label),
                    _resolution_text(dims, v, double, use_cw, cw or 0))

        res_inputs = [dims_state, double_upscayl, use_custom_width, custom_width]
        scale_step.input(on_scale, [scale_step, *res_inputs], [scale_step, scale_settings, resolution_md], queue=False)
        scale_settings.input(on_scale, [scale_settings, *res_inputs], [scale_step, scale_settings, resolution_md], queue=False)

        def on_custom_width_toggle(use_cw, scale, dims, double, cw):
            settings.set("use_custom_width", use_cw)
            label = _scale_label(scale, use_cw)
            return (gr.update(interactive=use_cw), gr.update(interactive=not use_cw, label=label),
                    gr.update(interactive=not use_cw, label=label), _resolution_text(dims, scale, double, use_cw, cw or 0))

        use_custom_width.change(on_custom_width_toggle, [use_custom_width, scale_step, dims_state, double_upscayl, custom_width],
                                [custom_width, scale_step, scale_settings, resolution_md], queue=False)
        for c in (double_upscayl, custom_width):
            c.change(lambda dims, sc, d, u, w: _resolution_text(dims, sc, d, u, w or 0),
                     [dims_state, scale_step, double_upscayl, use_custom_width, custom_width], resolution_md, queue=False)

        def on_model_change(key, dims, double, use_cw, cw):
            # Convenience: snap the scale to the model's native scale (deblur models are 1x).
            try:
                p = registry.profile(key)
            except KeyError:
                return gr.update(), gr.update(), gr.update()
            sc = p.entry.scale
            label = _scale_label(sc, use_cw)
            settings.set("scale", sc)
            return gr.update(value=sc, label=label), gr.update(value=sc, label=label), _resolution_text(dims, sc, double, use_cw, cw or 0)

        model_dd.input(on_model_change, [model_dd, *res_inputs], [scale_step, scale_settings, resolution_md], queue=False)

        # batch toggle
        def on_batch(batch, remember):
            out = gr.update() if remember else str(DEFAULT_OUTPUT_DIR)
            return (gr.update(visible=not batch), gr.update(visible=batch), gr.update(visible=not batch),
                    gr.update(visible=not batch), gr.update(visible=batch, value=instructions_md(True) if batch else ""),
                    out, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))

        batch_mode.change(on_batch, [batch_mode, remember_output],
                          [select_image_btn, folder_group, double_upscayl, image_in, batch_info, output_folder,
                           compare, result_row, batch_gallery], queue=False)

        # Step 1: image selection (button or drop/paste on the main area)
        def on_image_uploaded(path):
            logit("🖼 Selected Image Path:", path)
            return path

        select_image_btn.upload(on_image_uploaded, select_image_btn, image_in, queue=False)

        def on_image_change(path, scale, double, use_cw, cw):
            if not path:
                return {}, "", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            try:
                with Image.open(path) as im:
                    dims = {"w": im.width, "h": im.height}
            except Exception:
                gr.Warning("Invalid Image. Please select an image with a valid extension like PNG, JPG, JPEG, JFIF or WEBP.")
                return {}, "", gr.update(), gr.update(), gr.update()
            return (dims, _resolution_text(dims, scale, double, use_cw, cw or 0), gr.update(visible=False),
                    gr.update(visible=True), gr.update(visible=False))

        image_in.change(on_image_change, [image_in, scale_step, double_upscayl, use_custom_width, custom_width],
                        [dims_state, resolution_md, compare, image_in, result_row], queue=False)

        def on_pick_folder(current):
            path = native.pick("folder", "Select a folder of images", current or None)
            if not path:
                gr.Info("No folder selected - you can also paste a path into the box.")
                return gr.update(), gr.update()
            try:
                n = len(collect_folder(path))
            except ValueError as exc:
                gr.Warning(str(exc))
                return gr.update(), gr.update()
            return path, gr.update(visible=True, value=f"**Selected folder:** `{path}` ({n} images)")

        select_folder_btn.click(on_pick_folder, folder_path, [folder_path, batch_info])

        def on_folder_typed(path):
            if not path:
                return gr.update(value=instructions_md(True))
            try:
                n = len(collect_folder(path))
                return gr.update(value=f"**Selected folder:** `{path}` ({n} images)")
            except ValueError as exc:
                return gr.update(value=f"⚠️ {exc}")

        folder_path.submit(on_folder_typed, folder_path, batch_info, queue=False)
        folder_path.blur(on_folder_typed, folder_path, batch_info, queue=False)

        def on_folder_uploaded(files):
            files = [f for f in (files or []) if Path(f).suffix.lower().lstrip(".") in
                     {"png", "jpg", "jpeg", "jfif", "webp", "bmp", "tif", "tiff"}]
            return files, "", gr.update(visible=True, value=f"**Uploaded folder:** {len(files)} images")

        upload_folder_btn.upload(on_folder_uploaded, upload_folder_btn, [batch_files_state, folder_path, batch_info])

        # Step 2: model dialog
        browse_models_btn.click(lambda: (gr.update(visible=True), gr.update(value=_preview_gallery())), None,
                                [model_dialog, model_gallery], queue=False)
        close_dialog_btn.click(lambda: gr.update(visible=False), None, model_dialog, queue=False)

        def on_gallery_select(evt: gr.SelectData):
            keys = _gallery_keys()
            if evt.index is None or evt.index >= len(keys):
                return gr.update(), "", gr.update(visible=False), gr.update()
            key = keys[evt.index]
            p = registry.profile(key)
            if p.installed:
                settings.set("selected_model", key)
                choices = model_choices()
                return (gr.update(choices=choices, value=key), key, gr.update(visible=False), gr.update(visible=False))
            return gr.update(), key, gr.update(visible=True, value=f"⬇ Download {p.label}"), gr.update()

        model_gallery.select(on_gallery_select, None, [model_dd, gallery_pick, gallery_download_btn, model_dialog])

        # Step 3: output folder
        def on_set_output(current):
            path = native.pick("folder", "Select output folder", current or None)
            if not path:
                gr.Info("No folder selected - you can also type a path into the Output folder box.")
                return gr.update()
            logit("🗂 Setting Output Path:", path)
            return path

        set_output_btn.click(on_set_output, output_folder, output_folder)

        def on_open_folder(path):
            if not native.open_folder(path):
                gr.Warning(f"Could not open {path}")

        open_output_btn.click(on_open_folder, output_folder, None, queue=False)

        # ---------------- Step 4: Upscayl ----------------
        def build_request(model, restore, double, scale, out_dir) -> UpscaleRequest:
            st = settings.load()
            return UpscaleRequest(
                model=model, restore_model=restore or "", double=bool(double), scale=int(scale),
                use_custom_width=bool(st["use_custom_width"]), custom_width=int(st["custom_width"] or 0),
                save_as=st["save_image_as"], compression=int(st["compression"]), tile_size=int(st["tile_size"] or 0),
                tta=bool(st["tta_mode"]), gpu_id=str(st["gpu_id"] or ""), precision=st["precision"],
                copy_metadata=bool(st["copy_metadata"]), overwrite=bool(st["overwrite"]),
                output_dir=out_dir or str(DEFAULT_OUTPUT_DIR),
            )

        upscale_outputs = [progress_box, stop_btn, stop_btn_side, upscayl_btn, image_in, compare, result_row,
                           result_md, download_btn, batch_gallery, stats_box, last_result_state]

        def busy_frame(msg, batch_count=""):
            return [gr.update(visible=True, value=progress_html(msg, batch_count)), gr.update(visible=True),
                    gr.update(visible=True), gr.update(value="Upscaling ⏳", interactive=False),
                    gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(), gr.update(),
                    gr.update(visible=False), gr.update(), gr.update()]

        def idle_frame(**overrides):
            frame = dict(progress=gr.update(visible=False), stop=gr.update(visible=False),
                         stop_side=gr.update(visible=False), go=gr.update(value="Upscale 🚀", interactive=True),
                         image=gr.update(), compare=gr.update(), row=gr.update(), md=gr.update(), dl=gr.update(),
                         gallery=gr.update(), stats=gr.update(value=stats_html()), last=gr.update())
            frame.update(overrides)
            return list(frame.values())

        def do_upscayl(image_path, batch, folder, batch_files, model, restore, double, scale, out_dir,
                       view, zoom, request: gr.Request, progress=gr.Progress()):
            if not model:
                gr.Warning("No model installed yet. Open the Model Manager tab (or use Quick setup) to download one.")
                yield idle_frame()
                return
            if not out_dir:
                gr.Warning("Please select an output folder first")
                yield idle_frame()
                return
            files: list[Path] = []
            folder_name = ""
            if batch:
                try:
                    if folder:
                        files = collect_folder(folder)
                        folder_name = Path(folder).name
                    elif batch_files:
                        files = [Path(f) for f in batch_files]
                        folder_name = "upload"
                except ValueError as exc:
                    gr.Warning(str(exc))
                    yield idle_frame()
                    return
                if not files:
                    gr.Warning("No images found. Please select a folder of images first.")
                    yield idle_frame()
                    return
            elif not image_path:
                gr.Warning("No image selected. Please select an image to upscale.")
                yield idle_frame()
                return

            req = build_request(model, restore, double, scale, out_dir)
            cancel = threading.Event()
            _cancel[_session(request)] = cancel
            state = {"frac": 0.0, "msg": "Hold on...", "result": None, "error": None, "done": False}

            def report(frac, msg):
                state["frac"], state["msg"] = frac, msg

            def work():
                try:
                    if batch:
                        state["result"] = run_batch(files, folder_name, req, report, cancel)
                    else:
                        state["result"] = run_image(image_path, req, report, cancel)
                except Exception as exc:
                    state["error"] = exc
                finally:
                    state["done"] = True

            logit("🏁", "BATCH_UPSCALE" if batch else ("DOUBLE_UPSCALE" if double else "UPSCALE"), model,
                  f"restore={restore or '-'}", json.dumps({k: v for k, v in req.__dict__.items() if k != 'output_dir'}))
            thread = threading.Thread(target=work, daemon=True)
            thread.start()
            yield busy_frame("Hold on...")
            last = None
            while not state["done"]:
                time.sleep(0.2)
                msg = state["msg"]
                progress(state["frac"], desc=msg)
                if msg != last:
                    last = msg
                    yield busy_frame(msg)
            _cancel.pop(_session(request), None)

            err = state["error"]
            if isinstance(err, Cancelled):
                logit("🛑 Stopped")
                gr.Info("Upscaling stopped.")
                yield idle_frame()
                return
            if err is not None:
                logit("❌", f"{type(err).__name__}: {err}")
                gr.Warning(f"Error: {err}")
                yield idle_frame()
                return

            if batch:
                out_dir_path, results, errors = state["result"]
                for e in errors:
                    logit("⚠️", e)
                if errors:
                    gr.Warning(f"{len(errors)} image(s) failed - see Settings > Logs.")
                msg = f"**All done!** {len(results)} image(s) saved to `{out_dir_path}`"
                _toast_ok("Batch upscale finished!")
                if not settings.load()["turn_off_notifications"]:
                    native.notify(APP_NAME, "Folder upscaled successfully!")
                yield idle_frame(row=gr.update(visible=True), md=msg,
                                 dl=gr.update(visible=False),
                                 gallery=gr.update(visible=True, value=[r.display_output for r in results][:200]),
                                 last={"folder": str(out_dir_path)})
                return

            r = state["result"]
            for n in r.notes:
                gr.Info(n)
            logit("💯 UPSCALE_DONE:", r.saved_path)
            if not r.reused:
                _toast_ok("Image upscaled successfully!")
                if not settings.load()["turn_off_notifications"]:
                    native.notify(APP_NAME, "Image upscaled successfully!")
            info = (f"**{Path(r.saved_path).name}** · {r.in_size[0]}x{r.in_size[1]} → {r.out_size[0]}x{r.out_size[1]}"
                    + (f" · {_fmt_duration(r.seconds)}" if r.seconds else ""))
            yield idle_frame(
                image=gr.update(visible=False),
                compare=gr.update(visible=True, value=viewer.value(r.display_input, r.display_output, view, zoom)),
                row=gr.update(visible=True), md=info,
                dl=gr.update(visible=True, value=r.display_output),
                last={"input": r.display_input, "output": r.display_output, "saved": r.saved_path},
            )

        upscale_inputs = [image_in, batch_mode, folder_path, batch_files_state, model_dd, restore_dd, double_upscayl,
                          scale_step, output_folder, view_type, zoom_amount]
        upscayl_btn.click(do_upscayl, upscale_inputs, upscale_outputs,
                                          concurrency_id="gpu", concurrency_limit=1, show_progress="minimal")

        def do_stop(request: gr.Request):
            ev = _cancel.get(_session(request))
            if ev:
                ev.set()
                logit("🛑 Stopping")

        stop_btn.click(do_stop, None, None, queue=False)
        stop_btn_side.click(do_stop, None, None, queue=False)

        def do_open_result(last, out_dir):
            target = (last or {}).get("folder") or (last or {}).get("saved") or out_dir
            if not native.open_folder(target):
                gr.Warning(f"Could not open {target}")

        open_result_btn.click(do_open_result, [last_result_state, output_folder], None, queue=False)

        # drawer
        def on_view(view, zoom, last):
            upd = gr.update()
            if last and last.get("output"):
                upd = gr.update(value=viewer.value(last["input"], last["output"], view, zoom))
            return gr.update(visible=view == "slider"), upd

        view_type.change(on_view, [view_type, zoom_amount, last_result_state], [zoom_amount, compare], queue=False)
        zoom_amount.release(lambda v, z, last: on_view(v, z, last)[1], [view_type, zoom_amount, last_result_state],
                            compare, queue=False)

        def reset_image():
            logit("🔄 Resetting image paths")
            return (gr.update(value=None, visible=True), {}, "", gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=False), {}, [], "")

        reset_image_btn.click(reset_image, None,
                              [image_in, dims_state, resolution_md, compare, result_row, batch_gallery,
                               last_result_state, batch_files_state, folder_path], queue=False)

        # onboarding / quick setup
        def skip_onboarding():
            settings.set("onboarded", True)
            return gr.update(visible=False)

        skip_onboarding_btn.click(skip_onboarding, None, onboarding, queue=False)

        def quick_setup(progress=gr.Progress()):
            b = registry.backend("ncnn-vulkan")
            text = ""
            if not environments.is_ready(b):
                try:
                    for text in environments.run_operation("create", b.id, stop_worker=supervisor.stop_backend):
                        yield gr.update(visible=True, value=text), gr.update(), gr.update()
                except Exception as exc:
                    gr.Warning(f"Environment install failed: {exc}")
                    return
            p = registry.profile("ncnn-vulkan/upscayl-standard-4x")
            if not p.installed:
                lines = text + "\nDownloading Upscayl Standard…"

                def prog(frac, msg):
                    progress(frac, desc=msg)
                try:
                    downloads.download(p, prog)
                except Exception as exc:
                    gr.Warning(f"Download failed: {exc}")
                    return
                text = lines + " done."
            settings.save({"onboarded": True, "selected_model": p.key})
            gr.Info("You're all set! 🎉")
            yield (gr.update(visible=True, value=text + "\n✅ Ready - select an image and click Upscale 🚀"),
                   gr.update(choices=model_choices(), value=p.key), gr.update(visible=False))

        quick_setup_btn.click(quick_setup, None, [quick_log, model_dd, onboarding], concurrency_id="installer",
                              concurrency_limit=1)

        # settings misc
        refresh_logs_btn.click(log_text, None, logs_box, queue=False)

        def do_reset_settings():
            settings.reset()
            gr.Info("Settings have been reset. Reloading…")

        reset_settings_btn.click(do_reset_settings, None, None, queue=False).then(
            None, None, None, js="() => { setTimeout(() => window.location.reload(), 600); }")

        def pick_custom(current):
            return native.pick("folder", "Select custom models folder", current or None) or gr.update()

        pick_custom_btn.click(pick_custom, custom_models_folder, custom_models_folder)

        def import_custom_folder(folder):
            if not folder or not Path(folder).is_dir():
                gr.Warning("Select a folder that contains models first.")
                return gr.update(), gr.update()
            files = [str(p) for p in Path(folder).iterdir()
                     if p.suffix.lower() in (".param", ".bin", ".pth", ".safetensors", ".ckpt", ".pt")]
            results = downloads.import_files(files)
            gr.Info("\n".join(results) or "No models found in that folder.")
            return gr.update(choices=model_choices()), gr.update(choices=restore_choices())

        import_custom_btn.click(import_custom_folder, custom_models_folder, [model_dd, restore_dd])

        # ---------------- Model Manager ----------------
        mm_filters = [f_backend, f_task, f_status]

        def refresh_models(fb, ft, fs, selected):
            return (models_table(fb, ft, fs), gr.update(choices=model_select_choices(), value=selected),
                    models_summary(), gr.update(choices=model_choices()), gr.update(choices=restore_choices()))

        mm_refresh_outputs = [models_df, model_sel, models_summary_md, model_dd, restore_dd]
        for f in mm_filters:
            f.change(lambda fb, ft, fs: models_table(fb, ft, fs), mm_filters, models_df, queue=False)

        def on_row_select(evt: gr.SelectData, table, selected):
            try:
                row = evt.index[0]
                key = table.iloc[row, -1] if hasattr(table, "iloc") else table[row][-1]
            except Exception:
                return gr.update()
            selected = list(selected or [])
            if key in selected:
                selected.remove(key)
            else:
                selected.append(key)
            return gr.update(value=selected)

        models_df.select(on_row_select, [models_df, model_sel], model_sel, queue=False)

        def select_recommended():
            return gr.update(value=[p.key for p in registry.profiles() if p.entry.recommended])

        recommended_btn.click(select_recommended, None, model_sel, queue=False)

        def do_download(keys, auto, torch_choice, fb, ft, fs, progress=gr.Progress()):
            keys = list(keys or [])
            if not keys:
                gr.Warning("Select one or more models first.")
                yield ("", *refresh_models(fb, ft, fs, keys))
                return
            log: list[str] = []
            needed_envs = []
            for k in keys:
                b = registry.profile(k).backend
                if not environments.is_ready(b) and b.id not in needed_envs:
                    needed_envs.append(b.id)
            if needed_envs and auto:
                for bid in needed_envs:
                    log.append(f"Installing environment for {registry.backend(bid).name}…")
                    try:
                        for text in environments.run_operation("create", bid, torch_choice, supervisor.stop_backend):
                            yield ("\n".join(log) + "\n" + text[-3000:], *[gr.update()] * 5)
                        log.append("✅ environment ready")
                    except Exception as exc:
                        log.append(f"❌ environment install failed: {exc}")
            elif needed_envs:
                log.append("ℹ️ Some backends are not installed yet; install them from the Environments tab before use.")
            for i, k in enumerate(keys):
                p = registry.profile(k)
                if p.installed:
                    log.append(f"✔ {p.label} already downloaded")
                    continue
                if p.custom:
                    continue
                log.append(f"⬇ {p.label} ({downloads.human(p.size_bytes)})")
                yield ("\n".join(log), *[gr.update()] * 5)

                def prog(frac, msg, i=i):
                    progress((i + frac) / len(keys), desc=msg)
                try:
                    downloads.download(p, prog)
                    log.append(f"✅ {p.label}")
                    logit("⬇ Downloaded", p.key)
                except Exception as exc:
                    log.append(f"❌ {p.label}: {exc}")
                yield ("\n".join(log), *[gr.update()] * 5)
            _toast_ok("Downloads finished")
            yield ("\n".join(log), *refresh_models(fb, ft, fs, keys))

        download_sel_btn.click(do_download, [model_sel, auto_env, torch_choice_mm, *mm_filters],
                               [mm_log, *mm_refresh_outputs], concurrency_id="downloads", concurrency_limit=1)

        delete_sel_btn.click(lambda keys: gr.update(visible=bool(keys)), model_sel, confirm_delete_btn, queue=False)

        def do_delete(keys, fb, ft, fs):
            log = []
            for k in keys or []:
                p = registry.profile(k)
                supervisor.unload_if_loaded(k)
                downloads.delete(p)
                log.append(f"🗑 deleted {p.label}")
                logit("🗑 Deleted", k)
            return ("\n".join(log), gr.update(visible=False), *refresh_models(fb, ft, fs, []))

        confirm_delete_btn.click(do_delete, [model_sel, *mm_filters], [mm_log, confirm_delete_btn, *mm_refresh_outputs])

        def do_import_upload(files, fb, ft, fs):
            results = downloads.import_files([str(f) for f in (files or [])])
            return ("\n".join(results) or "Nothing to import.", None, *refresh_models(fb, ft, fs, []))

        import_upload_btn.click(do_import_upload, [custom_upload, *mm_filters],
                                [mm_log, custom_upload, *mm_refresh_outputs])

        # gallery "download this model"
        def gallery_download(key, progress=gr.Progress()):
            p = registry.profile(key)
            if not environments.is_ready(p.backend):
                try:
                    for _ in environments.run_operation("create", p.backend.id, None, supervisor.stop_backend):
                        progress(0.0, desc=f"Installing {p.backend.name} environment…")
                except Exception as exc:
                    gr.Warning(f"Environment install failed: {exc}")
                    return gr.update(), gr.update(), gr.update()
            try:
                downloads.download(p, lambda f, m: progress(f, desc=m))
            except Exception as exc:
                gr.Warning(f"Download failed: {exc}")
                return gr.update(), gr.update(), gr.update()
            settings.set("selected_model", key)
            _toast_ok(f"{p.label} downloaded")
            return gr.update(choices=model_choices(), value=key), gr.update(visible=False), gr.update(value=_preview_gallery())

        gallery_download_btn.click(gallery_download, gallery_pick, [model_dd, gallery_download_btn, model_gallery],
                                   concurrency_id="downloads", concurrency_limit=1)

        # ---------------- Environments ----------------
        def env_op(op):
            def run(backend_id, torch_choice):
                if not backend_id:
                    yield "Select a backend.", env_rows(), gr.update()
                    return
                text = ""
                try:
                    for text in environments.run_operation(op, backend_id, torch_choice, supervisor.stop_backend):
                        yield text, gr.update(), gr.update()
                    _toast_ok(f"{op.capitalize()} finished")
                except Exception as exc:
                    gr.Warning(f"{op.capitalize()} failed: {exc}")
                logit(f"🧰 env {op} {backend_id}")
                yield text, env_rows(), gr.update(visible=False)
            return run

        env_outputs = [env_log, env_df, env_confirm_remove_btn]
        for btn, op in [(env_install_btn, "create"), (env_verify_btn, "verify"), (env_update_btn, "update"),
                        (env_repair_btn, "repair")]:
            btn.click(env_op(op), [env_backend, env_torch], env_outputs, concurrency_id="installer", concurrency_limit=1)
        env_remove_btn.click(lambda: gr.update(visible=True), None, env_confirm_remove_btn, queue=False)
        env_confirm_remove_btn.click(env_op("remove"), [env_backend, env_torch], env_outputs,
                                     concurrency_id="installer", concurrency_limit=1)

        # ---------------- Jobs ----------------
        refresh_jobs_btn.click(jobs_rows, None, jobs_df, queue=False)

        def do_cleanup(days):
            n, freed = jobs.cleanup(float(days or 0))
            return f"Removed {n} job folder(s), freed {downloads.human(freed)}. Saved outputs are untouched.", jobs_rows()

        cleanup_btn.click(do_cleanup, cleanup_days, [jobs_msg, jobs_df])

        def do_clear_history():
            jobs.clear_history()
            return "History cleared.", jobs_rows(), stats_html()

        clear_history_btn.click(do_clear_history, None, [jobs_msg, jobs_df, stats_box])

        # ---------------- Diagnostics ----------------
        def diag_refresh():
            return (supervisor.health(), "\n".join(list(supervisor.log_tail)[-400:]) or "(no worker output yet)",
                    sysinfo.nvidia_smi_snapshot())

        diag_refresh_btn.click(diag_refresh, None, [health_json, worker_log, gpu_box])

        def do_stop_worker():
            supervisor.stop("(user request)")
            return "Worker stopped - VRAM released.", *diag_refresh()

        stop_worker_btn.click(do_stop_worker, None, [diag_msg, health_json, worker_log, gpu_box])

        def do_restart():
            try:
                msg = supervisor.hard_restart()
            except WorkerError as exc:
                msg = f"❌ {exc}"
            return msg, *diag_refresh()

        restart_worker_btn.click(do_restart, None, [diag_msg, health_json, worker_log, gpu_box],
                                 concurrency_id="gpu", concurrency_limit=1)

        def on_tab(evt: gr.SelectData):
            # The More-options drawer belongs to the Upscayl tab, like the left panel.
            studio = evt.index == 0
            return env_rows(), jobs_rows(), gr.update(visible=studio)

        main_tabs.select(on_tab, None, [env_df, jobs_df, drawer], queue=False)

        # Upscayl's sidebar show/hide toggle
        collapse_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None,
                           [sidebar_col, expand_btn], queue=False)
        expand_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)), None,
                         [sidebar_col, expand_btn], queue=False)

        # status strip
        status_timer.tick(status_strip, None, status_box, show_progress="hidden", queue=False)

        # ---------------- Public API (see README) ----------------
        with gr.Row(visible=False):
            api_image = gr.Image(type="filepath")
            api_model = gr.Textbox()
            api_scale = gr.Number(precision=0)
            api_restore = gr.Textbox()
            api_double = gr.Checkbox()
            api_result = gr.Image(type="filepath")
            api_models_out = gr.JSON()
            api_btn = gr.Button()
            api_models_btn = gr.Button()

        def api_upscale(image, model, scale, restore_model, double):
            """Upscale one image. `model` / `restore_model` are keys from /models.
            Format, compression, tile size, TTA and GPU come from the saved Settings."""
            if not image:
                raise gr.Error("image is required")
            st = settings.load()
            key = model or st["selected_model"]
            try:
                p = registry.profile(key)
            except KeyError:
                raise gr.Error(f"unknown model {key!r}; call /models") from None
            req = build_request(key, restore_model or "", bool(double), int(scale or p.entry.scale),
                                st["saved_output_path"] if st["remember_output_folder"] else str(DEFAULT_OUTPUT_DIR))
            req.overwrite = True
            try:
                result = run_image(image, req, lambda f, m: None, threading.Event())
            except (WorkerError, ValueError, Cancelled) as exc:
                raise gr.Error(str(exc)) from exc
            return result.display_output

        api_btn.click(api_upscale, [api_image, api_model, api_scale, api_restore, api_double], api_result,
                      api_name="upscale", concurrency_id="gpu", concurrency_limit=1)

        def api_models():
            """List models: key, name, task, native scale, backend, whether downloaded."""
            return [{"key": p.key, "name": p.label, "task": p.entry.task, "scale": p.entry.scale,
                     "backend": p.backend.id, "installed": p.installed} for p in registry.profiles()]

        api_models_btn.click(api_models, None, api_models_out, api_name="models", queue=False)

    # Only the two endpoints above are documented; UI plumbing stays out of the API page.
    for fn in demo.fns.values():
        if fn.api_name not in ("upscale", "models"):
            fn.api_visibility = "undocumented"
    return demo
