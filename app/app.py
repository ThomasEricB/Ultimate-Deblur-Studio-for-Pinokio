"""Deblur Studio - master process (control plane).

One Gradio UI that manages isolated backend environments. This process never
imports torch, ncnn or any model code; each backend runs as a headless worker
under its own venv interpreter (see studio/supervisor.py).

    python app.py            # launches on 127.0.0.1, first free port from 7860
"""

from __future__ import annotations

import atexit
import os
import signal
import sys

import gradio as gr

from studio import jobs
from studio.logs import logit
from studio.paths import ASSETS_DIR, JOBS_DIR, LOGS_DIR, MODELS_DIR, PLUGINS_DIR, STATE_DIR
from studio.supervisor import supervisor
from studio.ui import APP_NAME, VERSION, build
from studio.viewer import file_url


def _shutdown(*_args) -> None:
    supervisor.stop("(master exiting)")


def main() -> None:
    jobs.init()
    atexit.register(_shutdown)
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            signal.signal(sig, lambda *_: (_shutdown(), sys.exit(0)))

    assert "torch" not in sys.modules, "the master must stay CUDA-cold"
    logit(f"🚀 {APP_NAME} {VERSION} starting (python {sys.version.split()[0]}, gradio {gr.__version__})")

    fonts = file_url(str(ASSETS_DIR / "fonts")).split("?")[0]
    css = (ASSETS_DIR / "style.css").read_text().replace("__FONTS__", fonts)
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.Font("Poppins"), gr.themes.Font("ui-sans-serif"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")],
    ).set(
        body_background_fill_dark="#1d232a",
        background_fill_primary_dark="#191e24",
        background_fill_secondary_dark="#15191e",
        block_background_fill_dark="#191e24",
        button_primary_background_fill="linear-gradient(90deg, *primary_600, *secondary_500)",
        button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_500)",
        button_primary_text_color="white",
    )

    demo = build()
    demo.queue(default_concurrency_limit=2, max_size=16)
    demo.launch(
        server_name=os.environ.get("DS_HOST", "127.0.0.1"),
        share=False,
        inbrowser=False,
        theme=theme,
        css=css,
        favicon_path=str(ASSETS_DIR / "logo.png"),
        # Only the job spool (results/previews) and static assets are servable.
        allowed_paths=[str(JOBS_DIR), str(ASSETS_DIR), str(JOBS_DIR.parent / "previews")],
        blocked_paths=[str(PLUGINS_DIR), str(MODELS_DIR), str(STATE_DIR), str(LOGS_DIR)],
        max_file_size="500mb",
        footer_links=[],
    )


if __name__ == "__main__":
    main()
