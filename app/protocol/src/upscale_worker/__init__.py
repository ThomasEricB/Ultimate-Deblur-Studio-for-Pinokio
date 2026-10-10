"""Headless worker shell. Run with the plugin venv's interpreter:

    <plugin>/.venv/bin/python -m upscale_worker --backend <id> --adapter adapter:Cls \
        --plugin-dir <dir> --models-dir <dir> --spool <dir> --port <port>

It hosts a small FastAPI app on 127.0.0.1 that delegates to the plugin's
adapter. It never launches a UI.
"""

from .context import WorkerContext, anchored

__all__ = ["WorkerContext", "anchored"]
