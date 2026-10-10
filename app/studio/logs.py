"""Master log: Upscayl's `logit()` - shown in Settings > Logs and written to
runtime/logs/master.log."""

from __future__ import annotations

import collections
import threading
import time

from .paths import LOGS_DIR

LOG_FILE = LOGS_DIR / "master.log"
_buffer: collections.deque[str] = collections.deque(maxlen=1000)
_lock = threading.Lock()

if LOG_FILE.exists() and LOG_FILE.stat().st_size > 5 * 1024 * 1024:
    LOG_FILE.write_text("")


def logit(*parts: object) -> None:
    line = f"{time.strftime('%H:%M:%S')} " + " ".join(str(p) for p in parts)
    with _lock:
        _buffer.append(line)
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    print(line, flush=True)


def text() -> str:
    with _lock:
        return "\n".join(_buffer) if _buffer else "No logs to show"
