"""Job records (SQLite) and the shared job spool.

Every job gets an immutable request record. Stale RUNNING jobs are marked
INTERRUPTED when the master restarts. Upscayl's per-browser usage stats are
derived from this table instead of localStorage.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .paths import JOBS_DIR, STATE_DIR

DB_PATH = STATE_DIR / "registry.sqlite3"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    finished REAL,
    mode TEXT NOT NULL,              -- image | double | batch
    model TEXT NOT NULL,
    restore_model TEXT,
    request TEXT NOT NULL,           -- immutable JSON request
    input_name TEXT,
    status TEXT NOT NULL,            -- running | done | failed | cancelled | interrupted
    error TEXT,
    output_path TEXT,
    in_w INTEGER, in_h INTEGER, out_w INTEGER, out_h INTEGER,
    seconds REAL
);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _lock, _conn() as c:
        c.executescript(SCHEMA)
        c.execute("UPDATE jobs SET status='interrupted', error='master restarted' WHERE status='running'")


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def job_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id
    (d / "input").mkdir(parents=True, exist_ok=True)
    (d / "output").mkdir(parents=True, exist_ok=True)
    return d


def create(job_id: str, mode: str, model: str, restore_model: str | None, request: dict, input_name: str) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO jobs(id, created, mode, model, restore_model, request, input_name, status) VALUES (?,?,?,?,?,?,?,?)",
            (job_id, time.time(), mode, model, restore_model, json.dumps(request), input_name, "running"),
        )
    (job_dir(job_id) / "request.json").write_text(json.dumps(request, indent=2))


def finish(job_id: str, status: str, **fields) -> None:
    allowed = {"error", "output_path", "in_w", "in_h", "out_w", "out_h", "seconds"}
    cols = {k: v for k, v in fields.items() if k in allowed}
    sets = ", ".join(f"{k}=?" for k in cols)
    with _lock, _conn() as c:
        c.execute(
            f"UPDATE jobs SET status=?, finished=?{', ' + sets if sets else ''} WHERE id=?",
            (status, time.time(), *cols.values(), job_id),
        )
    (job_dir(job_id) / "result.json").write_text(json.dumps({"status": status, **cols}, indent=2))


def recent(limit: int = 200) -> list[dict]:
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]


def get(job_id: str) -> dict | None:
    with _lock, _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def stats() -> dict:
    """The numbers Upscayl shows in its More Options drawer."""
    with _lock, _conn() as c:
        row = c.execute(
            """SELECT
                 COUNT(*) AS total,
                 SUM(mode='batch') AS batch,
                 SUM(mode IN ('image','double')) AS image,
                 SUM(mode='double') AS double,
                 AVG(CASE WHEN status='done' THEN seconds END) AS avg_seconds,
                 MAX(created) AS last_used
               FROM jobs"""
        ).fetchone()
        last = c.execute("SELECT seconds FROM jobs WHERE status='done' ORDER BY created DESC LIMIT 1").fetchone()
    return {
        "total": row["total"] or 0,
        "batch": row["batch"] or 0,
        "image": row["image"] or 0,
        "double": row["double"] or 0,
        "avg_seconds": row["avg_seconds"] or 0.0,
        "last_seconds": (last["seconds"] if last else 0.0) or 0.0,
        "last_used": row["last_used"],
    }


def cleanup(older_than_days: float) -> tuple[int, int]:
    """Delete spool folders (not saved outputs) older than N days."""
    cutoff = time.time() - older_than_days * 86400
    removed = freed = 0
    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        try:
            if d.stat().st_mtime < cutoff:
                freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d)
                removed += 1
        except OSError:
            continue
    return removed, freed


def clear_history() -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM jobs WHERE status != 'running'")
