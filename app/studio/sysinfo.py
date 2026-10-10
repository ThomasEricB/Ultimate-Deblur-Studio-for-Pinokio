"""System information for the Settings > System Info panel and Diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6:
            rows.append({"index": parts[0], "name": parts[1], "mem_used_mb": parts[2], "mem_total_mb": parts[3],
                         "util_pct": parts[4], "driver": parts[5]})
    return rows


def nvidia_smi_snapshot() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "nvidia-smi not found (non-NVIDIA system or driver not installed)."
    try:
        return subprocess.run([exe], capture_output=True, text=True, timeout=10).stdout
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


def summary() -> dict:
    info = {
        "platform": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "master_python": sys.version.split()[0],
    }
    try:
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        info["ram_gb"] = round(pages * size / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        pass
    g = gpus()
    if g:
        info["gpus"] = [f"[{x['index']}] {x['name']} ({int(x['mem_total_mb']) // 1024} GB, driver {x['driver']})" for x in g]
    return info
