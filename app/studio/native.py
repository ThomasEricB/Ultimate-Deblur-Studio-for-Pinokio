"""Desktop integrations that Upscayl got from Electron: native folder/file
pickers, system notifications and "open folder". Uses Pinokio's `pterm` when
available (works for the machine Pinokio runs on), falls back to tkinter in a
subprocess, and degrades to typing a path."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _pterm() -> str | None:
    exe = shutil.which("pterm")
    if exe:
        return exe
    home = Path.home() / "pinokio" / "bin" / "npm" / "bin" / "pterm"
    return str(home) if home.exists() else None


_TK_SNIPPET = r"""
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
kind, title, start = sys.argv[1], sys.argv[2], sys.argv[3] or None
if kind == 'folder':
    r = filedialog.askdirectory(title=title, initialdir=start)
else:
    r = filedialog.askopenfilename(title=title, initialdir=start,
        filetypes=[('Images', '*.png *.jpg *.jpeg *.jfif *.webp *.bmp *.tif *.tiff'), ('All files', '*')])
print(r or '')
"""


def pick(kind: str = "folder", title: str = "Select", start: str | None = None) -> str | None:
    """Show a native picker on the Pinokio machine. Returns a path or None."""
    pterm = _pterm()
    if pterm:
        argv = [pterm, "filepicker", f"--type={kind}", f"--title={title}"]
        if start and Path(start).exists():
            argv.append(f"--path={start}")
        if kind == "file":
            argv.append("--filetype=images/*.png,*.jpg,*.jpeg,*.jfif,*.webp,*.bmp,*.tif,*.tiff")
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=600).stdout
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            if lines:
                return lines[0]
            return None
        except Exception:
            pass
    try:
        out = subprocess.run(
            [sys.executable, "-c", _TK_SNIPPET, kind, title, start or ""],
            capture_output=True, text=True, timeout=600,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def notify(title: str, message: str) -> None:
    pterm = _pterm()
    if pterm:
        try:
            subprocess.Popen([pterm, "push", message, f"--title={title}"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def open_folder(path: str) -> bool:
    target = Path(path)
    if target.is_file():
        target = target.parent
    if not target.is_dir():
        return False
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
