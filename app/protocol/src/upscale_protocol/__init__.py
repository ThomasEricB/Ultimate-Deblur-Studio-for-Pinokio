"""Shared contract between the Deblur Studio master and its backend workers.

This package is installed into the master venv *and* every plugin venv, so it
must stay dependency-light: pydantic only. Never import torch/numpy/PIL here.
"""

from .models import (
    PROTOCOL_VERSION,
    Health,
    JobRequest,
    JobStatus,
    LoadRequest,
    LoadResponse,
    Metadata,
    TERMINAL_STATES,
    protocol_compatible,
)

__all__ = [
    "PROTOCOL_VERSION",
    "Health",
    "JobRequest",
    "JobStatus",
    "LoadRequest",
    "LoadResponse",
    "Metadata",
    "TERMINAL_STATES",
    "protocol_compatible",
]
