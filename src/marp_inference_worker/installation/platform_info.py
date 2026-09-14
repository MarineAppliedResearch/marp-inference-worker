"""Canonical release platform identifiers."""

import platform
import sys


def current_platform() -> tuple[str, str]:
    system = "windows" if sys.platform == "win32" else platform.system().lower()
    machine = platform.machine().lower()
    architecture = "x86_64" if machine in {"amd64", "x86_64"} else machine
    return system, architecture


def installed_compute_runtime() -> str:
    """Return the runtime variant carried by this installed worker payload."""
    try:
        import torch
    except ImportError:
        return "cpu"
    cuda_version = getattr(torch.version, "cuda", None)
    return f"cuda{cuda_version}" if cuda_version else "cpu"
