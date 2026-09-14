"""Protect the worker's machine credential with Windows DPAPI."""

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Any


class _Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _blob(value: bytes) -> tuple[_Blob, Any]:
    buffer = ctypes.create_string_buffer(value)
    return _Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _windows_functions() -> tuple[Any, Any, Any]:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    protect_data = crypt32.CryptProtectData
    protect_data.argtypes = [
        ctypes.POINTER(_Blob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_Blob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_Blob),
    ]
    protect_data.restype = wintypes.BOOL
    unprotect_data = crypt32.CryptUnprotectData
    unprotect_data.argtypes = [
        ctypes.POINTER(_Blob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_Blob),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(_Blob),
    ]
    unprotect_data.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = [wintypes.HLOCAL]
    local_free.restype = wintypes.HLOCAL
    return protect_data, unprotect_data, local_free


def protect(value: str) -> bytes:
    if sys.platform != "win32":
        raise RuntimeError("Installed worker credentials require Windows DPAPI")
    source, source_buffer = _blob(value.encode("utf-8"))
    description = "MARP inference worker credential"
    result = _Blob()
    protect_data, _, local_free = _windows_functions()
    if not protect_data(
        ctypes.byref(source), description, None, None, None, 0x1, ctypes.byref(result)
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        local_free(ctypes.cast(result.data, wintypes.HLOCAL))
        del source_buffer


def unprotect(value: bytes) -> str:
    if sys.platform != "win32":
        raise RuntimeError("Installed worker credentials require Windows DPAPI")
    source, source_buffer = _blob(value)
    result = _Blob()
    _, unprotect_data, local_free = _windows_functions()
    if not unprotect_data(
        ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(result)
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(result.data, result.size).decode("utf-8")
    finally:
        local_free(ctypes.cast(result.data, wintypes.HLOCAL))
        del source_buffer


def save(path: Path, credential: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(protect(credential))
    temporary.replace(path)


def load(path: Path) -> str | None:
    return unprotect(path.read_bytes()) if path.is_file() else None
