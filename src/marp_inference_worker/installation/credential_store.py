"""Protect the worker's machine credential: Windows DPAPI, or file permissions elsewhere."""

import ctypes
import os
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

    # Off Windows the credential is stored as it is, and `save()` is what protects it by
    # creating the file 0600. Read this before assuming the name means encryption.
    #
    # What that defends against: another user account on the machine reading the file.
    # What it does not: anyone who can already run as this user, read this user's files,
    # or read the disk. There is no encryption at rest here at all.
    #
    # This is a deliberate reduction from the Windows behaviour, not an oversight. DPAPI
    # binds the credential to a Windows account and Linux has no equivalent primitive.
    # Secret Service (libsecret) is the closest analogue and was rejected because it needs
    # an unlocked keyring, which a headless machine logging in automatically does not have
    # -- an option that fails on the target hardware is not the more secure option, it is
    # the one that does not ship. Deriving a key from /etc/machine-id was rejected as
    # obfuscation: whatever can read this file can read machine-id too.
    #
    # The credential is machine-specific and individually revocable, so the blast radius of
    # losing one is that one volunteer's worker, not the pool.
    if sys.platform != "win32":
        return value.encode("utf-8")

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

    # The mirror of protect(): off Windows the bytes are the credential.
    if sys.platform != "win32":
        return value.decode("utf-8")

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

    # A previous run that died between creating this and renaming it leaves the temporary
    # behind, and O_EXCL below would then refuse forever rather than once.
    temporary.unlink(missing_ok=True)

    # Create with 0600 rather than writing and chmod-ing afterwards. The obvious order
    # leaves the credential on disk world-readable for as long as it takes to reach the
    # next line, which on a shared machine is the whole of the protection missing. O_EXCL
    # means we never write into a file somebody else made and left readable.
    #
    # The mode is ignored on Windows, where DPAPI is the protection and this is just a file.
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, protect(credential))
    finally:
        os.close(descriptor)

    temporary.replace(path)


def load(path: Path) -> str | None:
    return unprotect(path.read_bytes()) if path.is_file() else None
