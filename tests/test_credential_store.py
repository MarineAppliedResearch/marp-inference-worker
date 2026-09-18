# test_credential_store.py
# Created: 2026-09-18
# Author: Isaac Travers
#
# Tests for where the worker's machine credential lives at rest.
#
# These exist because the store was Windows-only and nobody could see it: a Linux
# volunteer's activation reached the coordinator, took the single-use code, and then
# raised writing the credential to disk. The failure was invisible until the first
# Linux machine tried, so the tier that can observe it is a real save and a real
# stat of the file that comes out. Refs #32.
#
# The Windows path cannot be executed here and these tests do not pretend to cover
# it -- see test_windows_path_is_not_taken_off_windows for what is actually asserted.

# stat reads the permission bits, which are the whole of the protection off Windows.
import os
import stat
import sys

import pytest

from marp_inference_worker.installation import credential_store

CREDENTIAL = "worker-credential-value-for-tests"


@pytest.fixture
def credential_path(tmp_path):
    return tmp_path / "credential"


# R5 -- a machine that has never activated has no credential, and asking for one is a
# normal question with a normal answer. Raising here would make "not yet activated" and
# "broken" look the same, which is how the original failure read.
def test_load_before_activation_returns_none(credential_path):
    assert credential_store.load(credential_path) is None


# R1, R5 -- the credential that comes back is the credential that went in.
def test_round_trip_returns_the_credential(credential_path):
    credential_store.save(credential_path, CREDENTIAL)

    assert credential_store.load(credential_path) == CREDENTIAL


# R4 -- the point of the whole change. A credential another user can read is not stored,
# it is published.
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_credential_file_is_owner_only(credential_path):
    credential_store.save(credential_path, CREDENTIAL)

    assert stat.S_IMODE(credential_path.stat().st_mode) == 0o600


# R4 -- and it is owner-only from the moment it exists. Writing first and chmod-ing after
# leaves the secret world-readable for the width of one statement, which is a real window
# on a shared machine. Asserting the final mode alone cannot tell the two implementations
# apart, so this asserts the temporary the write actually goes through.
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_credential_is_never_briefly_world_readable(credential_path, monkeypatch):
    observed = {}
    real_open = os.open

    def recording_open(path, flags, mode=0o777, *args, **kwargs):
        if str(path).endswith(".tmp"):
            observed["mode"] = mode
            observed["flags"] = flags
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", recording_open)
    credential_store.save(credential_path, CREDENTIAL)

    assert observed["mode"] == 0o600
    assert observed["flags"] & os.O_EXCL

    # The full set, so dropping a flag is a failure rather than a silent change.
    # O_BINARY is the one that matters and is the one this platform cannot see: it is 0
    # off Windows, where a text-mode descriptor would translate 0x0A to 0x0D 0x0A and
    # corrupt DPAPI ciphertext on write. Only the Windows suite can observe that.
    expected = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    assert observed["flags"] == expected


# R1 -- re-activating replaces the credential rather than refusing, and the replacement is
# no more readable than the original.
def test_saving_twice_replaces_the_credential(credential_path):
    credential_store.save(credential_path, "first")
    credential_store.save(credential_path, "second")

    assert credential_store.load(credential_path) == "second"
    if sys.platform != "win32":
        assert stat.S_IMODE(credential_path.stat().st_mode) == 0o600


# R1 -- a run that died between creating the temporary and renaming it leaves one behind.
# O_EXCL refuses an existing file, so without the unlink that wreckage would wedge every
# later activation on a machine rather than one.
def test_stale_temporary_does_not_wedge_activation(credential_path):
    credential_path.parent.mkdir(parents=True, exist_ok=True)
    credential_path.with_suffix(".tmp").write_text("left behind by a killed run")

    credential_store.save(credential_path, CREDENTIAL)

    assert credential_store.load(credential_path) == CREDENTIAL


# R1 -- the temporary is not left next to the credential afterwards.
def test_no_temporary_is_left_behind(credential_path):
    credential_store.save(credential_path, CREDENTIAL)

    assert not credential_path.with_suffix(".tmp").exists()


# R2 -- Windows keeps DPAPI. This asserts the branch, not the behaviour: DPAPI cannot run
# here, so what is checked is that the non-Windows path returns the bytes unchanged and
# therefore that nothing on win32 has been rerouted through it. Real coverage of the
# Windows path needs a Windows machine and is not claimed by this file.
def test_windows_path_is_not_taken_off_windows():
    if sys.platform == "win32":
        pytest.skip("DPAPI is the path on Windows; this asserts the other branch")

    assert credential_store.protect(CREDENTIAL) == CREDENTIAL.encode("utf-8")
    assert credential_store.unprotect(CREDENTIAL.encode("utf-8")) == CREDENTIAL
