# test_worker_identity.py
# Created: 2026-10-15
# Author: Isaac Travers
#
# Worker identity tests for the MARP Inference Worker.
# These cover the one thing the pool view depends on: that a computer is the
# same worker tomorrow as it was today. The coordinator upserts on `local_id`,
# so everything here is really about whether that id is a property of the
# machine or an accident of which directory the worker happened to start in.
#
# Unit tests only -- no coordinator, no network, no registry writes.

# pytest provides the monkeypatch fixture the fallback cases need.
import pytest

# The module under test.
from marp_inference_worker.jobs import identity


# test_a_reinstall_enrols_as_the_same_worker()
# Verifies two state directories on one machine derive one id.
# Inputs: tmp_path for two separate, empty state directories.
# Output: pytest pass/fail result.
#
# This is the defect, written down. A worker that lost its state directory --
# uninstalled, installed as another user, or pointed somewhere new -- minted a
# fresh uuid4 and enrolled as a stranger, so four computers held eleven rows in
# the pool and nobody could tell which was which.
def test_a_reinstall_enrols_as_the_same_worker(tmp_path, monkeypatch) -> None:

    # One machine, so one fingerprint, whatever the state directory is.
    monkeypatch.setattr(identity, "machine_fingerprint", lambda: "a-stable-machine")

    first = identity.load_or_create_identity(tmp_path / "install-one" / "worker-identity.json")
    second = identity.load_or_create_identity(tmp_path / "install-two" / "worker-identity.json")

    assert first.local_id == second.local_id


# test_a_stored_identity_is_returned_unchanged()
# Verifies an existing install keeps the id it already enrolled under.
# Inputs: tmp_path for the state directory.
# Output: pytest pass/fail result.
#
# The machines already in the pool enrolled under random ids, and deriving a new
# one for them on upgrade would fork every one of them a final time. The file
# stays authoritative (R2).
def test_a_stored_identity_is_returned_unchanged(tmp_path, monkeypatch) -> None:

    path = tmp_path / "worker-identity.json"
    monkeypatch.setattr(identity, "machine_fingerprint", lambda: "a-stable-machine")

    stored = identity.WorkerIdentity(local_id="11111111-2222-3333-4444-555555555555",
                                     worker_id="65")
    identity.save_identity(path, stored)

    loaded = identity.load_or_create_identity(path)

    assert loaded.local_id == "11111111-2222-3333-4444-555555555555"
    assert loaded.worker_id == "65"


# test_a_machine_that_cannot_identify_itself_still_enrols()
# Verifies no stable identifier is still a working worker.
# Inputs: monkeypatch to silence the fingerprint.
# Output: pytest pass/fail result.
#
# A locked-down machine must still be able to donate time (R5). It gets the old
# behaviour -- a random id per state directory -- rather than no id at all.
def test_a_machine_that_cannot_identify_itself_still_enrols(monkeypatch) -> None:

    monkeypatch.setattr(identity, "machine_fingerprint", lambda: None)

    first = identity.derive_local_id()
    second = identity.derive_local_id()

    # A usable uuid each time, and not pinned to anything, because there was
    # nothing to pin it to.
    assert len(first) == 36 and first.count("-") == 4
    assert first != second


# test_the_derived_id_does_not_carry_the_machine_identifier()
# Verifies the raw fingerprint is not what travels.
# Inputs: monkeypatch supplying a recognizable fingerprint.
# Output: pytest pass/fail result.
#
# `MachineGuid` identifies the computer to anything that can read it, so it is
# hashed into our own namespace rather than sent upstream as-is (R3).
def test_the_derived_id_does_not_carry_the_machine_identifier(monkeypatch) -> None:

    monkeypatch.setattr(identity, "machine_fingerprint", lambda: "7f3a9c21-MACHINE-GUID")

    derived = identity.derive_local_id()

    assert "MACHINE" not in derived.upper()
    assert derived != "7f3a9c21-MACHINE-GUID"


# test_the_reported_name_is_the_hostname()
# Verifies enrolment stops decorating the name with a random suffix.
# Inputs: monkeypatch over the activation module's hostname and transport.
# Output: pytest pass/fail result.
#
# Four rows for one computer were hard to read as four installs rather than four
# computers precisely because each name ended in eight characters of its own
# uuid (R4). The name is the hostname; the id is what distinguishes.
def test_the_reported_name_is_the_hostname(tmp_path, monkeypatch) -> None:

    from marp_inference_worker.installation import activation

    sent = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"worker_id": 65, "credential": "x", "token_prefix": "abcd"}

    def _post(url, json=None, timeout=None):
        sent.update(json or {})
        return _Response()

    monkeypatch.setattr(activation.httpx, "post", _post)
    monkeypatch.setattr(activation.socket, "gethostname", lambda: "ABYSS")
    monkeypatch.setattr(activation.credential_store, "save", lambda *a, **k: None)

    activation.activate("http://coordinator.invalid", "code", tmp_path)

    assert sent["name"] == "ABYSS"
