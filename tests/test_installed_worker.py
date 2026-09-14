import hashlib
import io
import json
import zipfile

import httpx
import pytest

from marp_inference_worker.installation import credential_store
from marp_inference_worker.installation import launcher
from marp_inference_worker.installation.updater import UpdateInstaller


def _package(version: str = "0.2.0", extra_name: str | None = None) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("manifest.json", json.dumps({
            "version": version,
            "platform": "windows",
            "architecture": "x86_64",
            "compute_runtime": "cuda12.6",
            "entrypoint": "marp-worker.exe",
        }))
        archive.writestr("marp-worker.exe", b"worker payload")
        if extra_name:
            archive.writestr(extra_name, b"must not escape")
    return output.getvalue()


def _release(package: bytes, version: str = "0.2.0") -> dict:
    return {
        "version": version,
        "platform": "windows",
        "architecture": "x86_64",
        "compute_runtime": "cuda12.6",
        "download_url": "https://github.com/MarineAppliedResearch/marp-inference-worker/releases/download/v0.2.0/worker.zip",
        "size_bytes": len(package),
        "sha256": hashlib.sha256(package).hexdigest(),
    }


def _serve(monkeypatch, package: bytes) -> httpx.Client:
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=package, request=request)
    ))
    monkeypatch.setattr(httpx, "stream", client.stream)
    return client


def test_dpapi_credential_round_trip_is_not_plaintext(tmp_path) -> None:
    path = tmp_path / "worker-credential.dpapi"
    secret = "svc_this-is-the-test-secret"
    credential_store.save(path, secret)

    assert credential_store.load(path) == secret
    assert secret.encode("utf-8") not in path.read_bytes()


def test_launcher_reads_windows_powershell_utf8_json(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"coordinator_url":"http://marp.test"}')

    assert launcher._read_json(path) == {"coordinator_url": "http://marp.test"}


def test_update_is_verified_and_staged_beside_the_active_release(monkeypatch, tmp_path) -> None:
    package = _package()
    client = _serve(monkeypatch, package)
    install_root = tmp_path / "install"
    state_dir = tmp_path / "state"
    install_root.mkdir()
    (install_root / "active.json").write_text('{"release":"0.1.0-cuda12.6"}', encoding="utf-8")

    try:
        target = UpdateInstaller(state_dir, install_root).stage(_release(package))
    finally:
        client.close()

    assert target == install_root / "versions" / "0.2.0-cuda12.6"
    assert (target / "marp-worker.exe").read_bytes() == b"worker payload"
    pending = json.loads((state_dir / "pending-update.json").read_text(encoding="utf-8"))
    assert pending == {
        "previous_release": "0.1.0-cuda12.6",
        "target_release": "0.2.0-cuda12.6",
    }


def test_update_rejects_path_traversal(monkeypatch, tmp_path) -> None:
    package = _package(extra_name="../outside.txt")
    client = _serve(monkeypatch, package)
    install_root = tmp_path / "install"
    install_root.mkdir()
    try:
        with pytest.raises(ValueError, match="unsafe update path"):
            UpdateInstaller(tmp_path / "state", install_root).stage(_release(package))
    finally:
        client.close()
    assert not (tmp_path / "outside.txt").exists()


def test_update_rejects_bytes_that_do_not_match_approval(monkeypatch, tmp_path) -> None:
    package = _package()
    release = _release(package)
    release["sha256"] = "0" * 64
    client = _serve(monkeypatch, package)
    try:
        with pytest.raises(ValueError, match="SHA-256"):
            UpdateInstaller(tmp_path / "state", tmp_path / "install").stage(release)
    finally:
        client.close()
