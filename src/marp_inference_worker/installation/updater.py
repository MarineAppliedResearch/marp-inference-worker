"""Download and stage an approved worker release without touching the active one."""

import hashlib
import json
import os
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from marp_inference_worker.installation.platform_info import current_platform


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class UpdateInstaller:
    """Stages one verified version beside the running installation."""

    def __init__(self, state_dir: Path, install_root: Path | None = None) -> None:
        local_app_data = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        self.install_root = install_root or Path(
            os.environ.get("MARP_WORKER_INSTALL_ROOT") or local_app_data / "MARP" / "Worker"
        )
        self.state_dir = state_dir

    def stage(self, release: dict[str, Any]) -> Path:
        version = str(release["version"])
        runtime = str(release["compute_runtime"])
        release_key = f"{version}-{runtime}"
        if not re.fullmatch(r"[A-Za-z0-9._-]+", release_key):
            raise ValueError("release version or compute runtime is unsafe")
        expected_size = int(release["size_bytes"])
        expected_sha = str(release["sha256"]).lower()
        staging_root = self.state_dir / "update-staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        archive = staging_root / f"{version}.zip.part"

        digest = hashlib.sha256()
        received = 0
        with httpx.stream("GET", str(release["download_url"]), follow_redirects=True, timeout=300) as response:
            response.raise_for_status()
            with archive.open("wb") as output:
                for chunk in response.iter_bytes():
                    received += len(chunk)
                    if received > expected_size:
                        raise ValueError("update exceeded its approved size")
                    digest.update(chunk)
                    output.write(chunk)
        if received != expected_size:
            raise ValueError(f"update size was {received}, expected {expected_size}")
        if digest.hexdigest() != expected_sha:
            raise ValueError("update SHA-256 did not match the approved release")

        target = self.install_root / "versions" / release_key
        staging = self.install_root / "versions" / f".{release_key}.staging"
        if target.exists():
            # An explicit retry after rollback targets the same side-by-side
            # release key. Replace that inactive copy from the freshly verified
            # archive so the operator's retry is meaningful and deterministic.
            shutil.rmtree(target)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        try:
            with zipfile.ZipFile(archive) as package:
                manifest = json.loads(package.read("manifest.json"))
                system, architecture = current_platform()
                if manifest.get("version") != version:
                    raise ValueError("package manifest version does not match release")
                if manifest.get("platform") != system or manifest.get("architecture") != architecture:
                    raise ValueError("package is for another platform or architecture")
                if manifest.get("compute_runtime") != release.get("compute_runtime"):
                    raise ValueError("package compute runtime does not match the approved release")

                for member in package.infolist():
                    normalized = member.filename.replace("\\", "/")
                    relative = PurePosixPath(normalized)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"unsafe update path: {member.filename}")
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise ValueError(f"update contains a symbolic link: {member.filename}")
                    destination = staging.joinpath(*relative.parts)
                    if member.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with package.open(member) as source, destination.open("wb") as output:
                            shutil.copyfileobj(source, output)

            staging.replace(target)
            active_path = self.install_root / "active.json"
            active = json.loads(active_path.read_text(encoding="utf-8")) if active_path.is_file() else {}
            _atomic_json(self.state_dir / "pending-update.json", {
                "previous_release": active.get("release"),
                "target_release": release_key,
            })
            return target
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        finally:
            archive.unlink(missing_ok=True)
