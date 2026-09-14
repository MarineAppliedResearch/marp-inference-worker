"""One-time installation activation for a Windows worker."""

import getpass
import os
import socket
from pathlib import Path

import httpx

from marp_inference_worker.installation import credential_store
from marp_inference_worker.installation.platform_info import current_platform, installed_compute_runtime
from marp_inference_worker.jobs.identity import load_or_create_identity, save_identity
from marp_inference_worker.jobs.worker_state import worker_version


def credential_path(state_dir: Path) -> Path:
    return state_dir / "worker-credential.dpapi"


def activate(coordinator_url: str, activation_code: str, state_dir: Path) -> dict:
    identity_path = state_dir / "worker-identity.json"
    identity = load_or_create_identity(identity_path)
    system, architecture = current_platform()
    response = httpx.post(
        f"{coordinator_url.rstrip('/')}/api/v2/gpu/workers/activate",
        json={
            "activation_code": activation_code,
            "local_id": identity.local_id,
            "name": f"{socket.gethostname()}-{identity.local_id[:8]}",
            "platform": system,
            "architecture": architecture,
            "compute_runtime": installed_compute_runtime(),
            "worker_version": worker_version(),
        },
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    credential_store.save(credential_path(state_dir), result["credential"])
    identity.worker_id = str(result["worker_id"])
    save_identity(identity_path, identity)
    return {"worker_id": identity.worker_id, "token_prefix": result["token_prefix"]}


def main() -> None:
    coordinator = os.environ.get("MARP_COORDINATOR_URL") or input("MARP API URL: ").strip()
    code = os.environ.pop("MARP_ACTIVATION_CODE", None) or getpass.getpass("Activation code: ")
    state_dir = Path(os.environ.get("MARP_WORKER_STATE_DIR") or Path.home() / "AppData" / "Local" / "MARP" / "Worker")
    result = activate(coordinator, code, state_dir)
    print(f"Activated worker {result['worker_id']} ({result['token_prefix']}…)")


if __name__ == "__main__":
    main()
