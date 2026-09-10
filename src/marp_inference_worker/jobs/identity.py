# identity.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Persistent worker identity for the MARP Inference Worker.
# A worker's identity has to survive a restart, or the coordinator sees a new
# machine every time the service bounces and loses the jobs it had leased (R3).
# This file owns reading, writing and generating that identity, and nothing else.

# json stores the identity file in a form a human can read and edit.
import json

# uuid4 generates the local identifier a fresh worker enrols with.
from uuid import uuid4

# Path handles the identity file location.
from pathlib import Path


# WorkerIdentity
# The worker's durable name plus whatever the coordinator assigned it.
# `local_id` is generated once on this machine and never changes; `worker_id` is
# what the coordinator calls this worker and is filled in by enrolment.
class WorkerIdentity:

    # __init__()
    # Builds an identity object from its stored fields.
    # Inputs: the machine-local id, and the coordinator's id if enrolled.
    # Output: initialized WorkerIdentity.
    def __init__(self, local_id: str, worker_id: str | None = None) -> None:

        # Generated once per machine; the stable thing across re-enrolments.
        self.local_id = local_id

        # Assigned by the coordinator at enrolment. None until the first enrol.
        self.worker_id = worker_id

    # to_dict()
    # Returns the identity as a JSON-safe mapping.
    # Inputs: none.
    # Output: dictionary written to the identity file.
    def to_dict(self) -> dict[str, str | None]:

        # Keep the file shape flat so it stays hand-editable.
        return {"local_id": self.local_id, "worker_id": self.worker_id}


# default_identity_path()
# Returns where the identity file lives by default.
# Inputs: optional state directory override.
# Output: path to worker-identity.json.
# Use this rather than writing the location into several call sites.
def default_identity_path(state_dir: Path | None = None) -> Path:

    # State defaults beside the service, under a directory the .gitignore covers.
    root = state_dir if state_dir is not None else Path("data") / "worker"
    return root / "worker-identity.json"


# load_or_create_identity()
# Reads the worker's identity, generating one on first run.
# Inputs: path to the identity file.
# Output: WorkerIdentity, already persisted to disk.
# Use this at startup, before the first coordinator call.
def load_or_create_identity(path: Path) -> WorkerIdentity:

    # An existing file is authoritative; a restart must not invent a new worker.
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        return WorkerIdentity(
            local_id=str(stored["local_id"]),
            worker_id=stored.get("worker_id"),
        )

    # First run on this machine: mint a local id and write it before returning,
    # so a crash between generating and enrolling does not produce two workers.
    identity = WorkerIdentity(local_id=str(uuid4()))
    save_identity(path, identity)
    return identity


# save_identity()
# Writes the identity file, creating its directory if needed.
# Inputs: path and identity object.
# Output: none.
# Use this after enrolment assigns a coordinator worker id.
def save_identity(path: Path, identity: WorkerIdentity) -> None:

    # The parent may not exist on a fresh machine.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write via a temporary file so an interrupted write cannot leave a
    # half-written identity that the next start would fail to parse.
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(identity.to_dict(), indent=2), encoding="utf-8")
    temp_path.replace(path)
