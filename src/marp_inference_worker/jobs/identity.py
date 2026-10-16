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

# uuid5 derives this machine's identifier; uuid4 is the fallback when the
# machine will not tell us anything stable. NAMESPACE_DNS seeds our own
# namespace so the derived value is this project's and nobody else's.
from uuid import NAMESPACE_DNS, uuid4, uuid5

# sys picks the right place to read the machine's identifier from.
import sys

# Path handles the identity file location.
from pathlib import Path


# The namespace every derived worker id is minted under.
# Derived rather than a magic literal, so it is reproducible by inspection: the
# same machine identifier under a different namespace gives a different id, which
# is what stops a MARP worker id colliding with some other system's.
_WORKER_NAMESPACE = uuid5(NAMESPACE_DNS, "worker.marp.marineappliedresearch.org")


# machine_fingerprint()
# Returns an identifier the operating system keeps for this computer.
# Inputs: none.
# Output: the raw identifier, or None when it cannot be read.
# Use this only through derive_local_id(); the raw value never leaves the
# machine, because it identifies the computer to anything that can read it.
#
# Windows keeps `MachineGuid` under Cryptography and it survives uninstalling an
# application; Linux keeps `/etc/machine-id`, with the dbus copy as the older
# spelling. Both change when the operating system is reinstalled, which is the
# right behaviour -- that really is a different machine as far as we can tell.
def machine_fingerprint() -> str | None:

    # Windows: the registry value, read from the 64-bit view explicitly so a
    # 32-bit interpreter is not silently redirected to the WOW6432Node copy.
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
            ) as key:
                value = str(winreg.QueryValueEx(key, "MachineGuid")[0]).strip()
                return value or None
        except OSError:
            # A locked-down or unusual machine. Not an error: R5 says a worker
            # that cannot identify itself still enrols.
            return None

    # Everywhere else: the systemd id, then the older dbus location.
    for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            value = Path(candidate).read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            continue
    return None


# derive_local_id()
# Returns the id this machine should enrol under.
# Inputs: none.
# Output: a UUID string, stable for this computer where one can be derived.
# Use this when there is no stored identity. It is a UUID either way, so the
# column, the coordinator and every existing test see exactly what they did
# before -- what changed is that the value is now a function of the machine
# rather than of whichever state directory happened to be in use.
def derive_local_id() -> str:

    fingerprint = machine_fingerprint()

    # Nothing stable to go on: behave exactly as this file always has.
    if not fingerprint:
        return str(uuid4())

    # Hashed into our namespace rather than sent as-is, so the machine's own
    # identifier is never what travels or what gets stored upstream.
    return str(uuid5(_WORKER_NAMESPACE, fingerprint))


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

    # First run with no stored identity: derive the id from the machine, so a
    # reinstall or a lost state directory comes back as the same worker rather
    # than as a stranger (R1). Written before returning, so a crash between
    # deriving and enrolling does not produce two workers.
    identity = WorkerIdentity(local_id=derive_local_id())
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
