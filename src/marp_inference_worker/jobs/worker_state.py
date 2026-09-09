# worker_state.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Shared worker state for the MARP Inference Worker.
# The job runner writes this and the FastAPI app reads it, which is the only
# thing the two share. Keeping it in one small object means /status reports what
# the runner actually believes rather than a second, drifting copy.
#
# It is the answer to the /status literal: describe() is assembled from the
# runner's real identity, slots, jobs, hardware and engine list (R12).
#
# The pause flag lives here too, because pause is set from two directions -- the
# machine's own operator through the loopback API, and the coordinator through a
# heartbeat -- and both mean the same thing.

# threading guards the fields the runner and the API touch from different threads.
import threading

# time stamps the last error and the process start.
import time

# Any types the free-form job and hardware mappings.
from typing import Any

# The version reported to callers, read from the installed package.
from importlib.metadata import PackageNotFoundError, version as package_version


# worker_version()
# Returns this worker's version.
# Inputs: none.
# Output: version string, or "unknown" when the package is not installed.
# Use this rather than a literal. The old /status hard-coded "0.1.0", which was
# already wrong the first time pyproject.toml's version moved.
def worker_version() -> str:

    # An editable install reports its version; a bare source checkout may not.
    try:
        return package_version("marp-inference-worker")
    except PackageNotFoundError:
        return "unknown"


# WorkerState
# What this worker currently is, as one object.
# Written by the runner, read by the API. Every mutator is guarded, because the
# runner loop and the API's request handlers are different threads.
class WorkerState:

    # __init__()
    # Builds an empty state for a worker that has not enrolled yet.
    # Inputs: none.
    # Output: initialized WorkerState.
    def __init__(self) -> None:

        # Guards every field below.
        self._lock = threading.Lock()

        # Identity, filled in once enrolment succeeds.
        self._local_id: str | None = None
        self._worker_id: str | None = None

        # Whether this worker is taking new work.
        self._paused = False
        self._pause_reason: str | None = None

        # The jobs currently running, as the runner described them.
        self._jobs: list[dict[str, Any]] = []

        # How many job slots this host has.
        self._slot_count = 0

        # The capabilities snapshot, so /status does not re-read hardware on
        # every request -- NVML queries are not free.
        self._capabilities: dict[str, Any] = {}

        # The last thing that went wrong, and when. Kept because a worker that
        # cannot reach its coordinator looks identical to an idle one otherwise.
        self._last_error: str | None = None
        self._last_error_at: float | None = None

        # When this process started, for an uptime a human can sanity-check.
        self._started_at = time.time()

    # set_identity()
    # Records the worker's identity after enrolment.
    # Inputs: the machine-local id and the coordinator's worker id.
    # Output: none.
    def set_identity(self, local_id: str, worker_id: str | None) -> None:

        with self._lock:
            self._local_id = local_id
            self._worker_id = worker_id

    # set_capabilities()
    # Records the hardware and engine snapshot.
    # Inputs: the capabilities mapping and the slot count.
    # Output: none.
    # Use this once at start. Hardware does not change while the process runs,
    # and re-reading NVML per request would make /status expensive.
    def set_capabilities(self, capabilities: dict[str, Any], slot_count: int) -> None:

        with self._lock:
            self._capabilities = capabilities
            self._slot_count = slot_count

    # set_jobs()
    # Records the currently running jobs.
    # Inputs: the list of job mappings from the runner.
    # Output: none.
    def set_jobs(self, jobs: list[dict[str, Any]]) -> None:

        with self._lock:
            self._jobs = jobs

    # set_paused()
    # Sets whether this worker takes new work.
    # Inputs: the flag and an optional reason.
    # Output: none.
    # Use this from the loopback API for the machine's own operator, and from
    # the runner when a heartbeat says pause. Pause stops new work; it does not
    # touch a job already running (A5).
    def set_paused(self, paused: bool, reason: str | None = None) -> None:

        with self._lock:
            self._paused = paused
            self._pause_reason = reason if paused else None

    # is_paused()
    # Whether this worker is currently refusing new work.
    # Inputs: none.
    # Output: True when paused.
    def is_paused(self) -> bool:

        with self._lock:
            return self._paused

    # note_error()
    # Records something that went wrong, for the operator to see.
    # Inputs: the message.
    # Output: none.
    # Use this for a coordinator failure the loop recovered from. A worker that
    # has been backing off for an hour must not look like an idle one.
    def note_error(self, message: str) -> None:

        with self._lock:
            self._last_error = message
            self._last_error_at = time.time()

    # describe()
    # Returns what is actually true about this worker.
    # Inputs: none.
    # Output: JSON-safe mapping for /status.
    # Use this from the status route. Nothing in it is a literal: identity comes
    # from enrolment, slots and hardware from discovery, jobs from the runner,
    # and the version from the installed package (R12).
    def describe(self) -> dict[str, Any]:

        with self._lock:

            # State is derived rather than stored, so it cannot disagree with
            # the job list beside it.
            if self._jobs:
                status = "working"
            elif self._paused:
                status = "paused"
            else:
                status = "idle"

            return {
                "worker_id": self._worker_id,
                "local_id": self._local_id,
                "enrolled": self._worker_id is not None,
                "status": status,
                "paused": self._paused,
                "pause_reason": self._pause_reason,
                "version": worker_version(),
                "uptime_s": round(time.time() - self._started_at, 1),
                "slots": {
                    "total": self._slot_count,
                    "busy": len(self._jobs),
                    "free": max(0, self._slot_count - len(self._jobs)),
                },
                "active_jobs": self._jobs,
                "capabilities": self._capabilities,
                "last_error": self._last_error,
                "last_error_at": self._last_error_at,
            }


# The process-wide worker state.
# A module-level singleton because the FastAPI app and the runner are two parts
# of one process and both need the same object; passing it through FastAPI's
# dependency graph would buy nothing here.
WORKER_STATE = WorkerState()
