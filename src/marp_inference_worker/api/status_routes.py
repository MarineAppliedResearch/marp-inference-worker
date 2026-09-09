# status_routes.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Worker status and control routes for the MARP Inference Worker API.
# These endpoints serve the machine's own operator, not MARP. The app binds to
# loopback only and the coordinator never calls it -- all coordinator traffic is
# outbound from the worker (R1). What the operator gets is what this worker
# currently is, and the ability to stop it taking new work.
#
# Route handlers stay thin: the state object owns what is true and the runner
# owns what changes it.

# FastAPI provides the router used to group these endpoints.
from fastapi import APIRouter

# Pydantic validates the pause request body.
from pydantic import BaseModel

# The shared worker state, written by the runner and read here.
from marp_inference_worker.jobs.worker_state import WORKER_STATE

# The engine registry reports what this worker can run.
from marp_inference_worker.engines import engine_registry


# Router for worker status and local control endpoints.
router = APIRouter(prefix="/status", tags=["status"])


# PauseRequest
# The operator's request to stop or resume taking new work.
# A reason is optional but worth recording: a worker paused for a week with no
# note is indistinguishable from one paused by mistake.
class PauseRequest(BaseModel):

    # True to stop taking new work, False to resume.
    paused: bool

    # Optional free-text note shown in /status while paused.
    reason: str | None = None


# get_worker_status()
# Returns what is actually true about this worker.
# Inputs: none.
# Output: identity, state, version, slots, running jobs, hardware and engines.
# Use this from the operator's own machine to see what the worker is doing.
#
# This replaced a hard-coded literal that reported worker_id "dev-worker",
# status "idle", version "0.1.0", no jobs and no models, regardless of what the
# worker was doing -- so it read exactly the same on an idle worker, a busy one
# and one that had never enrolled (R12).
@router.get("")
async def get_worker_status() -> dict[str, object]:

    # Assembled from the runner's state, plus the engine list, which the state
    # object does not hold because it is fixed by what is registered.
    described = WORKER_STATE.describe()
    described["engines"] = engine_registry.describe_engines()
    return described


# set_paused()
# Stops or resumes this worker taking new work.
# Inputs: the pause request body.
# Output: the worker's status after the change.
# Use this to drain a machine before shutting it down, or to free it for
# somebody who wants to use it.
#
# Pause deliberately leaves a running job alone. Stopping a job that is already
# running is the coordinator's `abandon`, and mixing the two would mean an
# operator pausing a worker to free their GPU silently threw away an hour of
# somebody's inference (A5).
@router.post("/pause")
async def set_paused(request: PauseRequest) -> dict[str, object]:

    # Record the flag and the reason; the runner reads it on its next pass and
    # reports zero free slots, which is how it stops being offered work.
    WORKER_STATE.set_paused(request.paused, reason=request.reason)

    # Return the new state, so the caller does not have to poll /status to see
    # whether the change took.
    return WORKER_STATE.describe()
