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
from typing import Literal

# The shared worker state, written by the runner and read here.
from marp_inference_worker.jobs.worker_state import WORKER_STATE
from marp_inference_worker.installation.operator_control import write_action

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


class ControlRequest(BaseModel):
    action: Literal["finish", "stop", "resume"]


class ScreenRequest(BaseModel):
    mode: Literal["off", "window", "fullscreen"]


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

    # Keep the original endpoint as an alias for Finish up and Resume. If it
    # changed only the in-memory flag, a restart could unexpectedly take work,
    # or it could contradict a durable Stop now selection.
    action = "finish" if request.paused else "running"
    write_action(action)
    WORKER_STATE.set_operator_action(action)
    if request.paused and request.reason:
        WORKER_STATE.set_paused(True, reason=request.reason)

    # Return the new state, so the caller does not have to poll /status to see
    # whether the change took.
    return WORKER_STATE.describe()


@router.post("/control")
async def control_worker(request: ControlRequest) -> dict[str, object]:
    action = "running" if request.action == "resume" else request.action
    write_action(action)
    WORKER_STATE.set_operator_action(action)
    return WORKER_STATE.describe()


@router.post("/screen")
async def set_screen_mode(request: ScreenRequest) -> dict[str, object]:
    # A start-menu launch reaches the already-running sign-in worker here. The
    # policy applies to later jobs; an existing job keeps the window it owns.
    WORKER_STATE.set_screen_mode(request.mode)
    return WORKER_STATE.describe()
