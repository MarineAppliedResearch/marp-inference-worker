# status_routes.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Worker status route definitions for the MARP Inference Worker API.
# This file should contain lightweight endpoints that report worker runtime
# status to coordinators, dashboards, and local developer tools. Heavy
# hardware checks, model loading, and job execution should not live here.

# FastAPI provides the router used to group worker status endpoints.
from fastapi import APIRouter


# Router for worker status endpoints.
# Keeping status routes separate makes the worker monitoring API easy to extend.
router = APIRouter(prefix="/status", tags=["status"])


# get_worker_status()
# Returns the worker's current status using the public status contract.
# Inputs: none.
# Output: dictionary containing worker identity, state, version, jobs, and models.
# Use this endpoint when a coordinator needs to decide whether this worker is available.
@router.get("")
async def get_worker_status() -> dict[str, object]:

    # Return a static status response until the worker state object is implemented.
    return {
        "worker_id": "dev-worker",
        "status": "idle",
        "version": "0.1.0",
        "active_jobs": 0,
        "loaded_models": [],
    }