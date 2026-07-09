# File name: system_routes.py
# Date created: 2026-07-08
# Author: Isaac Travers
# Purpose: Defines API routes for worker-local system resource information.
# System role: Gives coordinators and developers visibility into worker capacity.
# Code in this file should stay thin and delegate resource inspection to system utilities.

# FastAPI provides the router used to group system endpoints.
from fastapi import APIRouter

# Resource monitor owns local CPU, RAM, disk, Python, Torch, and CUDA inspection.
from marp_inference_worker.system.resource_monitor import get_system_resources


# Router for worker-local system information endpoints.
router = APIRouter(prefix="/system", tags=["system"])


# Return a point-in-time snapshot of local worker resources.
# Inputs: none.
# Outputs: JSON response with CPU, memory, disk, runtime, and GPU fields.
# Use this endpoint when a coordinator needs worker capacity information.
@router.get("/resources")
def read_system_resources() -> dict:

    # Delegate resource collection so route code stays thin.
    resources = get_system_resources()

    return resources