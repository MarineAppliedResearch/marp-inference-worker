# health_routes.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Health-check route definitions for the MARP Inference Worker API.
# This file should contain lightweight endpoints used to verify that the
# worker API is running and reachable. Do not place model loading, job
# execution, or expensive runtime checks in this file.

from fastapi import APIRouter


# Router for health-check endpoints.
# Keeping this router separate makes it easy to register and test independently.
router = APIRouter(prefix="/health", tags=["health"])


# health_check()
# Returns a small response confirming that the API process is reachable.
# Inputs: none.
# Output: dictionary containing a simple health status string.
# Use this for uptime checks, local testing, and basic service monitoring.
@router.get("")
async def health_check() -> dict[str, str]:

    # Return the smallest useful health response so this endpoint stays cheap.
    return {"status": "ok"}