# test_status.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Tests for the MARP Inference Worker status endpoint.
# This file should verify the public worker status contract used by future
# coordinators and dashboards. Keep these tests focused on API behavior,
# not internal implementation details.

from fastapi.testclient import TestClient

from marp_inference_worker.main import app


# test_status_returns_worker_status_contract()
# Verifies that /status returns the expected worker status fields.
# Inputs: none.
# Output: pytest pass/fail result based on response status and JSON body.
# Use this test to protect the coordinator-facing status API contract.
def test_status_returns_worker_status_contract() -> None:

    # Create a test client around the FastAPI app without starting a server.
    client = TestClient(app)

    # Call the status endpoint as an external coordinator would.
    response = client.get("/status")

    # Confirm the endpoint returns a successful HTTP response.
    assert response.status_code == 200

    # Parse the JSON body so the individual contract fields can be checked.
    body = response.json()

    # Confirm the static development worker identity is present.
    assert body["worker_id"] == "dev-worker"

    # Confirm the worker reports itself as available in the initial skeleton.
    assert body["status"] == "idle"

    # Confirm the API version is included for coordinator compatibility checks.
    assert body["version"] == "0.1.0"

    # Confirm the initial worker has no active jobs.
    assert body["active_jobs"] == 0

    # Confirm the initial worker has no loaded models.
    assert body["loaded_models"] == []