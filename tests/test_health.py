# test_health.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Tests for the MARP Inference Worker health-check endpoint.
# This file should contain lightweight API tests that confirm the worker
# process can start, expose its health route, and return the expected
# basic service status response.

from fastapi.testclient import TestClient

from marp_inference_worker.main import app


# test_health_check_returns_ok()
# Verifies that the /health endpoint is reachable and returns an OK status.
# Inputs: none.
# Output: pytest pass/fail result based on response status and JSON body.
# Use this as the first smoke test for the FastAPI application.
def test_health_check_returns_ok() -> None:

    # Create a test client around the FastAPI app without starting a real server.
    client = TestClient(app)

    # Call the health endpoint exactly as an external client would.
    response = client.get("/health")

    # Confirm the endpoint returns a successful HTTP response.
    assert response.status_code == 200

    # Confirm the response body matches the worker's health-check contract.
    assert response.json() == {"status": "ok"}