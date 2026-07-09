# File name: test_system_resources.py
# Date created: 2026-07-08
# Author: Isaac Travers
# Purpose: Tests the worker system resource API endpoint.
# System role: Protects the coordinator-facing resource route contract.
# Code in this file should test resource endpoint shape without requiring specific hardware.

# FastAPI TestClient provides in-process HTTP testing for API routes.
from fastapi.testclient import TestClient

# The app object exposes the configured MARP inference worker API.
from marp_inference_worker.main import app


# Shared API client used by this test module.
client = TestClient(app)


# Verify that the system resources route exists and returns the expected top-level shape.
# Inputs: none.
# Outputs: passing test when the route returns resource sections.
# This protects the future coordinator-facing resource endpoint contract.
def test_system_resources_route_returns_expected_sections():

    # Request the worker resource snapshot.
    response = client.get("/system/resources")

    # The resource endpoint should return successfully.
    assert response.status_code == 200

    # Parse the response body as JSON.
    data = response.json()

    # Verify stable top-level resource sections.
    assert "host" in data
    assert "cpu" in data
    assert "memory" in data
    assert "disk" in data
    assert "python" in data
    assert "torch" in data
    assert "job_pressure" in data

    # Verify a few coordinator-useful nested fields.
    assert "platform" in data["host"]
    assert "logical_cpu_count" in data["cpu"]
    assert "percent_used" in data["memory"]
    assert "free_bytes" in data["disk"]
    assert "torch_available" in data["torch"]
    assert "worker_accepting_jobs" in data["job_pressure"]