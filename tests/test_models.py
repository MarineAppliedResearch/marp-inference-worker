# test_models.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Tests for the MARP Inference Worker model-management endpoints.
# This file verifies the temporary model-loading contract used before real
# model caching and inference engines are implemented. Keep these tests focused
# on API behavior, not future engine internals.

# FastAPI TestClient lets tests call the API without starting Uvicorn.
from fastapi.testclient import TestClient

# The app object is imported so endpoint tests use the real API registration.
from marp_inference_worker.main import app

# The temporary model manager is cleared between tests to isolate fake state.
from marp_inference_worker.models import model_manager

# Mock patches cache behavior so endpoint tests do not download real artifacts.
from unittest.mock import patch


# test_loaded_models_starts_empty()
# Verifies that /models/loaded returns an empty list before models are loaded.
# Inputs: none.
# Output: pytest pass/fail result based on response status and JSON body.
# Use this to protect the initial loaded-model response contract.
def test_loaded_models_starts_empty() -> None:

    # Clear temporary model state so this test is independent from other tests.
    model_manager.clear_loaded_models()

    # Create a test client around the FastAPI app without starting a server.
    client = TestClient(app)

    # Request the current loaded-model list from the worker.
    response = client.get("/models/loaded")

    # Confirm the endpoint returns a successful HTTP response.
    assert response.status_code == 200

    # Confirm the fake worker starts with no loaded models.
    assert response.json() == {"loaded_models": []}


# test_load_model_records_model_spec()
# Verifies that POST /models/load stores and returns the provided model spec.
# Inputs: none.
# Output: pytest pass/fail result based on load response and loaded-model list.
# Use this to protect the temporary model-loading API contract.
def test_load_model_records_model_spec() -> None:

    # Clear temporary model state so this test is independent from other tests.
    model_manager.clear_loaded_models()

    # Create a test client around the FastAPI app without starting a server.
    client = TestClient(app)

    # Define a representative Ultralytics model spec without loading real files.
    model_spec = {
        "model_id": "demo_yolo",
        "engine": "ultralytics",
        "model_arch": "yolo11",
        "task": "detect",
        "artifact": {
            "url": "https://model-server/models/demo_yolo.pt",
            "format": "ultralytics_pt",
            "sha256": None,
        },
        "load_settings": {
            "device": "auto",
        },
        "labels": [
            {
                "class_id": 0,
                "class_name": "bat star",
                "external_id": None,
            }
        ],
    }

    # Mock cache behavior so this endpoint test does not download a real model.
    with patch(
        "marp_inference_worker.models.model_cache.ensure_artifact_cached",
        return_value={
            "cache_key": "demo_yolo",
            "artifact_path": "models\\cache\\demo_yolo\\demo_yolo.pt",
            "is_cached": True,
            "cache_action": "downloaded",
        },
    ):
        # Submit the model spec to the load endpoint.
        load_response = client.post("/models/load", json=model_spec)

    # Confirm the load endpoint accepts the valid model spec.
    assert load_response.status_code == 200

    # Confirm the response reports the model as loaded.
    assert load_response.json()["status"] == "loaded"

    # Confirm the response echoes the validated model contract.
    assert load_response.json()["model"] == model_spec

    # Confirm the response includes the local cache state.
    cache = load_response.json()["cache"]

    # Confirm the model ID is used as the cache key when no hash is provided.
    assert cache["cache_key"] == "demo_yolo"

    # Confirm the artifact path points to the expected cache filename.
    assert cache["artifact_path"].endswith("models\\cache\\demo_yolo\\demo_yolo.pt")

    # Confirm the mocked cache path reports the artifact as cached.
    assert cache["is_cached"] is True

    # Confirm the mocked cache path reports that a download occurred.
    assert cache["cache_action"] == "downloaded"

    # Request the loaded-model list after loading.
    loaded_response = client.get("/models/loaded")

    # Confirm the list endpoint still returns a successful response.
    assert loaded_response.status_code == 200

    # Confirm the loaded model appears in the worker's loaded-model state.
    assert loaded_response.json() == {"loaded_models": [model_spec]}


# test_load_model_rejects_missing_required_fields()
# Verifies that POST /models/load rejects incomplete model specs.
# Inputs: none.
# Output: pytest pass/fail result based on FastAPI validation behavior.
# Use this to protect required fields in the public model contract.
def test_load_model_rejects_missing_required_fields() -> None:

    # Clear temporary model state so this test is independent from other tests.
    model_manager.clear_loaded_models()

    # Create a test client around the FastAPI app without starting a server.
    client = TestClient(app)

    # Send an incomplete model spec missing required fields.
    response = client.post("/models/load", json={"model_id": "bad_model"})

    # Confirm FastAPI/Pydantic rejects the invalid request body.
    assert response.status_code == 422