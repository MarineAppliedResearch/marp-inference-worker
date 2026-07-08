# test_inference.py
# Created: 2026-07-08
# Author: Isaac Travers
#
# Tests for the MARP Inference Worker inference API routes.
# This file verifies that inference endpoints expose the expected API contract
# without requiring real model files, GPU hardware, or Ultralytics inference.

# TestClient allows tests to call the FastAPI app without running Uvicorn.
from fastapi.testclient import TestClient

# The FastAPI app is tested through its public HTTP route contract.
from marp_inference_worker.main import app

# Model manager is monkeypatched so route tests do not run real inference.
from marp_inference_worker.models import model_manager


# Client used by inference route tests.
client = TestClient(app)


# test_infer_frame_returns_detections()
# Verifies that POST /infer/frame returns normalized detection results.
# Inputs: monkeypatched model manager and JSON request body.
# Output: pytest pass/fail result based on response contract.
# Use this to protect the public frame-inference API shape.
def test_infer_frame_returns_detections(monkeypatch) -> None:

    # fake_infer_frame()
    # Replaces real model inference during this route test.
    # Inputs: model ID, image source, confidence threshold, and image flag.
    # Output: fake inference result dictionary.
    # Use this so the API route can be tested without loading YOLO.
    def fake_infer_frame(
        model_id: str,
        image_source: str,
        confidence: float,
        return_annotated_image: bool = False,
    ) -> dict[str, object]:

        # Confirm the route passes the request values into the manager.
        assert model_id == "demo_yolo"
        assert image_source == "C:\\test_frames\\frame_001.jpg"
        assert confidence == 0.25
        assert return_annotated_image is False

        # Return a representative inference result.
        return {
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "Bat star",
                    "confidence": 0.9,
                    "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
                    "bbox_xyxyn": [0.1, 0.2, 0.3, 0.4],
                }
            ]
        }

    # Patch the model manager so this test only checks route behavior.
    monkeypatch.setattr(model_manager, "infer_frame", fake_infer_frame)

    # Call the public frame-inference endpoint with a valid request body.
    response = client.post(
        "/infer/frame",
        json={
            "model_id": "demo_yolo",
            "image_source": "C:\\test_frames\\frame_001.jpg",
            "confidence": 0.25,
        },
    )

    # Confirm the route succeeds.
    assert response.status_code == 200

    # Parse the JSON response for contract checks.
    response_body = response.json()

    # Confirm request metadata is echoed in the response.
    assert response_body["model_id"] == "demo_yolo"
    assert response_body["image_source"] == "C:\\test_frames\\frame_001.jpg"
    assert response_body["confidence"] == 0.25
    assert response_body["return_annotated_image"] is False

    # Confirm detection count is derived from the returned detections list.
    assert response_body["detection_count"] == 1

    # Confirm no annotated image is returned unless requested.
    assert "annotated_image_format" not in response_body
    assert "annotated_image_base64" not in response_body

    # Confirm the returned detection includes pixel and normalized boxes.
    assert response_body["detections"] == [
        {
            "class_id": 0,
            "class_name": "Bat star",
            "confidence": 0.9,
            "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
            "bbox_xyxyn": [0.1, 0.2, 0.3, 0.4],
        }
    ]


# test_infer_frame_rejects_confidence_below_zero()
# Verifies that confidence must be greater than or equal to zero.
# Inputs: JSON request body with invalid confidence.
# Output: pytest pass/fail result based on validation failure.
# Use this to protect the FrameInferenceRequest confidence bounds.
def test_infer_frame_rejects_confidence_below_zero() -> None:

    # Call the endpoint with an invalid low confidence value.
    response = client.post(
        "/infer/frame",
        json={
            "model_id": "demo_yolo",
            "image_source": "C:\\test_frames\\frame_001.jpg",
            "confidence": -0.1,
        },
    )

    # Confirm Pydantic validation rejects the request before inference runs.
    assert response.status_code == 422


# test_infer_frame_rejects_confidence_above_one()
# Verifies that confidence must be less than or equal to one.
# Inputs: JSON request body with invalid confidence.
# Output: pytest pass/fail result based on validation failure.
# Use this to protect the FrameInferenceRequest confidence bounds.
def test_infer_frame_rejects_confidence_above_one() -> None:

    # Call the endpoint with an invalid high confidence value.
    response = client.post(
        "/infer/frame",
        json={
            "model_id": "demo_yolo",
            "image_source": "C:\\test_frames\\frame_001.jpg",
            "confidence": 1.1,
        },
    )

    # Confirm Pydantic validation rejects the request before inference runs.
    assert response.status_code == 422


# test_infer_frame_returns_400_when_model_is_not_loaded()
# Verifies that model-manager ValueError failures become HTTP 400 responses.
# Inputs: monkeypatched model manager and valid JSON request body.
# Output: pytest pass/fail result based on route error handling.
# Use this to protect unloaded-model error behavior.
def test_infer_frame_returns_400_when_model_is_not_loaded(monkeypatch) -> None:

    # fake_infer_frame()
    # Simulates manager failure when the requested model is not loaded.
    # Inputs: model ID, image source, confidence threshold, and image flag.
    # Output: raises ValueError.
    # Use this so the route error handling can be tested directly.
    def fake_infer_frame(
        model_id: str,
        image_source: str,
        confidence: float,
        return_annotated_image: bool = False,
    ) -> dict[str, object]:

        # Raise the same error type used by model_manager.infer_frame().
        raise ValueError(f"Model is not loaded: {model_id}")

    # Patch the manager to simulate an unloaded model.
    monkeypatch.setattr(model_manager, "infer_frame", fake_infer_frame)

    # Call the endpoint with a valid request that triggers the simulated failure.
    response = client.post(
        "/infer/frame",
        json={
            "model_id": "missing_model",
            "image_source": "C:\\test_frames\\frame_001.jpg",
            "confidence": 0.25,
        },
    )

    # Confirm manager ValueError is translated into a bad request.
    assert response.status_code == 400

    # Confirm the response includes the manager error text.
    assert response.json()["detail"] == "Model is not loaded: missing_model"