# inference_routes.py
# Created: 2026-07-08
# Author: Isaac Travers
#
# Inference route definitions for the MARP Inference Worker API.
# This file contains endpoints for running model inference against inputs
# available to the worker. Route handlers should stay thin and delegate model
# lookup, engine selection, and result normalization to the model manager.

# FastAPI provides route grouping and HTTP error responses.
from fastapi import APIRouter, HTTPException

# FrameInferenceRequest validates the public frame-inference request body.
from marp_inference_worker.models.inference_spec import FrameInferenceRequest

# Model manager owns loaded model state and dispatches inference to engines.
from marp_inference_worker.models import model_manager


# Router for inference endpoints.
# Keeping inference routes separate avoids mixing model loading with execution.
router = APIRouter(prefix="/infer", tags=["inference"])


# infer_frame()
# Runs inference on one image source using an already-loaded model.
# Inputs: FrameInferenceRequest request body.
# Output: normalized detections and basic request metadata.
# Use this for initial single-frame testing before video or tracking jobs.
@router.post("/frame")
async def infer_frame(request: FrameInferenceRequest) -> dict[str, object]:

    # Run frame inference and translate expected failures into API responses.
    try:
        detections = model_manager.infer_frame(
            model_id=request.model_id,
            image_source=request.image_source,
            confidence=request.confidence,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    # Return a stable response object that can grow later without breaking shape.
    return {
        "model_id": request.model_id,
        "image_source": request.image_source,
        "confidence": request.confidence,
        "detection_count": len(detections),
        "detections": detections,
    }