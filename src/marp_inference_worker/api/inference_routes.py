# inference_routes.py
# Created: 2026-07-08
# Author: Isaac Travers
#
# Inference route definitions for the MARP Inference Worker API.
# This file contains endpoints for running model inference against inputs
# available to the worker. Route handlers should stay thin and delegate model
# lookup, engine selection, and result normalization to the model manager.

# FastAPI provides route grouping, HTTP error responses, query validation, and raw responses.
from fastapi import APIRouter, HTTPException, Query, Response

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
# Output: normalized detections, optional annotated image, and request metadata.
# Use this for initial single-frame testing before video or tracking jobs.
@router.post("/frame")
async def infer_frame(request: FrameInferenceRequest) -> dict[str, object]:

    # Run frame inference and translate expected failures into API responses.
    try:
        inference_result = model_manager.infer_frame(
            model_id=request.model_id,
            image_source=request.image_source,
            confidence=request.confidence,
            return_annotated_image=request.return_annotated_image,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    # Pull detections out of the manager result so the route can add metadata.
    detections = inference_result["detections"]

    # Build a stable response object that can grow later without breaking shape.
    response_body = {
        "model_id": request.model_id,
        "image_source": request.image_source,
        "confidence": request.confidence,
        "return_annotated_image": request.return_annotated_image,
        "detection_count": len(detections),
        "detections": detections,
    }

    # Add annotated image fields only when the engine returned them.
    if "annotated_image_base64" in inference_result:
        response_body["annotated_image_format"] = inference_result["annotated_image_format"]
        response_body["annotated_image_base64"] = inference_result["annotated_image_base64"]

    # Return the final API response.
    return response_body


# infer_frame_image()
# Runs inference on one image source and returns an annotated JPG directly.
# Inputs: model ID, image source query parameter, and confidence threshold.
# Output: image/jpeg response bytes.
# Use this when a browser should display the annotated result directly.
@router.get("/frame/image")
async def infer_frame_image(
    model_id: str,
    image_source: str,
    confidence: float = Query(default=0.25, ge=0.0, le=1.0),
) -> Response:

    # Run image inference and translate expected failures into API responses.
    try:
        image_bytes = model_manager.infer_frame_image(
            model_id=model_id,
            image_source=image_source,
            confidence=confidence,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    # Return raw JPG bytes so browsers display the image directly.
    return Response(
        content=image_bytes,
        media_type="image/jpeg",
    )