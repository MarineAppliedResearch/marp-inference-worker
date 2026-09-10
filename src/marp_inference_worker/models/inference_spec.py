# inference_spec.py
# Created: 2026-07-11
# Author: Isaac Travers
#
# Inference request schema definitions for the MARP Inference Worker.
# This file validates public inference request payloads for API routes.
# Keep coordinator-facing inference contracts in this module.

# Pydantic provides request validation and field constraints.
from pydantic import BaseModel, Field


# FrameInferenceRequest
# Public request schema for single-frame inference.
# Inputs: model ID, frame/image source, confidence threshold, and image flag.
# Output: validated inference request object.
# Use this for POST /infer/frame request validation.
class FrameInferenceRequest(BaseModel):

    # ID of a previously loaded model.
    model_id: str

    # Image source path or URL readable by the worker.
    image_source: str

    # Confidence threshold constrained to valid probability range.
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)

    # Optional flag requesting an annotated image payload in JSON output.
    return_annotated_image: bool = False
