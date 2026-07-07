# model_routes.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Model route definitions for the MARP Inference Worker API.
# This file contains endpoints for loading models and listing models currently
# known to the worker. Route handlers should stay thin and delegate model
# state, caching, and engine behavior to the model manager.

# FastAPI provides the router used to group model-management endpoints.
from fastapi import APIRouter

# The model manager owns temporary loaded-model state for this first version.
from marp_inference_worker.models import model_manager

# ModelSpec validates the public model-loading request body.
from marp_inference_worker.models.model_spec import ModelSpec


# Router for model-management endpoints.
# Keeping model routes separate makes the model API easy to extend.
router = APIRouter(prefix="/models", tags=["models"])


# load_model()
# Accepts a model spec and records it as loaded by the worker.
# Inputs: ModelSpec request body.
# Output: status string and the loaded model spec.
# TODO: This is fake loading until real cache/download/engine loading is added.
@router.post("/load")
async def load_model(model_spec: ModelSpec) -> dict[str, object]:

    # Ask the manager to record the model and compute cache metadata.
    load_result = model_manager.load_model(model_spec)

    # Return a simple response contract for the coordinator and tests.
    return {
        "status": "loaded",
        **load_result,
    }


# get_loaded_models()
# Returns the models currently recorded as loaded by this worker.
# Inputs: none.
# Output: dictionary containing a list of loaded model specs.
# TODO: Later this should report real engine-loaded models and cache state.
@router.get("/loaded")
async def get_loaded_models() -> dict[str, list[ModelSpec]]:

    # Ask the manager for loaded models so route code stays thin.
    loaded_models = model_manager.get_loaded_models()

    # Wrap the list in an object so the response can grow without breaking shape.
    return {"loaded_models": loaded_models}