# app.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# FastAPI application factory for the MARP Inference Worker API.
# This file creates and configures the API application object, including
# service metadata and route registration. Keep route implementation,
# model loading, job management, and inference logic in separate modules.

# FastAPI creates the application object and route registry.
from fastapi import FastAPI

# Route modules are imported here so the application factory can register them.
from marp_inference_worker.api.health_routes import router as health_router
from marp_inference_worker.api.status_routes import router as status_router
from marp_inference_worker.api.model_routes import router as model_router

# Inference routes expose model execution endpoints.
from marp_inference_worker.api.inference_routes import router as inference_router


# create_app()
# Builds and returns the FastAPI application used by the ASGI server.
# Inputs: none.
# Output: configured FastAPI application instance.
# Use this when the service starts so app construction stays centralized.
def create_app() -> FastAPI:

    # Create the main FastAPI app and define metadata shown in Swagger/OpenAPI.
    app = FastAPI(
        title="MARP Inference Worker",
        version="0.1.0",
        description="Agnostic inference worker for frame and video-range model inference.",
    )

    # Register health routes so external tools can verify the worker is alive.
    app.include_router(health_router)

    # Register status routes so coordinators can inspect worker availability.
    app.include_router(status_router)

    # Register model routes so coordinators can load and inspect worker models.
    app.include_router(model_router)

    # Register inference routes so loaded models can run against worker-local inputs.
    app.include_router(inference_router)

    # Return the configured app object to main.py for Uvicorn/ASGI discovery.
    return app