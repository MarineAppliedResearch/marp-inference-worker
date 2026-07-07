# main.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Application entry point for the MARP Inference Worker API.
# This file exposes the FastAPI application instance used by Uvicorn.
# Keep this file thin. API construction should live in api/app.py, while
# route definitions and worker logic should live in their own modules.

from marp_inference_worker.api.app import create_app


# Create the FastAPI application instance that Uvicorn imports and serves.
# This object should remain module-level so ASGI servers can discover it.
app = create_app()