from fastapi import FastAPI

from marp_inference_worker.api.health_routes import router as health_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="MARP Inference Worker",
        version="0.1.0",
        description="Agnostic inference worker for frame and video-range model inference.",
    )

    app.include_router(health_router)

    return app