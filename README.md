# MARP Inference Worker

MARP Inference Worker is a Python FastAPI service for running computer vision inference on image and video data.

The project is designed to support MARP image-recognition workflows where models may be loaded from a remote model server and inference jobs may be submitted from other applications. It provides an API layer for loading models, checking worker status, processing frames or video ranges, and returning standardized detection results.

## Project Goals

- Provide a reusable inference service for MARP computer vision workflows
- Support multiple model engines over time, including Ultralytics and custom Torch models
- Support model artifacts hosted on a remote model server
- Support single-frame and video-range inference jobs
- Provide clear API documentation through FastAPI Swagger/OpenAPI docs
- Keep the project modular so inference engines, model loading, and job handling can evolve independently

## API Documentation

When the service is running, FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The OpenAPI JSON schema is available at:

```text
http://127.0.0.1:8000/openapi.json
```

## Development Setup

Create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project with development dependencies:

```powershell
pip install -e ".[dev]"
```

Run tests:

```powershell
pytest
```

Run the development server:

```powershell
uvicorn marp_inference_worker.main:app --reload
```

Check the health endpoint:

```text
http://127.0.0.1:8000/health
```

## Current Status

This project is in early development. The initial FastAPI skeleton and health endpoint are implemented. Additional model loading, worker status, job handling, and inference functionality will be added incrementally.
