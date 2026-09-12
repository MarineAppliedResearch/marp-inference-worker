# MARP Inference Worker

<!-- marp:brand start -->
<!-- Canonical source: MARP/README.md. Do not edit this block in a component repository;
     edit it here and run `marp harness sync`. -->

<p align="center">
  <strong>MARP</strong> is the Marine Analysis and Reporting Platform. It carries an ocean
  survey from the video a dive brings home through to the science: annotation, review,
  machine-learning assistance, processing and reporting, on a platform an organisation
  hosts for itself.
</p>

<p align="center">
  <a href="https://github.com/MarineAppliedResearch/MARP">Umbrella</a> &middot;
  <a href="https://github.com/MarineAppliedResearch/MARP_API">API</a> &middot;
  <a href="https://github.com/MarineAppliedResearch/marp-video-player">Video player</a> &middot;
  <a href="https://github.com/MarineAppliedResearch/marp-inference-worker">Inference worker</a> &middot;
  <a href="https://github.com/MarineAppliedResearch/marp-jellyfin">Video server</a>
</p>
<!-- marp:brand end -->

<p align="center">
  <img alt="Project status" src="https://img.shields.io/badge/status-internal%20production%20%7C%20active%20development-05b9c8">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-service-009688?logo=fastapi&logoColor=white">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-NVIDIA%20required-76B900?logo=nvidia&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-a7ec35">
</p>

<table width="100%">
  <tr>
    <td align="center" bgcolor="#03101f">
      <br>
      <img src="assets/marp-logo.png" alt="MARP logo" width="430">
      <br><br>
    </td>
  </tr>
</table>

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

## Install

Install runtime dependencies from `pyproject.toml`:

```powershell
pip install -e .
```

Install development dependencies (includes runtime + `dev` extras from `pyproject.toml`):

```powershell
pip install -e ".[dev]"
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
