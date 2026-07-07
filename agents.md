# AGENTS.md

# MARP Inference Worker Agent Guide

## Project Overview

This project is the MARP Inference Worker.

The goal is to build an agnostic Python inference worker that can run on different computers and receive inference jobs from a future coordinator service. Each worker should be able to load and cache models from a model server, process individual frames or video ranges, run inference using different model engines, report status, and return standardized detection results to the coordinator.

The worker is not the coordinator. It should not decide which videos to process, how jobs are distributed across machines, or how detections become MARP database observations. The future coordinator will own those decisions and will save results to the existing MARP database API.

The worker’s core contract is:

```text
Coordinator sends model, input source, range, and inference settings.
Worker loads or reuses the requested model.
Worker runs inference.
Worker reports job status and returns standardized detections.
Coordinator saves or interprets the results.
```

## Current Development State

The project has been initialized as a Python FastAPI service.

Current completed setup:

```text
Project name: marp-inference-worker
Python version in venv: Python 3.10.11
Package style: pyproject.toml
API framework: FastAPI
Test framework: pytest
Repository: Git/GitHub
```

Current implemented behavior:

```text
GET /health
```

The `/health` endpoint returns:

```json
{"status": "ok"}
```

Swagger/OpenAPI documentation is available through FastAPI at:

```text
/docs
/openapi.json
/redoc
```

Current project structure is intentionally minimal. More modules should be added only as the design requires them.

## Development Philosophy

Isaac is the programmer. The LLM is the assistant.

Do not rush ahead and design large systems without checking the direction interactively. Work one focused step at a time. Before writing code, explain what the change is meant to accomplish and why it belongs in the current step.

Do not assume code that has not been shown in the current context. If exact code matters and you are unsure, ask Isaac to paste the file or relevant function before proposing a patch.

Prefer small commits and small working milestones.

Each new API endpoint should usually come with a test or a modification to an existing test. The tests are part of the API contract that the future coordinator will rely on.

## Intended Architecture

The first project is the inference worker only.

The worker should eventually support:

```text
Health/status reporting
Model loading
Model caching
Remote model artifacts from a model server
Single-frame inference
Video-range inference
HLS or stream-based video input
Job status tracking
Job cancellation
Standardized detection result output
HTTP callback or polling result delivery
Multiple inference engines
```

The worker should not directly own:

```text
MARP database writes
Observation approval logic
Survey-specific processing decisions
Job scheduling across multiple machines
Coordinator responsibilities
Human review workflow
```

A future coordinator service will:

```text
Choose jobs
Choose workers
Resolve model URLs
Resolve video stream URLs
Submit jobs to workers
Receive inference results
Save detections or observations through the database API
Track global job status
```

## FastAPI Usage Rules

Use FastAPI as the HTTP API layer.

Use route files for endpoints and keep route handlers thin. Route handlers should translate HTTP requests into calls to manager/service classes. They should not contain heavy model loading, video decoding, inference, or job-processing logic.

Use FastAPI routers for endpoint groups.

Expected route organization over time:

```text
api/
  app.py
  health_routes.py
  status_routes.py
  model_routes.py
  job_routes.py
```

`main.py` should remain thin. It should expose the app object that Uvicorn imports.

`api/app.py` should construct the FastAPI app and include routers.

## Testing Rules

Every new endpoint or changed endpoint contract should be accompanied by a test.

At minimum, endpoint tests should verify:

```text
The route exists
The expected status code is returned
The response shape is correct
Required fields exist
Invalid input fails correctly when relevant
```

Internal refactors do not always need new tests, but existing tests must continue to pass.

Run tests with:

```powershell
pytest
```

## Comment Style Rules

Every source file must begin with a file header comment.

The file header must include:

```text
File name
Date created
Author
A few lines explaining what the file does
The file’s purpose in the system
What type of code should belong in the file
```

Every function must have a function header comment. The function header should be concise, usually no more than 4 or 5 lines. It should describe:

```text
What the function does
Its inputs
Its outputs
How or when to use it
Why it exists, when useful
```

After every function definition line, include one blank line before the first comment or executable line.

When two functions appear one after another, place exactly two blank lines between them.

Every object, variable, and class should have a short comment above it explaining why that object is being declared. This should usually be one or two lines.

Inline comments should appear throughout important code paths to explain what the code or algorithm is doing. These comments should usually be one line, sometimes two.

Comments are for future developers and future LLMs to understand the code. Comments should be professional. They should not contain chat metadata, debug notes, or references to the current conversation.

## Current Commented Files

The following files have been rewritten or planned using the project comment style:

```text
src/marp_inference_worker/main.py
src/marp_inference_worker/api/app.py
src/marp_inference_worker/api/health_routes.py
tests/test_health.py
```

## Current `main.py` Intent

`main.py` is the application entry point.

It should:

```text
Import create_app
Create the module-level app object
Remain thin
Avoid route logic
Avoid model logic
Avoid job logic
```

## Current `api/app.py` Intent

`api/app.py` is the FastAPI application factory.

It should:

```text
Create the FastAPI application
Define service metadata shown in Swagger/OpenAPI
Register routers
Return the configured app
```

It should not contain actual endpoint behavior, model loading, video processing, or job management.

## Current `health_routes.py` Intent

`health_routes.py` owns lightweight health-check endpoints.

It should:

```text
Expose cheap health checks
Avoid expensive runtime checks
Avoid model loading
Avoid job execution
Avoid GPU-heavy checks
```

## Development Sequence

Current milestone:

```text
Minimal FastAPI skeleton
/health endpoint
pytest test for /health
GitHub repository setup
AGENTS.md project guidance
```

Next likely milestones:

```text
1. Add /status endpoint
2. Add worker identity and runtime state object
3. Add tests for /status
4. Add model spec schema
5. Add fake model loading
6. Add tests for model loading
7. Add fake job lifecycle
8. Add tests for job submission and status
9. Add model artifact download/cache behavior
10. Add mock inference engine
11. Add real Ultralytics engine
12. Add video/HLS decoding
```

Do not jump to later milestones unless Isaac explicitly asks.

## Model System Direction

Models will be hosted on a model server.

The coordinator will eventually send the worker a model specification containing a URL and metadata. The worker should download the model if needed, cache it locally, verify it when possible, and reuse it from cache when the same artifact is requested again.

The cache should be based on model identity plus artifact hash or immutable version, not only the model name.

The worker should eventually support multiple engine types, especially:

```text
Ultralytics models
Custom Torch models
Mock engine for testing
Possibly ONNX or TensorRT later
```

FastAPI is only the API layer. It should not constrain the inference engines.

## Result Semantics

The worker should return detections, not finalized MARP observations.

A worker result should say:

```text
This model produced this detection on this frame.
```

The coordinator or database layer should decide:

```text
Whether a detection becomes an observation
Whether detections are merged
Whether results require review
Whether a result is saved as a proposed observation
How detections map to database records
```

## Preferred Working Style with LLMs

Work interactively.

Do not provide huge project-wide rewrites unless asked.

For each step:

```text
Explain the purpose of the change
Show the specific file or patch
Wait for Isaac to apply/test when appropriate
Use tests as checkpoints
```

When suggesting code changes, explain why the change is needed before giving the patch.

If exact code context is missing, ask for the exact file or function before advising where to patch.

Use PowerShell commands when giving Windows shell instructions.

Use `vim` for Linux config edits unless Isaac specifies otherwise.

Final JSON outputs should be placed inside code blocks for easier copying.

## Git Notes

The project is tracked in git and pushed to GitHub.

Use small commits.

Suggested commit style:

```text
Initial FastAPI inference worker skeleton
Add worker status endpoint
Add model load schema
Add fake model manager
Add video range job lifecycle
```

Do not commit:

```text
.venv/
.env
model artifacts
runtime data
cache folders
logs
```

## Environment Notes

Use a project-specific virtual environment.

Current recommended Python for this project:

```text
Python 3.10.x
```

Reason: this project will likely use Torch and Ultralytics, and Python 3.10 is a conservative choice for dependency compatibility.

The venv is not a standalone executable bundle. Each worker computer still needs a compatible Python installation.

## Current Commands

Create and activate the venv on Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install development dependencies:

```powershell
pip install -e ".[dev]"
```

Run tests:

```powershell
pytest
```

Run the API server:

```powershell
uvicorn marp_inference_worker.main:app --reload
```

Check the API:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```
