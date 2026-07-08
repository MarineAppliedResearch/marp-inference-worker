# AGENTS.md

# MARP Inference Worker Agent Guide

## Purpose of This File

This file is the project handoff guide for LLM assistants working with Isaac on the MARP Inference Worker. It should give a new LLM enough context to continue development without needing the full prior chat history.

Isaac is the programmer. The LLM is the programming assistant. Work interactively, make small changes, explain why each change is being made, and avoid large speculative rewrites.

## Project Overview

MARP Inference Worker is a Python FastAPI service for running computer vision and other model inference workloads on one worker machine.

The long-term system will have multiple inference workers running on different computers. A future coordinator service will decide which worker gets which job, send model and input information to the worker, receive standardized inference results, and save or interpret those results through the MARP database API.

The worker is intentionally not the coordinator.

The worker should eventually:

```text
Report worker health and status
Load model artifacts supplied by a coordinator or model server
Cache model artifacts locally
Dispatch models to the correct inference engine
Run inference on individual frames
Run inference on video ranges or streams
Track local jobs
Return standardized detections/results to the coordinator
Support multiple model engines over time
```

The worker should not own:

```text
Global job scheduling
MARP database writes
Observation approval logic
Survey-specific decision rules
Human review workflow
Final conversion of detections into database observations
```

## Current Project State

The project is initialized and working as a FastAPI service.

Current environment:

```text
Project name: marp-inference-worker
Python version used in venv: Python 3.10.11
Package configuration: pyproject.toml
API framework: FastAPI
Test framework: pytest
Repository: Git/GitHub
Development shell: Windows PowerShell
```

Current implemented endpoints:

```text
GET  /health
GET  /status
POST /models/load
GET  /models/loaded
```

FastAPI documentation is available when the server is running:

```text
/docs
/openapi.json
/redoc
```

Current test state at handoff:

```text
pytest passes with 10 tests
There is one non-blocking FastAPI/Starlette TestClient warning about httpx/httpx2
```

The warning is not currently being addressed.

## Current API Behavior

### GET /health

Returns:

```json
{"status": "ok"}
```

This is a cheap health check only. Do not add expensive GPU, model, or job checks here.

### GET /status

Currently returns a static worker status:

```json
{
  "worker_id": "dev-worker",
  "status": "idle",
  "version": "0.1.0",
  "active_jobs": 0,
  "loaded_models": []
}
```

Later, this should read from a real worker state object, job manager, and model manager.

### POST /models/load

Accepts a model spec. The worker now:

```text
Validates the model spec
Ensures the artifact is present in the local cache
Selects the requested engine through the engine registry
Loads the cached artifact through that engine
Stores the model spec in the loaded-model registry
Returns model, cache, and engine metadata
```

At the current stage, the only registered engine is `mock`.

### GET /models/loaded

Returns loaded model specs currently stored in the in-memory model manager registry.

Current loaded state is not persistent. It resets when the process restarts.

## Current Model Spec

The current model spec is defined in:

```text
src/marp_inference_worker/models/model_spec.py
```

Conceptually:

```json
{
  "model_id": "demo_mock",
  "engine": "mock",
  "model_arch": "mock_detector",
  "task": "detect",
  "artifact": {
    "url": "test_models/demo_mock.pt",
    "format": "mock",
    "sha256": null
  },
  "load_settings": {
    "device": "auto"
  },
  "labels": [
    {
      "class_id": 0,
      "class_name": "bat star",
      "external_id": null
    }
  ]
}
```

Important design decisions:

```text
artifact.url is intentionally used as a generic artifact locator.
If it starts with http:// or https://, it is treated as a remote URL.
Otherwise, it is treated as a local file path.
```

This allows local development with local model files before the model server is fully integrated.

`labels` maps model class IDs to class names. For example, class ID `0` can map to `"bat star"`. The worker should return class names when possible, but the coordinator/database layer decides how those labels map to database species or observations.

`load_settings.device` currently defaults to `"auto"`. Later this may control CPU/GPU device selection such as `"cpu"` or `"cuda:0"`.

## Current Model Cache Behavior

Model cache logic lives in:

```text
src/marp_inference_worker/models/model_cache.py
```

Current cache root:

```text
models/cache/
```

This path is ignored by git because `.gitignore` ignores `models/`.

Current behavior:

```text
If artifact.url is http or https:
  download with httpx into models/cache/<cache_key>/<filename>

If artifact.url is anything else:
  treat it as a local file path
  copy it into models/cache/<cache_key>/<filename>
```

Current cache key behavior:

```text
If artifact.sha256 is present:
  cache key = model_id + "_" + sha256

If artifact.sha256 is missing:
  cache key = model_id
```

Hash verification is not implemented yet. Existing cached artifacts are reused based on file existence only.

Known TODOs:

```text
Add sha256 verification before reusing cached artifacts
Add safer path/filename validation
Add auth headers/retries/progress for remote downloads
Possibly move cache root into config later
```

## Current Engine Layer

The engine layer was added to keep model-manager logic independent from specific ML runtimes.

Current files:

```text
src/marp_inference_worker/engines/base_engine.py
src/marp_inference_worker/engines/mock_engine.py
src/marp_inference_worker/engines/engine_registry.py
```

Current pattern:

```text
BaseEngine defines the shared interface.
MockEngine implements that interface without real inference.
engine_registry maps public engine names to engine instances.
model_manager asks engine_registry for the requested engine.
```

Current registered engine:

```text
mock
```

Planned engines:

```text
ultralytics
custom_torch
torchscript
possibly onnx
possibly tensorrt
```

Important distinction:

```text
Engine = runtime/library adapter that knows how to load and run a model.
Model = trained artifact plus metadata.
```

Examples:

```text
UltralyticsEngine
  handles Ultralytics-compatible YOLO models such as YOLOv8, YOLO11, YOLO28, and custom Ultralytics .pt files.

CustomTorchEngine
  will later handle Isaac's own PyTorch models, but this needs packaging rules because raw .pt checkpoints may require architecture code and preprocessing/postprocessing logic.

TorchScriptEngine
  may later handle exported TorchScript models that are easier to load generically.
```

## Current Model Manager

Model manager lives in:

```text
src/marp_inference_worker/models/model_manager.py
```

Current responsibilities:

```text
Ensure artifact is cached
Resolve cached artifact path
Select the engine
Ask the engine to load the model
Record the model spec as loaded in memory
Return model/cache/engine metadata
```

The loaded-model registry is currently an in-memory dictionary. It is temporary.

Important TODO:

```text
Replace fake loaded-model state with real loaded model handles.
The worker will eventually need loaded handles available for inference calls.
```

## Current Project Structure

Approximate structure after current work:

```text
marp-inference-worker/
  README.md
  AGENTS.md
  pyproject.toml
  .gitignore
  src/
    marp_inference_worker/
      __init__.py
      main.py
      api/
        __init__.py
        app.py
        health_routes.py
        status_routes.py
        model_routes.py
      engines/
        __init__.py
        base_engine.py
        mock_engine.py
        engine_registry.py
      models/
        __init__.py
        model_spec.py
        model_cache.py
        model_manager.py
  tests/
    test_health.py
    test_status.py
    test_models.py
    test_model_cache.py
    test_engine_registry.py
```

## FastAPI Design Rules

Use FastAPI only as the HTTP API layer.

Use route files for endpoint groups:

```text
api/
  health_routes.py
  status_routes.py
  model_routes.py
  job_routes.py
```

Route handlers should stay thin. They should:

```text
Validate request bodies through Pydantic schemas
Call manager/service classes
Return API responses
```

Route handlers should not contain:

```text
model download logic
engine loading logic
video decoding
job execution
heavy inference code
database-specific logic
```

`main.py` should remain thin and expose the module-level `app` object for Uvicorn.

`api/app.py` should create the FastAPI app, set metadata, and include routers.

## Testing Rules

Every new endpoint or changed endpoint contract should be accompanied by a test or a modification to an existing test.

Tests should verify, as appropriate:

```text
The route exists
Expected status code
Expected response shape
Required fields exist
Invalid input fails correctly
State changes are visible through API
```

For model/cache tests, avoid relying on external network access. Mock remote downloads or use temporary local files.

The project currently uses pytest:

```powershell
pytest
```

When Isaac says a change is “done,” assume he has edited it and run pytest unless he says otherwise.

## Code Comment Style Rules

Every source file must begin with a file header comment.

The file header must include:

```text
File name
Date created
Author
A few lines explaining what the file does
The file's purpose in the system
What type of code belongs in the file
```

Every function must have a concise function header comment, usually no more than 4 or 5 lines. It should describe:

```text
What the function does
Inputs
Outputs
How or when to use it
Why it exists, when useful
```

After every function definition line, include one blank line before the first comment or executable line.

When two functions appear one after another, place exactly two blank lines between them.

Every class must have a class header comment. After every class definition line, include one blank line before the first class body comment or code.

Every object, variable, and class should have a short comment above it explaining why it is declared. This is usually one or two lines.

Imports should be commented by import group. Do not necessarily comment every individual import if a group comment is clearer.

Example:

```python
# FastAPI provides the router used to group model-management endpoints.
from fastapi import APIRouter

# Model manager owns model loading and loaded-model state.
from marp_inference_worker.models import model_manager
```

Inline comments should appear throughout important code paths to explain what the code or algorithm is doing. These comments should usually be one line, sometimes two.

Comments are for future developers and future LLMs. They should be professional. Do not include chat metadata, debug chatter, or conversational rationale in code comments.

Patch rationale belongs in the assistant response before code, not in code comments.

## Patch Delivery Rules for LLMs

Before giving a patch, explain briefly what the change does and why it is being made. Do not put that explanatory rationale into code comments.

Isaac prefers patches to be specific and easy to apply.

For a whole-function replacement:

```text
Give the new full function only.
Do not include the old full function.
```

For a partial-function replacement:

```text
Give exact old code to find.
Give exact replacement code.
Include enough surrounding context so Isaac knows where it goes.
```

If adding a new import, state where it belongs and show the surrounding import group.

If adding a new file, provide the full file content.

If exact current code matters and is not known, ask Isaac to paste the exact file or function before advising where to patch.

## Working Style with Isaac

Isaac is the programmer. The LLM is the assistant.

Do not race ahead. Do not design huge systems without checking direction.

Work one small milestone at a time.

A useful step pattern is:

```text
1. Explain the next design decision.
2. Confirm the chosen direction when needed.
3. Explain what the patch changes and why.
4. Provide the exact code patch.
5. Ask Isaac to run pytest or manually test.
6. Use test results as the checkpoint.
```

Do not assume code that has not been shown in the current context.

Use PowerShell commands for Windows shell instructions.

Use `vim` for Linux config edits unless Isaac specifies otherwise.

Final JSON outputs should be in code blocks for easier copying.

Keep responses concise. Isaac explicitly asked for less overexplaining. Give enough context to reason about the change, but avoid broad surveys unless he asks.

## Git Notes

Use small commits.

Current useful commit milestones already completed or expected around this handoff:

```text
Initial FastAPI inference worker skeleton
Add static worker status endpoint
Add fake model loading endpoints
Add model load settings schema
Add model cache path planning
Add model artifact download/cache path
Support local model artifact caching
Add engine registry and mock engine
```

Do not commit:

```text
.venv/
.env
model artifacts
runtime data
cache folders
logs
models/cache/
test_models/ real model files unless explicitly intended
```

## Environment Notes

Use a project-specific virtual environment.

Recommended Python:

```text
Python 3.10.x
```

Reason: this project will likely use Torch and Ultralytics, and Python 3.10 is a conservative compatibility choice.

Create and activate the venv on Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install project dependencies:

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
http://127.0.0.1:8000/status
http://127.0.0.1:8000/docs
```

## Current Dependencies

Core dependencies currently include:

```text
fastapi
uvicorn[standard]
pydantic
httpx
```

Development dependencies include:

```text
pytest
httpx
ruff
```

The project has not yet added Ultralytics, Torch, OpenCV, PyAV, or other heavy ML/video dependencies.

## Current Next Step

The likely next implementation step is:

```text
Add a real UltralyticsEngine.
```

Recommended direction:

```text
1. Add ultralytics dependency carefully.
2. Register "ultralytics" in engine_registry.
3. Implement UltralyticsEngine.load_model().
4. Test loading a local real Ultralytics .pt artifact copied through artifact.url.
5. Keep inference endpoint separate until model loading is confirmed.
```

Do not start with fake `/infer/frame` unless Isaac changes direction. Isaac decided a fake inference endpoint would not be useful before real model loading works.

After Ultralytics loading works, the next likely feature is:

```text
POST /infer/frame
```

That endpoint should accept a directly supplied frame or image reference and return normalized model detections.

Later milestones:

```text
Frame inference with real Ultralytics model
Result normalization for detection outputs
Video/HLS range job endpoint
Local job manager
Job status and cancellation
Callback or polling result delivery
Custom PyTorch engine support
Hash verification for cached artifacts
Worker status backed by real model/job state
```

## Important Model and Result Semantics

The worker should return model detections/results, not finalized MARP observations.

The worker result means:

```text
This model produced this output for this input.
```

The coordinator/database layer decides:

```text
Whether a detection becomes an observation
Whether detections are merged
Whether results require human review
How class labels map to MARP database records
How inference results are saved
```

## Historical Context

The project was started because Isaac has existing MARP/ML inference scripts, including a large inference script, but they are not structured for:

```text
multiple jobs
multiple workers
multiple model engines
clean API usage
model loading from a model server
future distributed processing
```

The architecture direction is to avoid another large monolithic script. The project is being built as a modular API service with small files, tests, comments, and replaceable components.

Older design decisions:

```text
The coordinator will eventually talk to the MARP database API, model server, Jellyfin/video server, and multiple workers.
The worker should eventually be able to pull video or HLS streams directly from the video server.
For high-throughput video work, the worker should process video ranges locally rather than receive one HTTP request per frame.
Single-frame inference is still needed for testing, GUI tools, and debugging.
```

Keep this distinction clear:

```text
Coordinator = decides what should run and stores/interprets results.
Worker = loads models, processes assigned inputs, reports local state, returns detections/results.
```

```

If you do not know the code in a file, do not assume, just ask for it before suggesting an edit. 
when you tell me to add a new file, just give me the powershell command to run from the base of the project.

Current milestone:
  1. Load a real Ultralytics YOLO model.
  2. Store the loaded model handle in the UltralyticsEngine.
  3. Add frame inference through the same engine.
  4. Return normalized detections from one image/frame.

Not yet:
  Training jobs
  Job manager
  GPU resource endpoint
  ByteTrack
  Video range processing
  Coordinator behavior
