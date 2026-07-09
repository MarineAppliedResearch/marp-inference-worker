# AGENTS.md

# MARP Inference Worker Agent Guide

## Purpose of This File

This file is the project handoff guide for LLM assistants working with Isaac on the MARP Inference Worker. It should give a new LLM enough context to continue development without needing the full prior chat history.

Isaac is the programmer. The LLM is the programming assistant. Work interactively, make small changes, explain why each change is being made, and avoid large speculative rewrites.

The assistant should preserve the current architecture, ask for exact file contents when needed, and work one checkpoint at a time.

## Project Overview

MARP Inference Worker is a Python FastAPI service for running model workloads on one worker machine.

The project name is currently `marp-inference-worker`, but conceptually this worker will eventually support both inference and training. It is still called an inference worker for now.

The long-term system will have multiple workers running on different computers. A future coordinator service will decide which worker gets which job, send model and input information to the worker, receive standardized inference or training results, and save or interpret those results through MARP services.

The worker is intentionally not the coordinator.

The worker should eventually:

```text
Report worker health and status
Report local system resources and CUDA/GPU availability
Load model artifacts supplied by a coordinator or model server
Cache model artifacts locally
Dispatch models to the correct engine
Run inference on individual visual frames
Run inference on video ranges or streams
Support image URL and local file inputs
Track local long-running jobs
Train models when given training jobs
Return standardized model results to the coordinator
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

Keep this distinction clear:

```text
Coordinator = decides what should run and stores/interprets results.
Worker = loads models, processes assigned inputs, reports local state, returns model results.
```

## Current Environment

```text
Project name: marp-inference-worker
Python version used in venv: Python 3.10.11
Package configuration: pyproject.toml
API framework: FastAPI
Test framework: pytest
Repository: Git/GitHub
Development shell: Windows PowerShell
Editor: Visual Studio Code
```

Recommended Python:

```text
Python 3.10.x
```

Reason: this project uses or will likely use Torch, Ultralytics, OpenCV, and other ML/video libraries where Python 3.10 is a conservative compatibility choice.

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

Core dependencies now include at least:

```text
fastapi
uvicorn[standard]
pydantic
httpx
ultralytics
```

Development dependencies include:

```text
pytest
httpx
ruff
```

Ultralytics brings in its own ML stack dependencies such as Torch and OpenCV as needed.

## Current Test State

Current test checkpoint:

```text
pytest passes with 17 tests
There is one non-blocking FastAPI/Starlette TestClient warning about httpx/httpx2
```

The warning is not currently being addressed.

## Current Implemented Endpoints

```text
GET  /health
GET  /status
POST /models/load
GET  /models/loaded
POST /infer/frame
GET  /infer/frame/image
```

FastAPI documentation is available when the server is running:

```text
/docs
/openapi.json
/redoc
```

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

Later, this should read from a real worker state object, job manager, model manager, and resource monitor.

### POST /models/load

Accepts a model spec. The worker now:

```text
Validates the model spec
Ensures the artifact is present in the local cache
Selects the requested engine through the engine registry
Loads the cached artifact through that engine
Stores the model spec in the loaded-model registry
For Ultralytics, stores the real YOLO handle inside UltralyticsEngine
Returns model, cache, and engine metadata
```

Currently registered engines:

```text
mock
ultralytics
```

Example real Ultralytics model-load body used successfully during development:

```json
{
  "model_id": "demo_yolo",
  "engine": "ultralytics",
  "model_arch": "yolo",
  "task": "detect",
  "artifact": {
    "url": "C:\\Users\\isaac\\Documents\\Workspace\\MARE_Video_Annotations\\models\\CAMPA2025_group1_candidate5\\mixed\\weights\\CAMPA2025_group1_candidate5.pt",
    "format": "ultralytics_pt",
    "sha256": null
  },
  "load_settings": {
    "device": "auto"
  },
  "labels": []
}
```

Successful response shape includes:

```json
{
  "status": "loaded",
  "model": {},
  "cache": {
    "cache_key": "demo_yolo",
    "artifact_path": "models\\cache\\demo_yolo\\CAMPA2025_group1_candidate5.pt",
    "is_cached": true,
    "cache_action": "copied"
  },
  "engine": {
    "engine_name": "ultralytics",
    "model_id": "demo_yolo",
    "model_loaded": true,
    "artifact_path": "models\\cache\\demo_yolo\\CAMPA2025_group1_candidate5.pt",
    "loaded_model_type": "YOLO",
    "loaded_model_count": 1
  }
}
```

Loaded state is in memory and resets when the process restarts. The model must be loaded again after restarting the server.

### GET /models/loaded

Returns loaded model specs currently stored in the in-memory model manager registry.

### POST /infer/frame

Runs inference on one visual frame/image source using an already-loaded model.

Frame here means one still visual frame, not necessarily a frame extracted from a video. It can be a local image path or an HTTP/HTTPS image URL.

Request schema currently includes:

```json
{
  "model_id": "demo_yolo",
  "image_source": "C:\\path\\to\\frame.jpg",
  "confidence": 0.25,
  "return_annotated_image": false
}
```

Response includes:

```json
{
  "model_id": "demo_yolo",
  "image_source": "C:\\path\\to\\frame.jpg",
  "confidence": 0.25,
  "return_annotated_image": false,
  "detection_count": 4,
  "detections": [
    {
      "class_id": 5,
      "class_name": "Leather star",
      "confidence": 0.9078916311264038,
      "bbox_xyxy": [644.1, 656.3, 787.7, 769.7],
      "bbox_xyxyn": [0.5, 0.6, 0.6, 0.7]
    }
  ]
}
```

If `return_annotated_image` is true, the response also includes:

```json
{
  "annotated_image_format": "jpg",
  "annotated_image_base64": "..."
}
```

This base64 JSON response is useful for API clients but not convenient for direct browser viewing.

### GET /infer/frame/image

Runs inference on one visual frame/image source and returns an annotated image directly as `image/jpeg`.

This route is meant for quick browser testing. Paste a URL into the browser and see the annotated result.

Example:

```text
http://127.0.0.1:8000/infer/frame/image?model_id=demo_yolo&image_source=C:/Users/isaac/Documents/Workspace/MARE_Video_Annotations/datasets/CAMPA2025_group1_candidate2/eval/images/20240727_185645%20Fwd.mp4_obs_175495_291051_291052_291119_291120_291570_291571_291638_291639_frame_29319.jpg&confidence=0.25
```

Use forward slashes in Windows paths for browser URLs. Spaces should be encoded as `%20`.

Current route returns JPG regardless of input image type. Isaac raised that later it may be better to support:

```text
output_format=auto
output_format=jpg
output_format=png
```

This has not been implemented yet.

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

`labels` maps model class IDs to class names. The worker should return class names when possible. The coordinator/database layer decides how those labels map to database species or observations.

`load_settings.device` currently defaults to `"auto"`. Later this may control CPU/GPU device selection such as `"cpu"` or `"cuda:0"`.

## Current Inference Spec

The current frame inference request schema lives in:

```text
src/marp_inference_worker/models/inference_spec.py
```

It currently contains `FrameInferenceRequest` with:

```text
model_id: str
image_source: str
confidence: float = 0.25 with bounds 0.0 to 1.0
return_annotated_image: bool = False
```

`image_source` is intentionally generic. It can be a local file path or HTTP/HTTPS image URL. URL image handling is now done by the shared frame source input utility, not directly by Ultralytics.

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

One prior test failure happened because a cache test reused an old model ID. The fix was to use `uuid4().hex` in the test model ID so the cache test does not accidentally reuse a previous cache folder.

## Current Engine Layer

The engine layer keeps model-manager logic independent from specific ML runtimes.

Current files:

```text
src/marp_inference_worker/engines/base_engine.py
src/marp_inference_worker/engines/mock_engine.py
src/marp_inference_worker/engines/ultralytics_engine.py
src/marp_inference_worker/engines/engine_registry.py
```

Current pattern:

```text
BaseEngine defines the shared load interface.
MockEngine implements a lightweight engine without real inference.
UltralyticsEngine implements real YOLO model loading and frame prediction.
engine_registry maps public engine names to engine instances.
model_manager asks engine_registry for the requested engine.
```

Current registered engines:

```text
mock
ultralytics
```

Planned engines:

```text
custom_torch
torchscript
onnx
tensorrt
possibly others
```

Important distinction:

```text
Engine = runtime/library adapter that knows how to load and run a model.
Model = trained artifact plus metadata.
Input utility = prepares local paths, URLs, frames, streams, etc.
Renderer = turns normalized outputs into images or overlays.
Job runner = handles long-running work such as video ranges or training.
```

Examples:

```text
UltralyticsEngine
  Handles Ultralytics-compatible YOLO .pt models.
  Loads YOLO handles with YOLO(cached_artifact_path).
  Runs prediction on one prepared visual frame.
  Normalizes detections into worker response dictionaries.
  Does not own generic URL downloading anymore.
  Does not own generic detection rendering anymore.

CustomTorchEngine
  Will later handle Isaac's own PyTorch models.
  Raw .pt checkpoints may require architecture code and preprocessing/postprocessing rules.

TorchScriptEngine
  May later handle exported TorchScript models that are easier to load generically.
```

## Current UltralyticsEngine Behavior

`UltralyticsEngine` currently:

```text
Lazily imports YOLO inside load_model()
Loads a cached Ultralytics .pt artifact
Stores real YOLO handles by model_id inside the engine
Exposes get_loaded_model(model_id)
Exposes infer_frame(...)
Exposes infer_frame_image(...)
Uses a private _run_frame_prediction(...) helper to avoid duplicated prediction parsing
Returns normalized detections with class ID, class name, confidence, bbox_xyxy, bbox_xyxyn
Calls shared frame source preparation for local paths and URL images
Calls shared detection renderer for annotated images
```

The engine should remain focused on Ultralytics runtime behavior and Ultralytics output normalization.

Avoid pushing these generic responsibilities back into the engine:

```text
URL downloading
Local/remote source normalization
Image annotation drawing
Video decoding
HLS handling
Job management
Training job orchestration
```

## Current Input Utilities

Shared input utilities were added under:

```text
src/marp_inference_worker/inputs/
  __init__.py
  frame_source.py
```

`frame_source.py` owns preparing visual frame sources for engines.

Current behavior:

```text
If image_source is a local path:
  return it directly as prediction_source

If image_source starts with http:// or https://:
  download once using httpx
  follow redirects
  require image/* content type
  write bytes to a temp file with a reasonable suffix
  return the temp file path as prediction_source
  allow cleanup after inference
```

This was added because passing arbitrary HTTP URLs directly to Ultralytics/OpenCV can cause OpenCV to misclassify a URL as a video stream and spam errors such as:

```text
WARNING Video stream unresponsive, please check your IP camera connection.
retrieveFrame Picture does not contain data
```

Do not pass raw arbitrary image URLs directly into `YOLO.predict(source=...)`. Use `prepare_frame_source()`.

## Current Rendering Utilities

Shared rendering utilities were added under:

```text
src/marp_inference_worker/rendering/
  __init__.py
  detection_renderer.py
```

`detection_renderer.py` owns drawing normalized detections onto an image and encoding the image.

Current public renderer function:

```text
render_detections_to_image_bytes(source_image, detections, output_format="jpg") -> bytes
```

Current rendering behavior:

```text
Draw yellow detection boxes
Draw translucent black label bars
Draw white text labels with class name and confidence
Place label bar below the detection box when possible
If below would leave the image, place label inside the bottom of the box
Center label bar horizontally on the detection box
Allow label bar to be wider than very small boxes so text remains readable
Clamp label bar inside the image
Shrink font only down to a readable minimum
Truncate text with ... only if it still cannot fit
```

Rendering constants live near the top of `detection_renderer.py` to avoid magic numbers. Constants include:

```text
LABEL_MIN_FONT_SCALE
LABEL_MAX_FONT_SCALE
LABEL_FONT_SCALE_BOX_HEIGHT_DIVISOR
LABEL_MIN_WIDTH_PIXELS
LABEL_HORIZONTAL_PADDING_PIXELS
LABEL_MIN_VERTICAL_PADDING_PIXELS
LABEL_VERTICAL_PADDING_SCALE
DETECTION_BOX_THICKNESS_PIXELS
LABEL_BACKGROUND_ALPHA
LABEL_IMAGE_ALPHA
LABEL_FONT_SCALE_REDUCTION_STEP
```

Isaac reports the current rendering is working well and is visually how he wants it.

Current limitation:

```text
Encoded annotated output is currently JPG.
The user is interested in output_format=auto later.
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
Dispatch infer_frame to the engine for an already-loaded model
Dispatch infer_frame_image to the engine for direct annotated image output
Build class_id to class_name maps from ModelSpec.labels
```

The loaded-model registry is currently an in-memory dictionary. It stores `ModelSpec` by model ID. Real YOLO handles are stored inside `UltralyticsEngine`, also in memory.

Important TODO:

```text
Replace temporary loaded-model state with a real worker state layer.
The worker will eventually need loaded handles, job state, and resource state.
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
        inference_routes.py
      engines/
        __init__.py
        base_engine.py
        mock_engine.py
        ultralytics_engine.py
        engine_registry.py
      inputs/
        __init__.py
        frame_source.py
      models/
        __init__.py
        model_spec.py
        inference_spec.py
        model_cache.py
        model_manager.py
      rendering/
        __init__.py
        detection_renderer.py
  tests/
    test_health.py
    test_status.py
    test_models.py
    test_model_cache.py
    test_engine_registry.py
    test_inference.py
```

## FastAPI Design Rules

Use FastAPI only as the HTTP API layer.

Use route files for endpoint groups:

```text
api/
  health_routes.py
  status_routes.py
  model_routes.py
  inference_routes.py
  job_routes.py
```

Route handlers should stay thin. They should:

```text
Validate request bodies through Pydantic schemas
Validate query parameters through FastAPI/Pydantic
Call manager/service classes
Translate expected exceptions into HTTP errors
Return API responses
```

Route handlers should not contain:

```text
model download logic
engine loading logic
image source preparation
detection rendering logic
video decoding
job execution
heavy inference code
database-specific logic
```

`main.py` should remain thin and expose the module-level `app` object for Uvicorn.

`api/app.py` should create the FastAPI app, set metadata, and include routers.

## Testing Rules

Isaac’s development preference on tests:

```text
Design and write the function first.
Then write or update tests after the intended behavior exists.
```

Do not insist on strict test-first development.

Every new endpoint or changed endpoint contract should eventually be accompanied by a test or a modification to an existing test.

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

For inference route tests:

```text
Do not run real YOLO inference in normal pytest route tests.
Monkeypatch model_manager.infer_frame or model_manager.infer_frame_image.
Use mocked detections to protect route response shape.
Keep real YOLO/image inference as manual validation or optional integration tests later.
```

Monkeypatching in this project means temporarily replacing a real function such as `model_manager.infer_frame()` with a fake test function so the route can be tested without CUDA, model files, image files, or Ultralytics execution.

Current pytest command:

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

If exact current code matters and is not known, ask Isaac to paste the exact file or function before advising where to patch.

Do not assume code that has not been shown in the current context.

For a whole-function replacement:

```text
Give the new full function only.
Do not include the old full function.
Use this when the function is small or when many small edits would be harder to apply.
```

For a partial-function replacement:

```text
Give exact old code to find.
Give exact replacement code.
Include enough surrounding context so Isaac knows where it goes.
Use this only when replacing a clearly identifiable small block is easier than replacing the whole function.
```

Isaac prefers whole-function replacements when functions are small instead of hunting for many tiny code blocks.

For adding a new import:

```text
State where it belongs.
Show the surrounding import group when helpful.
```

For adding a new file:

```text
First give the PowerShell command to create the file from the project root.
Then provide the full file content.
```

Example:

```powershell
New-Item -ItemType File -Force .\src\marp_inference_worker\inputs\frame_source.py
```

Isaac uses Visual Studio Code and does not need instructions for manually inspecting files. Do not tell him how to inspect files unless he asks.

Use PowerShell commands for Windows shell instructions.

Use `vim` for Linux config edits unless Isaac specifies otherwise.

Final JSON outputs should be in code blocks for easier copying.

Keep responses concise. Isaac explicitly asked for less overexplaining. Give enough context to reason about the change, but avoid broad surveys unless he asks.

## Working Style with Isaac

Isaac is the programmer. The LLM is the assistant.

Do not race ahead. Do not design huge systems without checking direction.

Work one small milestone at a time.

If Isaac asks for a test to perform, give exactly that test and wait for his result before moving to the next implementation step.

Do not give a long sequence of future edits and tests unless he asks for a full plan.

A useful step pattern is:

```text
1. Explain the immediate design decision.
2. Explain what the patch changes and why.
3. Provide the exact code patch.
4. Ask Isaac to run pytest or manually test.
5. Use test results as the checkpoint.
```

If Isaac reports a failure, focus on the failure first. Do not pile on unrelated improvements.

When creating routes or tests, do not skip the concrete request body. Isaac expects exact JSON or exact browser URLs when asked to test an endpoint.

When testing browser image routes, provide a directly pasteable URL.

## Git Notes

Use small commits.

Useful commit milestones completed or expected around this handoff:

```text
Initial FastAPI inference worker skeleton
Add static worker status endpoint
Add fake model loading endpoints
Add model load settings schema
Add model cache path planning
Add model artifact download/cache path
Support local model artifact caching
Add engine registry and mock engine
Add Ultralytics engine registration and load_model
Add frame inference API route
Add direct annotated frame image route
Add shared frame source input utility
Add shared detection renderer
Refactor Ultralytics frame prediction helper
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

## Current Manual Validation Milestones

These have been manually validated:

```text
A real local Ultralytics YOLO .pt model can be loaded through POST /models/load.
The model artifact is copied into models/cache/.
UltralyticsEngine loads the cached artifact into a YOLO handle.
POST /infer/frame can run inference on a local image.
POST /infer/frame returns normalized detections.
Each detection includes pixel bbox_xyxy and normalized bbox_xyxyn.
GET /infer/frame/image can return an annotated image directly to a browser.
The annotated image shows boxes and readable labels.
The renderer now uses shared constants for label sizing and layout.
```

A successful detection response included species labels such as:

```text
Leather star
Bat star
California sea cucumber
Fish-eating anemone
```

These labels came from the model’s Ultralytics names when `ModelSpec.labels` was empty.

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

## Architectural Direction Before Jobs

Do not start implementing jobs until Isaac asks.

Before jobs, keep the current layering clean:

```text
API route
  Receives HTTP request.
  Validates request body or query parameters.
  Calls model_manager.

Model manager
  Finds the loaded model spec.
  Selects the correct engine.
  Builds label maps.
  Dispatches the operation.

Engine
  Loads a runtime model.
  Runs model-specific prediction.
  Converts model-specific outputs into normalized worker results.

Input utilities
  Prepare local paths, URLs, future uploaded files, future decoded frames, etc.

Rendering utilities
  Turn normalized detections into annotated images.

Future job runners
  Process long-running work like video ranges, streams, training, evaluation.
```

For video range processing later:

```text
A video job runner should decode a local video or HLS stream into visual frames.
An image/frame-capable engine should process each decoded frame.
The job runner should aggregate results and track progress.
```

Do not force video/HLS handling directly into `UltralyticsEngine`.

For future training:

```text
Training is a long-running job, not just an engine load call.
Training will need job state, progress, metrics, artifacts, cancellation, and resource tracking.
```

## Future Capabilities and Planned Directions

Likely future additions:

```text
GET /system/resources
  Reports CPU, RAM, disk, CUDA availability, GPU names, VRAM, Torch version.

Local job manager
  Tracks queued/running/finished/failed jobs.
  Supports job status and cancellation.

Training job route
  Accepts training configs.
  Runs long jobs.
  Reports epochs, metrics, artifacts.

Video/HLS range job route
  Accepts local video path or HLS URL plus frame/time range.
  Decodes frames.
  Runs frame inference repeatedly.
  Aggregates standardized detections.

ByteTrack support
  Should come after frame inference and probably after video/frame sequence handling.
  Tracking needs sequential frame state and should not be mixed into simple model loading.

Output format support
  Add output_format=auto/jpg/png for annotated image routes.
  Current output is JPG.

Hash verification
  Verify sha256 before reusing cached artifacts.

Worker status backed by real state
  /status should eventually reflect real loaded models, active jobs, and possibly resource state.
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
training and inference as worker tasks
```

The architecture direction is to avoid another large monolithic script. The project is being built as a modular API service with small files, tests, comments, and replaceable components.

Older design decisions:

```text
The coordinator will eventually talk to the MARP database API, model server, Jellyfin/video server, and multiple workers.
The worker should eventually be able to pull video or HLS streams directly from the video server.
For high-throughput video work, the worker should process video ranges locally rather than receive one HTTP request per frame.
Single-frame inference is still needed for testing, GUI tools, and debugging.
```

## Current Best Next Steps

At this handoff, the code is working and tests pass.

Good next steps, depending on Isaac’s choice:

```text
1. Add or update tests for the direct image route and renderer behavior.
2. Add output_format=auto/jpg/png for GET /infer/frame/image.
3. Add a system/resource endpoint foundation.
4. Start designing a lightweight job model before video range processing.
5. Begin planning ByteTrack support, but do not implement until frame sequence processing exists.
```

Do not begin a large job-manager or training implementation without first checking direction with Isaac.
the dev system we are working with is currently windows, but we want this program to be able to work on linux as well, usually ubuntu
