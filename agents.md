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


## Recent Project History: Jellyfin Streams, Dataset Building, ByteTrack Reference Code, and Training Pipeline

This section records work completed after the earlier FastAPI inference-worker foundation described above.

### Legacy AI Scripts as Reference Material

Older MARP AI scripts have been copied into the repository so they can be run independently during migration and used as reference material while building the new worker architecture.

These scripts are **not** the final architecture. They are temporary reference and compatibility code.

Use them for:

```text
Manual validation
Understanding older MARP AI workflows
Comparing behavior while porting features
Testing models and datasets before the new worker job system exists
```

Do not use them as the destination for new production architecture.

Reusable logic from the old scripts should gradually move into structured packages such as:

```text
src/marp_inference_worker/media/
src/marp_inference_worker/inputs/
src/marp_inference_worker/engines/
src/marp_inference_worker/jobs/
src/marp_inference_worker/training/
```

The old scripts can still be run independently from the project root with the virtual environment active.

### ByteTrack Reference and Migration Notes

ByteTrack reference code is present in the repository so the old tracking workflow can still run during migration.

Current status:

```text
The old ByteTrack workflow can be used for manual testing.
ByteTrack should not be wired directly into model loading or single-frame inference.
Future ByteTrack support belongs after video or frame-sequence processing exists.
Tracking requires sequential frame state and should eventually be part of a video job or tracking module.
```

When tuning for the MARP review workflow, Isaac prefers higher recall even if this increases false positives:

```text
False positives are easier for biologists to reject in review mosaics.
False negatives are more expensive because they require reviewing much more video.
```

For high-recall tracking tests, verify that YOLO confidence thresholds are actually applied before tuning ByteTrack parameters. If low-confidence raw YOLO detections do not contain the missed animals, ByteTrack tuning will not fix the false negatives.

### Jellyfin Client Port

The old C# application already had a mature Jellyfin API client. The Python worker now has an initial port of the reusable service/client pieces.

New media package:

```text
src/marp_inference_worker/media/
  __init__.py
  jellyfin_client.py
  video_source_resolver.py
```

`jellyfin_client.py` is the reusable Python Jellyfin API client. Its first responsibilities are:

```text
Authenticate to Jellyfin using /Users/AuthenticateByName
Store base_url, access_token, and user_id
Use X-Emby-Token for authenticated requests
Get top-level libraries
Get child items
Search video items
Get PlaybackInfo
Request constrained transcode PlaybackInfo using a DeviceProfile
Build direct/original stream URLs
Convert Jellyfin relative URLs into absolute URLs
```

Environment variables used for development testing:

```powershell
$env:JELLYFIN_BASE_URL="http://47.208.203.78:8096"
$env:JELLYFIN_USERNAME="guest1"
$env:JELLYFIN_PASSWORD="guest1"
```

Do not hardcode these credentials into source code. Environment variables are acceptable for local development. Long term, this should become settings/config.

Manual smoke testing confirmed:

```text
Jellyfin authentication succeeds.
Jellyfin item search works when using normalized filename stems.
OpenCV can open direct Jellyfin stream URLs.
OpenCV can read frame count, FPS, width, and height from at least one Jellyfin stream.
OpenCV can decode and save a frame from a Jellyfin stream.
```

### Jellyfin Video Source Resolver

`video_source_resolver.py` sits above the raw Jellyfin client.

Purpose:

```text
Convert database video_source values into something OpenCV can open.
Prefer an existing local path when available.
Fall back to Jellyfin stream resolution when local video is missing.
Normalize filename differences between MARP database values and Jellyfin item names.
```

The resolver handles cases such as:

```text
Database video_source:
20240727_185645 Fwd.mp4

Jellyfin item name:
20240727_185645_Fwd

Jellyfin path:
/mnt/rov-video-new/CAMPA2024/Dive 1/20240727_185645_Fwd.mp4
```

The resolver searches in stages:

```text
Exact video_source
Filename basename
Filename without extension
Space-to-underscore variant
Underscore-to-space variant
Timestamp prefix when available
```

It ranks candidates by normalized item name and path basename so that spaces, underscores, hyphens, and extensions do not prevent matching.

Manual resolver testing confirmed:

```text
Input:
20240727_185645 Fwd.mp4

Resolved:
Jellyfin item 20240727_185645_Fwd

Match score:
100

OpenCV:
Successfully opened direct Jellyfin stream and decoded a frame.
```

### Dataset Builder Jellyfin Integration

The legacy dataset builder was updated so it can use the new `VideoSourceResolver` instead of only assuming every database `video_source` exists inside a selected local folder.

Old behavior:

```text
video_source → selected local folder path → cv2.VideoCapture(local path)
```

New behavior:

```text
video_source → VideoSourceResolver
    1. try selected local folder
    2. if missing, resolve through Jellyfin
    3. pass local path or Jellyfin stream URL to cv2.VideoCapture
```

Validated runtime behavior:

```text
The dataset builder authenticated to Jellyfin.
The resolver became available for missing local videos.
The builder opened at least one Jellyfin direct stream URL.
The existing frame-processing progress output appeared after OpenCV began reading the stream.
```

Example output:

```text
[INFO] Jellyfin resolver is available for missing local videos.
[INFO] Opening video '20240730_190910 Fwd.mp4' using jellyfin_stream: http://47.208.203.78:8096/Videos/.../stream?static=true&api_key=...
.
```

This confirms the dataset builder can reach frame processing through Jellyfin.

### Temporal Split Infinite Loop Bug

While testing Jellyfin-backed dataset building, the script appeared to stall before video opening. Debugging confirmed this was not a Jellyfin stream problem. It was a pre-existing temporal split bug.

Problem pattern:

```python
while start < total_annotated:
    train_end = int(start + total_annotated * segment_train_ratio)
    eval_end = int(train_end + total_annotated * segment_eval_ratio)
    start = eval_end
```

With small annotated-frame counts, this can produce:

```text
total_annotated=7
segment_train_ratio=0.1
segment_eval_ratio=0.05

train_end=0
eval_end=0
start remains 0 forever
```

Diagnostic output confirmed:

```text
[ERROR] Temporal split loop would stall for video '20240801_161909 Fwd.mp4'.
start=0, train_end=0, eval_end=0, total_annotated=7,
segment_train_ratio=0.1, segment_eval_ratio=0.05
```

The fix was to convert ratios into minimum segment sizes:

```text
train_segment_size = max(1, int(total_annotated * segment_train_ratio))
eval_segment_size = max(1, int(total_annotated * segment_eval_ratio))
```

and to guard that `start` always advances.

The summary print was also protected from division by zero by computing `split_frame_count` before calculating percent train.

Validated result after fix:

```text
20240801_161909 Fwd.mp4: 4 train frames, 3 eval frames
20240731_150901 Fwd.mp4: 4 train frames, 4 eval frames
```

The dataset builder then reached observation split assignment, database registration, resolver setup, and Jellyfin video opening.

### Training Pipeline Review

The legacy training pipeline currently uses a three-phase YOLO training process.

Current conceptual behavior:

```text
Run one training phase.
Find that phase's weights/best.pt.
Use that best.pt as starting weights for the next phase.
Repeat for all phases.
```

The weight handoff logic is valid.

However, the documented phase order and implementation should be checked. The intended conceptual order is likely:

```text
crops → frames → mixed
```

Rationale:

```text
crops:
  teach organism appearance and small-object detail

frames:
  teach real ROV full-frame context

mixed:
  consolidate crop-scale detail and full-frame context
```

Current epoch logic shortens some phases. Make sure the database records store the actual phase-adjusted epoch count, not just the original configured epoch count.

Potential issue: the training function appears to evaluate and save metrics in the normal success path, then also evaluate and save metrics again in a `finally` block. Since `finally` always runs, this may create duplicate metric summaries on successful training. Refactor later so final evaluation and DB summary happen exactly once.

### Current Working State After This Session

Current validated facts:

```text
Old scripts are available as runnable reference material.
ByteTrack reference code is available for manual tracking tests.
A Python JellyfinClient exists under marp_inference_worker.media.
A VideoSourceResolver exists under marp_inference_worker.media.
Standalone Jellyfin stream and resolver smoke tests work.
OpenCV can read direct Jellyfin stream URLs from the Python worker environment.
The dataset builder can resolve missing local videos through Jellyfin.
The dataset builder can begin reading a Jellyfin stream through OpenCV.
The temporal split infinite-loop bug for low annotated-frame-count videos was found and fixed.
```

Current caution:

```text
The old scripts are still legacy migration code.
Do not treat old_scripts as final architecture.
Port reusable pieces into marp_inference_worker modules gradually.
Avoid turning the training script into another permanent monolith.
```

### Updated Near-Term Next Steps

Good next steps from this point:

```text
1. Let the Jellyfin-backed dataset build complete and inspect output folders.
2. Remove temporary debug prints from the dataset builder after confirming stability.
3. Commit the Jellyfin client, resolver, smoke tests, dependency updates, and temporal split fix.
4. Search the tracking script for all confidence/threshold filters before tuning ByteTrack further.
5. Run raw YOLO-only detection at very low confidence to determine whether false negatives are from the detector or the tracker.
6. Change training phase order to crops → frames → mixed if Isaac confirms.
7. Fix training DB total_epochs to store the actual phase-adjusted epoch count.
8. Later refactor duplicate evaluation/metrics logic so final metrics are saved once.
9. Later migrate dataset building and training into proper worker job modules instead of old_scripts.
```
