# marp-inference-worker — agent instructions

A Python FastAPI service for running model workloads on one worker machine. One component
of the [MARP platform](https://github.com/MarineAppliedResearch/MARP).

**This file is the source.** `CLAUDE.md` and `.github/copilot-instructions.md` point here.
The shared block below is synced from the umbrella — edit `MARP/AGENTS.md` and run
`marp harness sync`, never this copy.

The Copilot instructions in this repository already said that, in July: *"the AGENTS.md
file is the authoritative source of truth for LLM assistants working on this project."*
Two things were wrong with it and are now fixed — the file was at the repository root,
where Copilot does not look for it, and the file it named did not exist. `agents.md` and
`AGENTS.md` are the same file on Windows and two different files everywhere else,
including on GitHub.

## What this file used to be, and why it changed

The previous `agents.md` ran to 1,488 lines. Roughly half was `## Current …` prose
describing endpoint shapes, response bodies, cache behaviour and project structure by
hand — a copy of the code with no way to notice when it stops being true, and two months
stale by the time the harness was built. It has been removed rather than updated.

**The API surface is documented by the API.** FastAPI generates OpenAPI; read `/docs` or
the schema rather than a prose transcription of it. If something about current behaviour is
worth knowing and the code does not say it, that belongs in a comment in the code or in a
decision record.

Also removed: *Patch Delivery Rules for LLMs*, which described handing code back through a
chat window as whole-function replacements and PowerShell `New-Item` commands. Agents
working here edit files directly, so those rules were not merely obsolete but wrong. And
the session-state sections — what was working on a given day, and what was next — which are
the parts that go stale silently.

What survived: the design rules, the testing rules, the comment style, the architectural
direction, the model and result semantics, the history, and the findings that cost real
time to establish. *Working Style with Isaac* — the clearest statement of the development
loop anywhere in the platform — was promoted into the shared block, so every repository
gets it.

## State of this repository

**Work happens on `develop`.** Other branches exist and are not the question; the
registry says `develop` and that is the answer.

Dormant since 2026-07. One thing still needs doing before an agent is pointed at it: the
checked-in `.venv` points at a Python that does not exist on the current machine and needs
recreating against 3.12 — deliberately 3.12 rather than newer, because `torch` and
`ultralytics` wheels lag new Python releases.

Running it needs an NVIDIA GPU. Building it does not.

<!-- marp:shared start -->
<!-- Canonical source: MARP/AGENTS.md. Do not edit this block in a component repository;
     edit it here and run `marp harness sync`. -->

## The platform

MARP is a polyrepo. `services/repos.yml` in the umbrella repository is the registry of
what MARP consists of, and it is authoritative — including for which branch to work on.

**Start from `repos.yml`'s `default_branch`, not from GitHub's default branch.** They
differ deliberately. `master` in this platform means *what is in production*, and
production is promoted by hand, so `master` can be far behind and that is not decay. Work
happens on `develop` where a repository has one.

Branch model is Gitflow: `master` is production, `develop` is integration, and every task
gets its own branch off `develop` named for its issue (`68-mosaic-review-prototype`).
Never commit directly to `master` or `develop`.

## Rules that are not negotiable

- **Commit authorship is the human developer only.** Never add an AI assistant as author
  or co-author, never add a `Co-Authored-By` trailer, and never mention an assistant or
  vendor in a commit message, PR title, or PR body. This applies to merge and squash
  commits too.
- **Never commit `.env` files, credentials, tokens, keys, or host passwords.** Each
  repository has a `.env.example` documenting variable *names*. Operational detail for a
  specific machine goes in `.marp/local/`, which is git-ignored.
- **The production database is a scientific record.** `mare_v1` holds years of annotation
  that is queried and reported on by people and tools outside this workspace. Any
  transformation of existing data must either preserve everything currently possible or
  lose nothing — a column that stops being populated, a value that becomes ambiguous, or a
  format an existing query no longer parses all count as loss, even when the application
  still works. Derived columns are part of the contract.
- **Ask about meaning rather than inferring it from the data.** How a field is meant to
  work, what an empty value means, whether two similar rows are one thing or two — these
  are answerable by the person who recorded them and not reliably by inspection.

## Keep commit messages short

Subject under ~72 characters plus a few one-line bullets. Reference the issue with
`Refs #NN` or `Closes #NN`. Cross-repository work references the other side in full:
`MarineAppliedResearch/MARP_API#68`.

## The workflow, and where it stops for a human

```
G0  Intake      read the task, this file, and the repository's decision records
G1  Design      investigate -> write .marp/task.md -> surface assumptions
    GATE          the human answers. Nothing is implemented while a `blocking`
                  assumption is open. This is enforced, not requested.
G2  Implement   implement the settled spec. Fast, autonomous, no questions --
                  unless a NEW material assumption appears, which returns to G1.
G3  Test plan   write .marp/verification.md: what will be tested, which
    GATE          requirement each test proves, and what is NOT covered.
                  The human reviews the PLAN before anything is run.
G4  Verify      run the approved verification, record real results including
    GATE          failures, verbatim. The human reviews the evidence.
G5  PR          opened only when the human says so. Never automatically.
G6  Merge       CI green plus human approval.
```

`.marp/task.md` is the task specification and it lives on the task's own branch, so it
travels with the code and appears in the pull request. `.marp/task.template.md` is the
skeleton. Durable decisions are promoted out of it into decision records
(`docs/decisions/` for one repository, the umbrella's `architecture/decisions/` for
anything spanning two).

## Surfacing assumptions is the point

Agents make plausible but incorrect assumptions, and a material assumption must never
silently become an implementation decision. During G1, write down anything of these kinds
that the task does not settle:

behavioural · product/UI · scientific or data-meaning · database/schema · API contract ·
architectural · performance/concurrency · security/permissions · destructive operations ·
cross-repository integration · environment

Each goes in `## Open assumptions` in `.marp/task.md` as a checklist item tagged with its
category and whether it is `blocking`. `marp spec check` fails while a blocking assumption
is unticked, which is what actually stops G2 from starting.

Trivial local choices that follow an established pattern in the repository are not
assumptions. If you are unsure whether something is material, the test is: *would a
different reasonable answer change the behaviour, the schema, the interface, or the
data?* If yes, it is material.

Discovering a new material assumption during G2 is normal and is not a failure. Append it,
say so, and stop — do not guess to preserve momentum.

## Working in parallel

Several agents can work at once, and the model is the ordinary one: **each works on its own
branch, in its own copy of the repository, and pushes that branch when the work is done.**
Branches are merged the usual way. The only extra requirement is that two agents must not
collide over the things a running MARP needs.

```bash
marp agent start marp-api 71-thumbnail-lifecycle
```

That gives the branch its own copy, its own database on its own port, its own API port, a
written `.env`, and its dependencies installed — so it can run and test without touching
anybody else's. It installs the nested packages too — an application with its own `package.json` is a
package in its own right, and an agent that finds no `node_modules` there cannot run its
tests.

`marp agent list` shows what is set up and where; `marp agent env <branch>` prints the
settings again; `marp agent stop` and `marp agent remove` shut down the servers and the
database that workspace started. **`remove` keeps the branch**, because tidying up and
discarding work should never be the same command.

**Stop what you start.** A server outliving its work is not untidiness — one left running
in another checkout was adopted by a different workspace's browser tests, which then
graded that checkout's code for an hour without saying so.

`marp harness check` reports when two workspaces collide: the same port is a failure, an
exclusive resource named twice in `needs:` is a failure, and two agents on one repository
is a note for a human to judge.

On a second machine there is nothing to set up: clone the repository, check out the branch,
and it is already isolated. The command exists for putting several on one machine.

Two things are deliberately shared:

- **Jellyfin.** Every agent talks to the central MARP media server. It holds the real
  library, and the tests that touch it read far more than they write. A task that genuinely
  needs its own instance says so; nothing else should.
- **The PostgreSQL binaries**, downloaded once. Only the data directory is per-agent.

**Parallelism comes after the design is settled, never before.** Two agents each doing
their own investigation on overlapping surface is how two incompatible interpretations of
MARP get built. One agent settles the assumptions with the human; then the work fans out.

## Testing doctrine

Learned the expensive way, and it holds everywhere in this platform:

- **A defect is not fixed until it has a named test at a tier that can actually observe
  it.** Several defects here were reported twice because the first fix was verified at a
  tier that structurally could not see the bug. Store-level checks cannot see what was
  drawn; unit tests cannot see what a browser rendered.
- **A test that narrates a result without asserting it can lie.** This applies to
  walkthrough videos especially: a scene that says "the tile is now excluded" and only
  asserts that a panel opened will pass for weeks while excluding nothing.
- **Run the fast tiers after every change. Run the whole suite before calling anything
  done.** Parse and unit checks cost about a second and are the working loop. The slow
  tiers — browser, database, hardware — are not for routine feedback, but nothing is
  finished until they have passed. Run them as often as the work needs; never skip them
  to declare something working. `marp verify run` is that run.
- **CI runs the fast tiers only, deliberately.** A minute of browser tests on every push
  taxes every commit. That means **CI going green is not the same as the work being
  verified** — G4 is not satisfied by a green pipeline.
- **A skipped suite looks green.** Prerequisites missing should fail, not skip.

## Documentation that states an environment fact

**Point at the command; do not restate the value.** A host, a port, a path or a version
written into prose goes stale silently and an agent cannot tell. Write *"run `marp db
status` to see yours"* rather than naming a host and port.

This is not a style preference. The umbrella's own `CLAUDE.md` once described the database
in two contradictory ways sixty lines apart, and an agent resolved the contradiction toward
the stale half and built a plan on it. `marp harness check` now greps tracked instruction
files for environment literals and retired markers.

The same rule retires any document that promises to stay in sync with code it cannot
observe. Do not write a `## Current API` section by hand; point at the generated contract.

## How corrections become durable

When a human corrects an agent, the correction should make the same mistake less likely
next time. Route it by this ranking:

> **A correction becomes a check if it possibly can, a test if it cannot be a check, and a
> sentence only if it can be neither.**

| The correction is about | Where it goes |
| --- | --- |
| what the system should do | `.marp/task.md` requirements, plus a test naming that requirement |
| a decision that constrains future work | a decision record |
| how agents should work, everywhere | this shared block |
| a rule for one area of one tree | `.github/instructions/*.instructions.md` |
| a defect | a named test at the tier that can see it |
| a mechanically checkable invariant | `marp doctor` or `marp harness check` or CI |
| something an agent should not do | a hook or a permission rule |

## Permissions

**Free:** read anything, search, run parse/unit/contract tiers, write to a task branch,
write `.marp/*`, commit locally, query a local disposable database, read the GitHub API.

**Ask first:** `git push` · opening a pull request (this is gate G5) · migrations against
anything but a local disposable database · any write to a shared database · adding a
dependency · editing generated output by hand · changing a published contract surface.

**Never without the human present:** anything against production `mare_v1` · the live
Jellyfin service and its configuration · force push · branch deletion · rewriting
published history · restoring anything from a `retired-migrations` directory · rotating
credentials.

## Working style

The human is the programmer; the agent is the assistant.

- Do not race ahead, and do not design large systems without checking direction.
- Work one milestone at a time. If asked for a test, give exactly that test and wait for
  the result before moving on.
- If a failure is reported, focus on that failure. Do not pile on unrelated improvements.
- **Report a failure the moment you see it.** Do not silently run diagnostics while
  somebody waits, and never present a partial result as a finished one.
- State assumptions explicitly. If several interpretations exist, present them rather than
  picking silently. If a simpler approach exists, say so.
- Minimum code that solves the problem. No speculative features, no abstractions for
  single-use code, no configurability that was not asked for.
- Touch only what the task requires. Do not reformat, refactor or "improve" adjacent code.
  Match the existing style even where you would do it differently. Remove only the imports
  and variables your own change orphaned.
- Comments: many short ones rather than a few long ones, about two lines on average, and
  they explain *why* far more than *what*.

<!-- marp:shared end -->


## This repository

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

## Findings worth keeping

Root causes that cost real time to establish. These came out of one session's notes;
the surrounding session log -- what was working that day and what was next -- has been
dropped, because that is the part that goes stale without anyone noticing.

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

