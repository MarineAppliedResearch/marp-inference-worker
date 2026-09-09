# Handoff — MARP distributed GPU compute, Milestone 1

State as of 9 Sep 2026. Written so this work can be resumed without
reconstructing it from a conversation. Read `.marp/task.md` for the spec and
`.marp/verification.md` for what is and is not tested.

## Where the work lives

| Repo | Branch | Commits | Tests | Pushed |
|---|---|---|---|---|
| `MARP_API` | `3-gpu-orchestration-api` | 5 | 264 pass | yes |
| `marp-inference-worker` | `3-worker-dial-out-loop` | 5 | 138 pass | yes |

Tracking issue: `MarineAppliedResearch/marp-inference-worker#3` — its comments
carry the settled decisions. Dashboard design: `MARP_API#104`.

Research (7 files, ~40k words) is in the session scratchpad at
`C:\Users\mare\AppData\Local\Temp\claude\C--MARE-CODE-DEVELOPMENT-MARP\<session>\scratchpad\research\`,
with `00-compiled-plan.md` as the entry point. **Note this path is not inside
the workspace** — an earlier brief gave `../scratchpad/...` and an agent
concluded the files did not exist.

## Environment, verified working

- **CUDA works.** `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128` in
  `marp-inference-worker/.venv312` (Python 3.12.10). Verified by running a real
  kernel, not just `is_available()`: RTX 5060 Laptop, 8.5 GB, `sm_120`, and
  `sm_120` is in the build's arch list.
- **The driver is the constraint.** 573.13 supports CUDA 12.x only. `cu130`
  wheels (which have torch 2.14.0 exactly) install cleanly and then cannot see
  the device — `cudaGetDeviceCount()` returns `cudaErrorNotSupported`. Going
  back to torch 2.14 needs a driver of 580 or newer. Ultralytics does not care
  (`torch>=1.8`, no ceiling).
- **ByteTrack needed a shim.** The vendored copy uses `np.float`/`np.int`/
  `np.bool`, removed in numpy 1.24; it imports fine and fails on the first frame
  carrying a detection. Shim in `tracking/byte_tracker_adapter.py`, with
  `ByteTrack/` left byte-identical to upstream.
- `cython-bbox` compiled against MSVC 14.51.

## How to bring it up

```
marp db up                                    # from the umbrella
cd MARP_API && npm run dev                    # port 3000, branch checked out
# worker token, if a new one is needed:
node scripts/create-application-token.js --app "MARP GPU worker (dev)" \
  --permissions workers:enrol,jobs:execute,jobs:read,jobs:write
cd ../marp-inference-worker
MARP_WORKER_TOKEN=<svc_…> MARP_COORDINATOR_URL=http://localhost:3000 \
  ./.venv312/Scripts/python.exe -m marp_inference_worker.worker_main
```

The worker's own loopback API is on `127.0.0.1:8010`. Confirm enrolment with
`GET /api/v2/gpu/workers` using the token.

## What is proven

Enrolment, end to end, against the real coordinator: the pool shows
`SoftwareEngineering-<id>` online, one slot, the real GPU with VRAM and
capability, and both engines (`marp_tracking`, `mock`).

## What is NOT proven — start here

1. **The job round trip.** lease → progress → cancel → result → artifact has
   never run. The `mock` engine needs no GPU, so this is the cheapest next step
   and it exercises nine of the twelve routes.
2. **Real YOLO over a real Jellyfin video.** Now possible on this machine.
   Highest-risk part is frame-range *seeking* against a stream Jellyfin may
   transcode rather than direct-play — manual step 3 in
   `.marp/verification.md` exists for exactly this.
3. **The dashboard** (`R15`) is not built, by decision. Design first: `MARP_API#104`.

## Three bugs the end-to-end run found, all now fixed

Recorded because they show what the two test suites structurally could not see:

1. Enrolment answered `400: name is required` — the worker sent only its
   durable id. Name is now hostname plus an id slice: stable across restarts,
   unique across machines, so two machines sharing a hostname cannot share a
   pool row and steal each other's leases.
2. **The test fake had drifted from the contract.** `FakeCoordinator.enrol()`
   took two arguments where the real client needs five, and all 138 tests passed
   against it. A fake that no longer matches the contract tests nothing about it.
3. Capabilities were sent only at first enrolment, so the pool read "no GPU"
   long after CUDA worked. The worker now re-enrols on every start.

Before these, the frame-range convention had diverged the same way: MARP_API
implemented inclusive bounds, the worker half-open, and **both suites passed**.
Settled as **half-open `[start, end)`**; both repos now assert it, and the
worker's test was checked for vacuity by mutating the engine to `end + 1`.

## Open decisions

- **Ultralytics licensing** — deferred deliberately. AGPL-3.0, and the
  licensor's position names internal tools and self-trained models as requiring
  the paid Enterprise Licence. Three routes: buy it; comply with AGPL (MARP is
  public, but `MARP`/`MARP_API` have *no licence file* and the worker is MIT,
  which cannot contain AGPL code); or avoid Ultralytics where it matters, since
  PyTorch and torchvision are BSD-3-Clause.
- Timeout values: proposed 10 s heartbeat / 60 s timeout / 24 h attempt cap.
- One venv per engine, or a single pinned environment for Milestone 1.

## Found and deliberately left alone

- **`routes/jellyfin.routes.js` has 13 route registrations and no permission
  gate.** `/api/v2/jellyfin/libraries` answers **200 to an anonymous request** on
  the running server — verified, not inferred. Contradicts `AGENTS.md`.
  Pre-existing on `develop`, unfiled.
- Nothing can set a worker to `paused` through the API; it takes a direct
  database update. Left for the dashboard work.
- GitHub reports 57 vulnerabilities on `MARP_API`'s default branch (4 critical,
  28 high). `AGENTS.md` records only 4 deliberate ones.
- `src/marp_inference_worker.egg-info/` is tracked build output; `README.md` and
  those files carry pre-existing uncommitted edits, untouched throughout.
