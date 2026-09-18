# Handoff — MARP distributed GPU compute, Milestone 1

State as of 9 Sep 2026. Written so this work can be resumed without
reconstructing it from a conversation. Read `.marp/task.md` for the spec and
`.marp/verification.md` for what is and is not tested.

## Where the work lives

| Repo | Branch | Commits | Tests | Pushed |
|---|---|---|---|---|
| `MARP_API` | `3-gpu-orchestration-api` | 5 | 264 pass | yes |
| `marp-inference-worker` | `3-worker-dial-out-loop` | 7 | 180 pass | 6 of 7 |

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
# an activation code, from a signed-in user (not a service token):
#   POST /api/v2/gpu/worker-activation-codes
cd ../marp-inference-worker
MARP_COORDINATOR_URL=http://localhost:3000 \
  ./.venv312/Scripts/python.exe -m marp_inference_worker.worker_main \
  --activate-code-file <file holding the code>
# then, with no token in the environment at all:
MARP_COORDINATOR_URL=http://localhost:3000 \
  ./.venv312/Scripts/python.exe -m marp_inference_worker.worker_main
```

**Activation, not a hand-minted token, and this is not a preference.** An
application token from `create-application-token.js` **cannot enrol a worker**
and has not been able to since `MARP_API#190`: `authorizeEnrol` requires the
credential to be bound already to a `gpu_workers` row whose `local_id` matches,
and a new token is bound to nothing, so it is `403 This credential belongs to a
different worker identity` before `local_id` is ever compared. That message is
misleading for the commonest case -- a credential bound to nothing at all --
and reads as though somebody else's credential had been used.

Reading still works with such a token, which is the trap: `GET /workers`
answers 200, so the token tests fine and then fails at the one call that
matters.

Activation exchanges the code for a per-machine credential, DPAPI-protected at
`<state dir>/worker-credential.dpapi`. After it, **no token goes in the
environment at all** -- the worker loads its own. The token script is still
right for a service consumer that only reads.

The worker's own loopback API is on `127.0.0.1:8010`. Confirm enrolment with
`GET /api/v2/gpu/workers` as a signed-in user. Note that reading the pool and
enrolling into it are gated differently -- this line used to say "using the
token", which is what made a minted token look like it worked.

## What is proven

Enrolment, end to end, against the real coordinator: the pool shows
`SoftwareEngineering-<id>` online, one slot, the real GPU with VRAM and
capability, and both engines (`marp_tracking`, `mock`).

**The job round trip, across two machines on two networks -- 17 Sep 2026.** It
had never run at all before that day; item 1 below was open for a week. Two
workers enrolled against one coordinator over a public address, and a
`marp_tracking` job ran on a laptop holding no model: MARP streamed the weights
from `GET /api/v2/model/:id/artifact`, the `sha256` in the spec verified them on
arrival, ultralytics and ByteTrack ran on the second machine's GPU, and a
content-addressed artifact came back whose hash matched what MARP recorded.
Stop and resume too: a stopped run reports `yielded` with the frame it reached,
the job returns to the queue, and the next lease begins at exactly that frame.

What the second machine found, and one machine could not: every model in a job
spec was named by an absolute Windows path, because the registered artifact had
no bytes behind it; the mock engine could not run without weights it has no use
for; `yielded` was a word the coordinator refused; and the watch window had
never opened on any machine, because its page existed in neither repository.
All four were invisible with one computer.

## What is NOT proven — start here

1. ~~**The job round trip.**~~ **Done, 17 Sep 2026**, and it took a second
   machine -- see *What is proven* above. The `mock` engine was meant to be the
   cheap first step and could not run at all: the runner fetched and verified a
   model for every job, including the one engine that needs none.
2. ~~**Real YOLO over a real Jellyfin video.**~~ **Done, 9 Sep 2026**, coordinator
   excluded — see *Results — real model over real video* in
   `.marp/verification.md`. Stock `yolov8n` over a CAMPA2021 clip, driven
   through `jobs.child_main` with a hand-written envelope.

   The headline: **seeking is exact.** `CAP_PROP_POS_FRAMES` landed on the
   frame it was asked for on all 22 targets tried, across a 979-frame clip and
   a 38,159-frame one. The stream direct-plays rather than transcodes, which is
   why — ffmpeg can seek by byte offset in an indexed MP4. A **transcoded**
   stream is still unmeasured and is the remaining risk here.

   Also settled: `infer_stream`'s normalization of Ultralytics `Results` is
   correct against the real 8.4.145 object, and the `[0,300)` / `[300,600)`
   seam holds on real video — 0..299 and 300..413.

   Still not run: anything coordinator-mediated. Manual steps 4-7 are open.
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
