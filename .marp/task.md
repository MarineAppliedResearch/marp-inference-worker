---
task: MarineAppliedResearch/marp-inference-worker#18
repos: [marp-inference-worker, marp-api]
status: implementing
needs: []
---

## Goal

A person on a supported Windows NVIDIA computer opens one small MARP installer once. It
prepares, activates, and starts the inference worker without asking them to install or
understand developer tools, runtimes, browsers, or packages.

## Requirements

- **R1** — The Windows x64 development installer is one-click and per-user. It requires no
  preinstalled Python, Node, Git, browser, CUDA toolkit, or C++ compiler.
- **R2** — The bootstrap remains small by downloading large versioned components during
  setup, with visible aggregate progress. It does not bundle PyTorch, Chromium, models, or
  the completed runtime inside its initial executable.
- **R3** — Setup detects a supported NVIDIA driver and GPU and installs the approved runtime
  variant. CUDA 12.6 is first, while release metadata can name other variants later.
- **R4** — Every downloaded component has an approved URL, byte size, and SHA-256. Setup
  rejects incomplete, altered, wrong-platform, and path-traversing content before use.
- **R5** — The installer accepts a one-time activation code, creates a durable machine
  identity, and exchanges the code for a narrowly scoped credential unique to that worker.
- **R6** — The credential is protected with Windows DPAPI, is never logged or passed on a
  command line, and remains outside replaceable version directories. MARP can revoke one
  machine without affecting another.
- **R7** — Setup installs versions side by side, points a stable launcher at the active
  version, keeps persistent configuration and caches separately, and can uninstall cleanly.
- **R8** — The installed worker starts in the interactive user session at Windows sign-in,
  authenticates automatically, and permits visible watched-job windows by default.
- **R9** — Models remain outside the installer and are downloaded, verified, and cached from
  MARP_API when a job requests them.
- **R10** — Setup reports understandable progress and actionable failures and finishes by
  verifying worker startup, API enrollment, GPU discovery, and local health.
- **R11** — The first milestone produces an unsigned development installer usable for a
  real two-computer API job. Volunteer releases remain blocked on later code signing.
- **R12** — The installation layout and manifest support later operator-requested atomic
  updates and rollback without requiring that full update workflow in the first pilot.

## Open assumptions

- [x] **A1 · product/UI · blocking** — answered 2026-09-14: the volunteer clicks one setup
  application; dependency installation is entirely handled by setup.
- [x] **A2 · distribution · blocking** — answered 2026-09-14: the initial installer should
  be small; large runtimes are downloaded during setup rather than embedded in it.
- [x] **A3 · environment · blocking** — answered 2026-09-14: Windows x64 first, per user,
  start at sign-in, CUDA 12.6 first, with other runtime variants possible later.
- [x] **A4 · security/API contract · blocking** — answered 2026-09-14: a one-time activation
  code becomes a unique revocable machine credential protected with DPAPI.
- [x] **A5 · distribution/API contract · blocking** — answered 2026-09-14: MARP_API approves
  release metadata; immutable packages may be downloaded from GitHub Releases.
- [x] **A6 · release · blocking** — answered 2026-09-14: unsigned development installers are
  acceptable for the pilot; public volunteer releases require code signing.
- [x] **A7 · scope · blocking** — answered 2026-09-14: prove installation and a real job on a
  second computer first; production automatic rollout follows as a separate milestone.

## Decisions

- **2026-09-14** — Reuse installer, launcher, activation, and manifest work preserved on
  `backup-11-installer-scope`; remove its monolithic bundled-runtime approach.
- **2026-09-14** — Treat bootstrap size and installed/downloaded size as separate facts. The
  setup experience is one click even though GPU inference dependencies are necessarily large.

## Plan

1. Extract the existing installer, launcher, credential, and manifest code from the backup.
2. Replace the bundled multi-gigabyte payload with verified component downloads.
3. Implement the matching one-time activation endpoint and machine-bound credential.
4. Register per-user start-at-sign-in and retain visible watch mode as the default.
5. Build an unsigned development installer and write the focused verification plan.
6. After plan approval, install it on the second computer and run one real API-assigned job.

## Acceptance criteria

- Opening the development installer on a clean supported Windows computer is the only manual
  installation action.
- The machine appears separately in the MARP worker pool with its real GPU capabilities.
- Restarting Windows starts the worker without another activation or login.
- A real assigned inference job downloads its model from MARP_API, runs, reports progress,
  and displays its viewer when requested.
- Removing or revoking that worker credential prevents only that machine from taking work.

## Test plan

Written at G3 after the pilot implementation is assembled. It will name focused manifest,
download, activation, credential, launcher, and clean-machine installation checks.

## Status

- **Gate:** implementing
- **Notes:** All material product, distribution, security, and first-platform decisions were
  answered during issue #11 and carried here. The old branch is source material, not a base:
  issue #18 starts from current `develop` and deliberately excludes its monolithic payload.
