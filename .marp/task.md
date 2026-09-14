---
task: MarineAppliedResearch/MARP_API#187
repos: [MARP_API, marp-inference-worker]
status: implementing
needs: []
---

## Goal

The worker downloads model weights from MARP API with its service credential, verifies
them, and reuses the verified local cache when the same model version is requested again.

## Requirements

- **R1** — Relative API artifact URLs are resolved against the configured coordinator.
- **R2** — Model downloads present the worker's existing bearer service credential.
- **R3** — Downloaded and cached model bytes must match the job's SHA-256 before loading.
- **R4** — A verified cached model is reused without another request.
- **R5** — Existing public HTTP and local-file model cache callers remain compatible.

## Open assumptions

- [x] **A1 · cross-repository · blocking** — answered 2026-09-14: MARP API serves the
  local artifact; workers never depend on its filesystem path.
- [x] **A2 · security/permissions · blocking** — answered 2026-09-14: reuse the worker's
  existing MARP API bearer credential.

## Decisions

- **2026-09-14** — Keep transport in the existing model cache and pass authorization only
  for coordinator-hosted artifact locators.

## Plan

1. Resolve API-relative artifact URLs through the coordinator client.
2. Allow streamed model downloads to carry request headers.
3. Cover authenticated download and cache reuse with focused tests.

## Acceptance criteria

- A relative model locator downloads from the coordinator with its bearer credential.
- A repeat run uses verified cached bytes without reaching the server.
- A hash mismatch is removed and refused as before.

## Test plan

See `.marp/verification.md`; awaiting human review before execution.

## Status

- **Gate:** ready-for-pr
- **Notes:** 72 focused tests pass, the suffix regression set passes 19/19, and two real
  API/Jellyfin/CUDA attempts succeeded with the verified model cache.
