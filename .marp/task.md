---
task: MarineAppliedResearch/marp-inference-worker#20
repos: [marp-inference-worker, MARP_API]
status: design
needs: []
---

## Goal

A volunteer who presses stop on a running job sees it stop, and the work the run
actually completed reaches MARP. Today the stop is reported with a word the
coordinator has never accepted, so it is refused, the attempt is never published,
nothing is ingested, and the volunteer is told nothing. The person stopping a job
is the whole point of the screen-saver behaviour; at the moment it is the one
action that silently loses work.

## Requirements

- **R1** — When an operator stops a job mid-run, the worker reports a terminal
  outcome the coordinator accepts. It does not invent a word that is not in the
  coordinator's vocabulary.
- **R2** — The report carries how far the run actually got, so MARP can tell
  which part of the frame range is real work.
- **R3** — A result the coordinator refuses does not cause the worker to forget
  the attempt. The in-flight record survives, so the attempt is reported rather
  than left to expire against the lease and then the 24h attempt cap.
- **R4** — A refused stop is visible to the operator, not only written into
  `note_error` where nobody looks.
- **R5** — The worker's test asserts what it actually sends on a stop, named
  against the same requirement as `MARP_API`'s test that the coordinator accepts
  it. Neither suite may pass on its own assumption about the other.

## Open assumptions

- [x] **A1 · API contract · blocking** — answered 2026-09-17: **(a)**. The
      coordinator's vocabulary grows a fourth outcome, `yielded`. A run stopped
      early with real work in it is a distinct thing from a cancel, and ingest
      keys on it directly rather than on a nullable frame number. Rejected: (b),
      reporting `cancelled` and carrying the whole signal in
      `completed_through_frame` — `cancelled` already means "called off, nothing
      to keep", and overloading it makes the ingest condition depend on a number
      rather than on what happened. → ADR in the umbrella before this merges; it
      spans two repositories.

- [ ] **A2 · cross-repository · blocking** — R3 changes what a worker does with
      an attempt the coordinator refused. If it keeps the in-flight record and
      reports it at restart, the coordinator sees a second result call for an
      attempt it already refused. Is that idempotent today for a refused attempt,
      or does it need handling on the `MARP_API` side too? Asked of the desktop
      session; it owns that file.

## Decisions

- **2026-09-17** — A stop reports `yielded`, a fourth outcome, rather than
  reusing `cancelled`. Isaac's call, answering A1. The point of no return is
  `MARP_API`'s migration: the check constraint on `gpu_job_attempts.outcome`
  spells its vocabulary out, so adding the word is a migration and removing it
  later is another one.
- **2026-09-17** — Worker side and API side are split: this branch changes
  `runner.py`, `operator_control.py`, `worker_state.py` and `watch/` only. The
  outcome vocabulary, the `completed_through_frame` column and the ingest
  condition belong to `MARP_API` and to the desktop session working there.
- **2026-09-17** — The stop/outcome contract is done before job targeting and
  before repeated ingest. It is small, it is genuinely broken, and both of those
  goals stand on it.

## Plan

A1 is settled, A2 is not. Ordered, and the first two are the whole of R1/R2 here:

1. `runner.py` reports `yielded` only once `MARP_API` accepts it — the two land
   together or the worker is broken against a coordinator that has not shipped.
2. Send `completed_through_frame` as it already computes it, and assert it.
3. R3: keep the in-flight record when the result call is refused, so the attempt
   is reported at restart instead of expiring. Blocked on A2.
4. R4: surface a refused stop to the operator.

## Acceptance criteria

- Pressing stop on a running job produces a terminal result the coordinator
  accepts, and the attempt leaves `leased`.
- The frame the run reached is stored against the attempt and readable back.
- Killing the coordinator during the stop, then restarting the worker, still
  reports the attempt rather than leaving it to expire.
- A test in each repository names R1 and fails if the other side's assumption
  changes.

## Test plan

G3. Not written.

## Status

- **Gate:** design
- **Notes:** Issue #20 filed. Branch cut from `develop` at 56cdcff. Nothing
  implemented. A1 is answered; A2 is open and blocking. Networking to the desktop's API
  is arranged but not yet proven — no live round trip has been run from this
  machine.
