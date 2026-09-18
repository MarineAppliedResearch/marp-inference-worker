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

- [x] **A2 · cross-repository · blocking** — answered 2026-09-17 by reading
      `MARP_API`, and independently by the desktop session: **safe, build the
      retry.** `requiredEnum` throws in the service layer before
      `publishResult` is reached, so a refused result writes nothing — the
      attempt keeps its live state, worker and epoch. A later retry is therefore
      a first report, not a replay: `isReplay` is false because the state is not
      terminal, `leaseRefusalReason` checks four things and no wall clock, and
      if all four hold it publishes normally. If the sweeper took the lease back
      meanwhile the answer is `{action:'abandon', accepted:false}`, which the
      client already understands. The window is the lease, not forever — a
      restart after the sweeper is told to let go and the work is still lost.
      Recovering beyond the lease is a second change, on the `MARP_API` side,
      and deliberately not in scope here.
- [x] **A3 · behavioural / database · blocking** — answered 2026-09-17: **(b)**.
      A stopped job returns to `queued` with the remaining range, and another
      volunteer carries on from `completed_through_frame`. A stop is a normal
      event in a volunteer pool, not a decision to abandon the range. This is
      what makes `completed_through_frame` earn its place: something later reads
      it to resume from.
- [ ] **A4 · database/schema · blocking** — does a yield consume an attempt?
      `selectClaimableJobRow` filters on `attempts_made < max_attempts`, so if a
      yield increments `attempts_made` like a failure, a job stopped by a few
      volunteers in turn stops being claimable while most of its range is
      unrun — the pool would quietly stop finishing exactly the jobs that get
      passed around most. A yield is not a failure and probably should not
      count, but that is a coordinator decision. `MARP_API`'s change, my
      requirement.
- [ ] **A5 · API contract · blocking** — per-job ingest idempotency versus
      ingesting a job in segments. `POST /jobs/:id/ingest` documents itself as
      "idempotent per job: a job whose observations are already present reports
      `already_ingested` and writes nothing", and `published_attempt_id` on
      `gpu_jobs` holds exactly one attempt. Under (b) one job is finished by
      several attempts and must ingest more than once — which is goal 4. So the
      idempotency key has to move from the job to the attempt or the frame
      range, or the second volunteer's work is refused as already ingested.
      Raised now because it changes the same migration.


## Decisions

- **2026-09-17** — A stopped job is requeued with its remaining range rather than
  cancelled. Isaac's call, answering A3. Consequence: one job is now finished by
  several attempts in sequence, which is what raises A4 and A5 — neither was
  visible while a job had exactly one publishing attempt.
- **2026-09-17** — Adding `yielded` to `RESULT_OUTCOMES` alone is not enough and
  is worse than doing nothing: `gpu_job_attempts` has no `outcome` column, so the
  outcome is the attempt state. `ATTEMPT_STATES` and the
  `gpu_job_attempts_state_check` constraint must grow the word too, and
  `publishResult` needs a branch of its own — its `else` hardcodes
  `state='cancelled'`. Without all three, a stop stops returning 400 and starts
  quietly recording a cancel, erasing the distinction the fourth word exists to
  make.
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
   is reported at restart instead of expiring. A2 answered — safe to build.
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
  implemented. A1, A2 and A3 are answered; A4 and A5 are open and blocking, and both were
  raised by A3's answer. Token held in `.marp/local/coordinator.md`, verified
  against the live pool: `GET /api/v2/gpu/workers` answers 200 and shows the
  desktop's worker. Local venv proven — torch 2.11.0+cu128, CUDA on the RTX
  5060. The desktop's
  API is reachable from this network at the address Isaac gave — 200 on root,
  401 on the GPU family, so the forward and the auth gate are both proven. The
  family is mounted under `/api/v2/gpu`, not the `/api/gpu` its `path:` fields
  declare; `registerVersionedRoute` rewrites it. No worker token yet. Networking to the desktop's API
  is arranged but not yet proven — no live round trip has been run from this
  machine.
