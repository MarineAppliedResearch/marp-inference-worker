---
task: MarineAppliedResearch/marp-inference-worker#NN
repos: [marp-inference-worker, MARP_API]
status: implementing
needs: []
---

## Goal

One computer is one worker in the pool, however many times the worker has been installed on
it. Today the pool shows 31 workers for four machines: a machine that loses its state
directory comes back as a stranger, and the Windows build check enrols a brand new worker
every time it runs. Somebody looking at the Workers screen cannot tell which rows are
computers and which are debris.

## Requirements

- **R1** — A worker's `local_id` is derived from the machine, so the same computer enrols
  as the same worker after a reinstall or a lost state directory.
- **R2** — An existing install keeps the `local_id` it already has. The stored identity
  file stays authoritative when it is present, so the machines already in the pool do not
  fork on upgrade.
- **R3** — The derived id is a UUID string, exactly the shape the column and the
  coordinator already accept, and it does not expose the raw machine identifier.
- **R4** — A worker's name is its hostname. The random suffix that made four rows for one
  computer look like four computers is gone.
- **R5** — A machine that can offer no stable identifier still enrols, with a random id, as
  it does today. Losing identity must never stop a volunteer working.
- **R6** — The Windows build check does not enrol a worker into the pool.
- **R7** — Duplicate rows already in a database can be merged onto one row per machine
  without losing which attempts ran where, and the synthetic rows can be removed. This is
  `MARP_API`'s side.

## Open assumptions

- [x] **A1 · architectural · blocking** — answered 2026-10-15: derive `local_id` from the
      machine — `MachineGuid` under `HKLM\SOFTWARE\Microsoft\Cryptography` on Windows,
      `/etc/machine-id` on Linux — rather than minting a random one per state directory.
- [x] **A2 · behavioural · blocking** — answered 2026-10-15: an existing identity file
      wins, so the four machines in the pool keep their current rows. The consequence is
      that a machine forks once more if it ever loses its state directory, and is stable
      from then on. Accepted as the low-risk path.
- [x] **A3 · destructive · blocking** — answered 2026-10-15: clean up the rows that exist.
      `gpu_job_attempts.worker_id` is `RESTRICT` and the real machines carry 6,792
      attempts, so duplicates are **merged** — attempts repointed at the surviving row —
      and never deleted. The 17 `buildcheck` rows hold no attempts and are deleted.
- [x] **A4 · scope · blocking** — answered 2026-10-15: the build-check change rides with
      this work. Isaac asked for the fix and the cleanup without separating it, and a
      cleanup that leaves the build check enrolling would be undone by the next build.
- [ ] **A5 · destructive · non-blocking** — The three `jest-*` rows hold 50 attempts
      between them. They are test leftovers, but deleting them deletes those attempts.
      Proposed: leave them to a fix in the suite that created them, and report rather than
      delete.
- [ ] **A6 · security · non-blocking** — A merged-away worker's `service_tokens` row is
      left behind. Revoking it is probably right; it is not needed to make the pool
      readable.

## Decisions

- **2026-10-15** — The derived id is `uuid5` over a fixed namespace and the machine's own
  identifier. That keeps `local_id` a UUID string — unchanged for the column, the
  coordinator and every existing test — while making it a pure function of the machine,
  and it means the raw `MachineGuid` never leaves the computer.
- **2026-10-15** — The coordinator side was already right. `gpu.repository.js` upserts on
  `local_id` and its comment explains that keying on the name was the earlier bug. Nothing
  there changes; the worker simply has to supply an id worth keying on.

## Plan

1. `jobs/identity.py` — derive the machine id, fall back to random, keep the stored file
   authoritative.
2. `installation/activation.py` — the name is the hostname.
3. `packaging/build-windows.ps1` — stop enrolling.
4. Tests in `tests/test_worker_identity.py`, proved red first.
5. `MARP_API` — a checked-in cleanup script, dry run by default.

## Acceptance criteria

- Two enrolments from the same machine with no stored identity produce the same `local_id`.
- An existing identity file is returned unchanged.
- A machine with no stable identifier still gets an id and still enrols.
- The name carries no random suffix.
- After the cleanup, one row per machine, with every attempt still attributed.

## Test plan

Filled at G3.

## Status

- **Gate:** implementing
- **Notes:** The design was settled in conversation on 2026-10-15; the assumptions above
  record it. No issue number yet — this needs one, and the branch is named descriptively
  until it has one.
