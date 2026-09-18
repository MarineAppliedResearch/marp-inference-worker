---
task: MarineAppliedResearch/marp-inference-worker#32
repos: [marp-inference-worker]
status: implementing
needs: []
---

## Goal

A volunteer on Linux can enrol their machine. Today they cannot: the worker encrypts its
credential with Windows DPAPI and raises on any other platform, so activation reaches the
coordinator, receives a credential, and then dies writing it to disk. The machine is left
holding a local id and nothing that can authenticate, and the single-use activation code is
already spent. No Linux worker has ever reached the job loop, and none can until the
credential has somewhere to live.

## Requirements

- **R1** — On a non-Windows platform the worker stores its credential in the state
  directory with owner-only permissions (0600), and reads it back.
- **R2** — Windows behaviour is unchanged. DPAPI remains the path on `win32`; this adds a
  branch rather than replacing one.
- **R3** — `protect()` states its own threat model in words: what the filesystem protects
  against, what it does not, and that the reduction from DPAPI is deliberate.
- **R4** — A credential file created by this path is not readable by another user on the
  machine, and the permissions are set before the secret reaches the file rather than
  after.
- **R5** — A round trip through `save()` and `load()` returns the credential unchanged, and
  `load()` on a machine that has never activated returns `None` rather than raising.

## Open assumptions

- [x] **A1 · security/permissions · blocking** — answered 2026-09-18: how should a Linux
  worker hold its credential at rest, given Linux has no DPAPI equivalent? **0600 file
  permissions in the state directory.** Isaac's decision, confirmed directly and relayed
  through MARP-DESKTOP-DEV. The alternatives were Secret Service/libsecret and a key
  derived from `/etc/machine-id`. libsecret is the closest analogue to DPAPI but needs an
  unlocked keyring, which a headless autologin machine does not have — demonstrated on this
  box today, where GDM autologin left the login keyring locked. *An option that fails on
  the target hardware is not the more secure option, it is the one that does not ship.* The
  machine-id derivation was rejected as obfuscation: anything that can read the credential
  can read `/etc/machine-id`. Isaac attached a condition — the threat model goes in the
  code, not left implied. That condition is R3.

- [x] **A2 · architectural · blocking** — answered 2026-09-18: does this replace DPAPI or
  sit alongside it? **Alongside.** A `sys.platform` branch. Windows keeps DPAPI exactly as
  it is. Confirmed by MARP-DESKTOP-DEV in the issue.

- [x] **A3 · security/permissions · blocking** — answered 2026-09-18 by Isaac: **option 3,
  weaken it to the round trip only.** Asked twice and confirmed, so it is a decision rather
  than a slip. The test is renamed `test_credential_round_trip`, because a test named for an
  assertion it no longer makes is worse than no test — and a comment above it records what
  was dropped and where the replacement lives.

  Recorded because both I and MARP-DESKTOP-DEV recommended option 2 and were overruled: the
  cost is that **no test now asserts DPAPI encrypts anything on Windows**. The Linux mode
  assertion in `tests/test_credential_store.py` does not substitute for it — different
  guarantee, different platform. If Windows coverage is revisited, that is the gap.

  The question, as originally raised during G2:
  `tests/test_installed_worker.py:50` is `test_dpapi_credential_round_trip_is_not_plaintext`,
  and line 56 asserts `secret.encode("utf-8") not in path.read_bytes()` — *the credential is
  never plaintext on disk*. That is an existing test asserting the exact property A1
  deliberately gives up on Linux, so on this platform it cannot pass. **What should happen
  to it?**

  It already fails on `develop` here, before any change of mine, because `protect()` raised
  rather than returned — so this is not a regression I introduced. But it stops being a
  platform gap and starts being a contradiction the moment 0600 lands, and it should be
  settled deliberately rather than left red.

  Three ways, and the choice is a statement about what the test is for:
  1. **Mark it `skipif(sys.platform != "win32")`.** The function is named for DPAPI and is
     testing DPAPI; on a platform without DPAPI it is not applicable. Cuts against the
     testing doctrine's *a skipped suite looks green*, though what is skipped here is
     genuinely absent rather than merely unavailable.
  2. **Split it in two** — keep the not-plaintext assertion for `win32`, and assert the
     0600 property off it. Costs a little duplication with
     `tests/test_credential_store.py`, which already asserts the Linux half.
  3. **Weaken the assertion to the round trip only**, dropping the not-plaintext check.
     Cheapest, and the worst: it silently removes the assertion that DPAPI is doing
     anything at all on Windows, which is the one thing that test exists to prove.

  My read is **2**, because it keeps the Windows guarantee asserted rather than skipped,
  and 3 would quietly delete a real security assertion on the platform where it still
  holds. Not acting on that read — it is a security assertion and the call is Isaac's.

## Decisions

- **2026-09-18** — 0600 file permissions on non-Windows, DPAPI retained on Windows. The
  threat model is stated in `protect()` rather than implied. Durable enough to promote to a
  decision record if the credential store is revisited; left here for now because it
  records one platform branch rather than an architecture.

- **2026-09-18** — the file is opened with `O_CREAT | O_EXCL` and mode `0o600` so the
  permissions exist before the secret does. Writing then `chmod`-ing leaves a window where
  the credential is on disk world-readable, which on a multi-user box is the whole of the
  defence missing for as long as it takes to run the next line.

## Plan

1. Branch `32-linux-credential-store` off `develop` at `1d0f50f`. *(done)*
2. Add the non-Windows branch to `protect()` and `unprotect()`, with the threat model
   written into `protect()`.
3. Make `save()` create the file with 0600 from the moment it exists, not after.
4. Tests at the tier that can see it: a round trip, the permission bits, `load()` on a
   machine that never activated, and that the Windows path is untouched.
5. Report to MARP-DESKTOP-DEV before activating — worker 1081 holds this machine's
   `local_id`, so a fresh activation returns 409 until that row is cleared.

## Acceptance criteria

- `marp-worker-activate` completes on Linux and leaves a credential the worker can read.
- The credential file is `-rw-------`.
- `unprotect(protect(x)) == x` on Linux.
- `load()` returns `None`, not an exception, before first activation.
- No change to behaviour on `win32`.

## Test plan

Filled at G3. Targeted at `tests/` for the credential store only — not the whole suite,
per the testing doctrine. The Windows path cannot be executed here, so R2 is covered by
asserting the branch is not taken rather than by running DPAPI, and that limit is stated
rather than left to look like coverage.

## Status

- **Gate:** implementing (A3 answered; R1–R5 complete)
- **Notes:** R1–R5 are implemented and committed on this branch, and
  `tests/test_credential_store.py` is 8/8 green. Proven red first: 7 of those 8 fail against
  the original module, the eighth being `load()` before activation, which never reaches the
  DPAPI call and passes either way.

  **A3** turned up during G2, went back to Isaac, and is answered. The contradicting test is
  weakened to the round trip and renamed. `tests/test_installed_worker.py` now runs 11 passed
  where it ran 3 failed, 2 passed on `develop`.

  Not verified end to end: activation cannot be retested until MARP-DESKTOP-DEV clears
  worker 1081 and mints a new code — the previous one is spent. So the first acceptance
  criterion is unproven, and the tests here cover the store rather than the enrolment.

  Found and left alone, in `tests/test_installed_worker.py` and pre-dating this branch:
  `test_update_is_verified_and_staged_beside_the_active_release` and
  `test_update_rejects_path_traversal` both fail on `develop` here. Unrelated to the
  credential store and not investigated.
