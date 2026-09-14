# Verification — MarineAppliedResearch/MARP_API#187 worker

## What each test proves

| Requirement | Test | Tier | Proves |
| --- | --- | --- | --- |
| R1, R2 | `pytest tests/test_model_artifact_auth.py` | unit | Relative URLs use the coordinator origin and bearer token; another origin never receives that token. |
| R2, R3, R4 | `pytest tests/test_model_cache.py tests/test_model_cache_hashing.py` | loopback HTTP + filesystem | The download carries authorization, verifies bytes, removes mismatches, and makes only one request across repeated cache use. |
| R1, R2, R5 | `pytest tests/test_job_runner.py tests/test_coordinator_contract.py` | process + HTTP contract | The runner passes the resolved locator and headers without breaking its coordinator contract. |
| R1-R5 | Short real job sequence in the API verification plan | full system | The shipped runner interoperates with the real route, database, Jellyfin stream, and CUDA engine. |

## Requirements with no test

None.

## Edge cases

- Relative API route, same-origin absolute URL, different-origin URL, local path.
- Valid cached file, corrupt cached file, failed partial download, and hash mismatch.

## Regression coverage

- Existing local-file model loading remains supported for direct development callers.
- Existing unauthenticated public HTTP model sources do not receive the MARP token.

## Known gaps

- Network resume after a worker process restart is outside this focused repair.
- Cache eviction and worker software update delivery remain separate work.

## Manual steps

Covered by the API repository's full-system sequence.

---

## Results

Run 2026-09-14 with Python 3.12.

- The first sandboxed invocation reported verbatim:
  `PermissionError: [WinError 5] Access is denied: 'C:\\Users\\isaac\\AppData\\Local\\Temp\\pytest-of-isaac'`.
  It was rerun outside that filesystem restriction.
- Approved focused set: `72 passed in 36.15s`.
- Post-integration suffix regression set: `19 passed in 1.24s`.
- The first real attempt downloaded and verified the API artifact but failed because its
  cache filename had no `.pt` suffix. After the fix, its coordinator retry succeeded.
- Second real API/Jellyfin/CUDA job: `job=89 state=succeeded seconds=7.22 cache=True`.
