# test_model_cache_hashing.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Tests for model artifact hash verification in the MARP Inference Worker.
#
# The first version of model_cache computed a sha256 and never compared it to
# anything, so a truncated download, a proxy error page saved as a .pt, or the
# wrong model entirely would be cached under the right name and handed to an
# engine without complaint. These tests are what make R13 real: the fix itself
# is unobservable without them, because the broken code and the fixed code
# behave identically on a file that happens to be correct.
#
# Kept in its own file rather than appended to test_model_cache.py, which covers
# cache-path planning and is a different concern.

# hashlib computes the real hashes these tests compare against.
import hashlib

# Path types the temporary artifact files.
from pathlib import Path

# UUID gives each test its own cache entry, so a test cannot pass by finding a
# previous run's artifact already present.
from uuid import uuid4

# Pytest checks the expected refusals.
import pytest

# The module under test.
from marp_inference_worker.models import model_cache

# ModelSpec carries the artifact locator and the claimed hash.
from marp_inference_worker.models.model_spec import ModelSpec


# _spec_for()
# Builds a model spec pointing at a local file with a stated hash.
# Inputs: the artifact path and the sha256 to claim.
# Output: a validated ModelSpec with a unique model id.
# Use this so each case gets its own cache key.
def _spec_for(artifact_path: Path, sha256: str | None) -> ModelSpec:

    return ModelSpec(
        model_id=f"hash_test_{uuid4().hex}",
        engine="mock",
        model_arch="mock_detector",
        task="detect",
        artifact={"url": str(artifact_path), "format": "mock", "sha256": sha256},
    )


# test_correct_hash_is_verified_and_reported()
# Verifies the success path of hash checking.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves R13's happy case: the hash is checked, and the cache state says so
# rather than leaving the caller to guess whether it was.
def test_correct_hash_is_verified_and_reported(tmp_path: Path) -> None:

    contents = b"correct model bytes"
    source = tmp_path / "good.pt"
    source.write_bytes(contents)
    expected = hashlib.sha256(contents).hexdigest()

    cache_state = model_cache.ensure_artifact_cached(_spec_for(source, expected))

    assert cache_state["sha256"] == expected
    assert cache_state["sha256_verified"] is True
    assert Path(cache_state["artifact_path"]).is_file()


# test_wrong_hash_is_refused()
# Verifies the failure path of hash checking.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R13, and this is the test that would have caught the original defect.
# The bytes are fine; the stated hash is not the hash of those bytes. The old
# code accepted this without a word and handed the file to an engine.
def test_wrong_hash_is_refused(tmp_path: Path) -> None:

    source = tmp_path / "mislabelled.pt"
    source.write_bytes(b"some model bytes")

    # A hash that is valid in shape and wrong in fact.
    wrong = "a" * 64

    with pytest.raises(model_cache.ArtifactHashMismatch) as raised:
        model_cache.ensure_artifact_cached(_spec_for(source, wrong))

    # The message names the expected hash, so the mismatch is diagnosable.
    assert wrong in str(raised.value)


# test_a_mismatched_artifact_is_removed_so_a_retry_can_refetch()
# Verifies cleanup after a mismatch.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves the recovery. Leaving the bad file cached would mean every retry found
# it already present and failed the same way forever, with no way out but
# clearing the cache by hand on the worker machine.
def test_a_mismatched_artifact_is_removed_so_a_retry_can_refetch(tmp_path: Path) -> None:

    source = tmp_path / "bad.pt"
    source.write_bytes(b"wrong bytes")

    spec = _spec_for(source, "b" * 64)
    cache_path = model_cache.get_artifact_cache_path(spec)

    with pytest.raises(model_cache.ArtifactHashMismatch):
        model_cache.ensure_artifact_cached(spec)

    # Nothing left behind for a retry to find.
    assert not cache_path.is_file()


# test_an_already_cached_artifact_is_verified_not_assumed()
# Verifies that the cached-file shortcut still checks.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves the early-return path is guarded too, and this is the branch the
# original defect was most dangerous in: a file already in the cache was
# returned without any check at all, so corruption after the first successful
# write -- a full disk truncating it, say -- was permanently invisible.
def test_an_already_cached_artifact_is_verified_not_assumed(tmp_path: Path) -> None:

    contents = b"model that will be corrupted"
    source = tmp_path / "will-corrupt.pt"
    source.write_bytes(contents)
    expected = hashlib.sha256(contents).hexdigest()

    spec = _spec_for(source, expected)

    # First call caches it successfully.
    cache_state = model_cache.ensure_artifact_cached(spec)
    cache_path = Path(cache_state["artifact_path"])
    assert cache_path.is_file()

    # Corrupt the cached copy in place, as a truncating write would.
    cache_path.write_bytes(b"truncated")

    # The second call must notice, rather than taking the early return.
    with pytest.raises(model_cache.ArtifactHashMismatch):
        model_cache.ensure_artifact_cached(spec)


# test_hash_comparison_is_case_insensitive()
# Verifies tolerance of hash casing.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves a coordinator sending upper-case hex is not treated as a mismatch --
# the casing carries no information, so refusing it would be a false alarm.
def test_hash_comparison_is_case_insensitive(tmp_path: Path) -> None:

    contents = b"case test bytes"
    source = tmp_path / "case.pt"
    source.write_bytes(contents)
    expected = hashlib.sha256(contents).hexdigest()

    cache_state = model_cache.ensure_artifact_cached(_spec_for(source, expected.upper()))
    assert cache_state["sha256_verified"] is True


# test_spec_without_a_hash_is_measured_but_reported_unverified()
# Verifies the optional-hash case.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves the distinction is not blurred. ModelSpec.artifact.sha256 is optional
# for the local-development frame routes, so a spec without one is allowed --
# but the response says plainly that nothing was verified, and the job runner
# requires a hash on every job's model regardless.
def test_spec_without_a_hash_is_measured_but_reported_unverified(tmp_path: Path) -> None:

    contents = b"unhashed model bytes"
    source = tmp_path / "unhashed.pt"
    source.write_bytes(contents)

    cache_state = model_cache.ensure_artifact_cached(_spec_for(source, None))

    # Measured, so the caller can see what it got.
    assert cache_state["sha256"] == hashlib.sha256(contents).hexdigest()

    # But not verified, and it says so rather than implying it was.
    assert cache_state["sha256_verified"] is False


# test_verify_sha256_returns_the_digest_on_success()
# Verifies the standalone helper.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves the helper behaves the same way in isolation, so a future caller cannot
# get a silently different answer from the one ensure_artifact_cached() gets.
def test_verify_sha256_returns_the_digest_on_success(tmp_path: Path) -> None:

    contents = b"helper test bytes"
    path = tmp_path / "helper.bin"
    path.write_bytes(contents)
    expected = hashlib.sha256(contents).hexdigest()

    assert model_cache.verify_sha256(path, expected) == expected

    with pytest.raises(model_cache.ArtifactHashMismatch):
        model_cache.verify_sha256(path, "c" * 64)


# test_large_file_is_hashed_in_chunks()
# Verifies that hashing does not load the whole file.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves the streaming path is correct, not just present: a file larger than one
# chunk must hash to the same value a single read would produce.
def test_large_file_is_hashed_in_chunks(tmp_path: Path) -> None:

    # Two and a bit chunks, so the loop runs more than once.
    contents = b"x" * (1024 * 1024 * 2 + 12345)
    path = tmp_path / "large.bin"
    path.write_bytes(contents)

    assert model_cache.compute_sha256(path) == hashlib.sha256(contents).hexdigest()


# test_download_uses_a_timeout()
# Verifies that fetches cannot hang forever.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves the other half of the R13 fix. The original used urlretrieve, which
# takes no timeout, so a stalled connection hung the worker indefinitely -- and
# a worker stuck in a fetch keeps its lease alive by heartbeat while doing
# nothing, which is the worst of both outcomes.
#
# Asserted against the source because the alternative is a test that stalls a
# real socket for the length of a real timeout, which is not a fast-tier test.
def test_download_uses_a_timeout() -> None:

    import inspect

    source = inspect.getsource(model_cache.download_artifact)

    # A timeout is passed, and it comes from the module's own constant rather
    # than being written in at the call site.
    assert "timeout=_FETCH_TIMEOUT_S" in source

    # And urlretrieve, which cannot take one, is no longer imported. Checked as
    # an import rather than as a word, because the module's own comments explain
    # why it was removed and a substring check would match that prose.
    module_source = inspect.getsource(model_cache)
    assert "from urllib.request import" not in module_source
    assert "urlretrieve(" not in module_source


# test_download_lands_on_a_partial_name_first()
# Verifies that an interrupted fetch leaves nothing usable behind.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves a deliberate detail. Writing straight to the final path means an
# interrupted download leaves a partial file the next run finds and trusts. The
# hash check would catch it, but only after a wasted load, and on a spec with no
# hash it would not be caught at all.
def test_download_lands_on_a_partial_name_first() -> None:

    import inspect

    source = inspect.getsource(model_cache.download_artifact)

    # Written to a .partial name, renamed on completion, and removed on failure.
    assert ".partial" in source
    assert "partial_path.replace(destination)" in source
    assert "partial_path.unlink(missing_ok=True)" in source
