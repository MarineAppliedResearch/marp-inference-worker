# model_cache.py
# Created: 2026-07-11
# Author: Isaac Travers
#
# Model artifact cache for the MARP Inference Worker.
# This file computes deterministic cache paths and ensures model artifacts are
# present locally, complete and correct, before an engine loads them.
#
# The correctness part is the point of this module and was the bug in its first
# version: it computed a sha256 and never compared it to anything, so a truncated
# download, a proxy error page saved as a .pt, or the wrong model entirely would
# be cached under the right name and loaded without complaint. Every path out of
# ensure_artifact_cached() now verifies the hash before returning (R13).
#
# Cache layout, fetching and verification belong here. Engine loading does not.

# hashlib verifies a cached artifact against its expected sha256.
import hashlib

# shutil copies local-file artifacts into the cache.
import shutil

# URL parsing identifies local versus remote artifact locators.
from urllib.parse import urlparse

# Path handles filesystem-safe cache path creation and file checks.
from pathlib import Path

# ModelSpec provides validated artifact metadata and model identity.
from marp_inference_worker.models.model_spec import ModelSpec


# Root folder used for local model artifact cache storage.
_CACHE_ROOT = Path("models") / "cache"


# How long a fetch may stall before it is abandoned, in seconds.
# The first version used urlretrieve with no timeout at all, which on a stalled
# connection hangs the worker indefinitely -- and a worker stuck in a fetch keeps
# its lease alive by heartbeat while doing nothing.
_FETCH_TIMEOUT_S = 60.0


# How many bytes are read at a time when hashing or downloading.
_CHUNK_BYTES = 1024 * 1024


# ArtifactHashMismatch
# Raised when a cached or fetched artifact does not have its expected sha256.
# A distinct type so a caller can tell "the bytes are wrong" apart from "the
# fetch failed", and re-fetch in the first case.
class ArtifactHashMismatch(RuntimeError):
    pass


# compute_sha256()
# Computes a file's sha256 without loading it into memory.
# Inputs: path to the file.
# Output: lower-case hex digest.
# Use this for both verification and identity; model files are large enough that
# reading them whole matters.
def compute_sha256(path: Path) -> str:

    # Stream in chunks so a multi-gigabyte artifact costs one chunk of memory.
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# verify_sha256()
# Checks a file against an expected sha256 and raises when it does not match.
# Inputs: path to the file and the expected lower-case hex digest.
# Output: the digest that was computed.
# Use this before any artifact reaches an engine. Comparison is case-insensitive
# because a coordinator may send either case, but nothing else is forgiven.
def verify_sha256(path: Path, expected_sha256: str) -> str:

    # Compute what is actually on disk.
    actual = compute_sha256(path)

    # Mismatch means the bytes are not the model that was asked for.
    if actual.lower() != expected_sha256.lower():
        raise ArtifactHashMismatch(
            f"artifact {path} has sha256 {actual}, expected {expected_sha256}"
        )
    return actual


# get_cache_key()
# Builds a deterministic cache key for a model artifact.
# Inputs: validated ModelSpec object.
# Output: cache key string.
# Use this so different hashes of the same model ID do not collide.
def get_cache_key(model_spec: ModelSpec) -> str:

    # Use the immutable artifact hash when available.
    if model_spec.artifact.sha256:
        return f"{model_spec.model_id}_{model_spec.artifact.sha256}"

    # Fall back to model ID when a hash is not provided.
    return model_spec.model_id


# get_artifact_cache_path()
# Builds the expected local artifact path in the cache.
# Inputs: validated ModelSpec object.
# Output: Path to cached artifact file.
# Use this to keep cache layout stable across route and manager calls.
def get_artifact_cache_path(model_spec: ModelSpec) -> Path:

    # Resolve the model-specific cache subfolder key.
    cache_key = get_cache_key(model_spec)

    # Derive the output filename from the artifact locator.
    artifact_name = Path(urlparse(model_spec.artifact.url).path).name
    if not artifact_name:
        artifact_name = f"{cache_key}.bin"

    # Return models/cache/<key>/<artifact_name>.
    return _CACHE_ROOT / cache_key / artifact_name


# is_remote_url()
# Checks whether an artifact locator should be downloaded over HTTP(S).
# Inputs: artifact locator string.
# Output: True when locator is HTTP/HTTPS, otherwise False.
# Use this to choose download versus local file copy behavior.
def is_remote_url(locator: str) -> bool:

    # Parse locator and classify only HTTP and HTTPS as remote.
    parsed_locator = urlparse(locator)
    return parsed_locator.scheme in {"http", "https"}


# download_artifact()
# Fetches a remote artifact to a local path, with a timeout.
# Inputs: source url and destination path.
# Output: none.
# Use this rather than urlretrieve, which takes no timeout. The download lands
# on a temporary name and is renamed on completion, so an interrupted fetch
# cannot leave a partial file that the next run would find and trust.
def download_artifact(url: str, destination: Path) -> None:

    # Imported here because the API process may never fetch anything.
    import httpx

    # Write beside the destination so the rename is on the same filesystem.
    partial_path = destination.with_suffix(destination.suffix + ".partial")

    # Stream to disk; a model file is far too large to hold in memory.
    try:
        with httpx.stream("GET", url, timeout=_FETCH_TIMEOUT_S, follow_redirects=True) as response:
            response.raise_for_status()
            with partial_path.open("wb") as file_handle:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    file_handle.write(chunk)
    except Exception:
        # Leave nothing behind that a later run could mistake for a complete
        # download; the hash check would catch it, but only after wasting a load.
        partial_path.unlink(missing_ok=True)
        raise

    # Only now does the file get its real name.
    partial_path.replace(destination)


# ensure_artifact_cached()
# Ensures a model artifact exists at its cache path and is the right bytes.
# Inputs: validated ModelSpec object.
# Output: cache state dictionary for API responses and manager state.
# Use this before engine load, so engines always receive a local path to an
# artifact whose hash has been checked.
def ensure_artifact_cached(model_spec: ModelSpec) -> dict[str, object]:

    # Compute deterministic cache key and artifact target path.
    cache_key = get_cache_key(model_spec)
    artifact_path = get_artifact_cache_path(model_spec)
    expected_sha256 = model_spec.artifact.sha256

    # Ensure parent cache folders exist before copy/download.
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    # An already-cached artifact is still verified, not assumed. A file can be
    # truncated by a full disk after it was written, and the cache key contains
    # the hash, so a mismatch here means the file is corrupt rather than stale.
    if artifact_path.is_file():
        verified_sha256 = _verify_if_expected(artifact_path, expected_sha256)
        return {
            "cache_key": cache_key,
            "artifact_path": str(artifact_path),
            "is_cached": True,
            "cache_action": "cached",
            "sha256": verified_sha256,
            "sha256_verified": expected_sha256 is not None,
        }

    # Fetch or copy the artifact into the cache path.
    if is_remote_url(model_spec.artifact.url):
        download_artifact(model_spec.artifact.url, artifact_path)
        cache_action = "downloaded"
    else:
        # Copy local artifacts into cache for local development workflows.
        source_path = Path(model_spec.artifact.url)
        if not source_path.is_file():
            raise FileNotFoundError(f"Model artifact does not exist: {source_path}")
        shutil.copy2(source_path, artifact_path)
        cache_action = "copied"

    # Verify what arrived before anything is allowed to load it. On a mismatch
    # the bad file is removed, so a retry re-fetches rather than finding it
    # cached and failing the same way forever.
    try:
        verified_sha256 = _verify_if_expected(artifact_path, expected_sha256)
    except ArtifactHashMismatch:
        artifact_path.unlink(missing_ok=True)
        raise

    # Return normalized cache state metadata.
    return {
        "cache_key": cache_key,
        "artifact_path": str(artifact_path),
        "is_cached": artifact_path.is_file(),
        "cache_action": cache_action,
        "sha256": verified_sha256,
        "sha256_verified": expected_sha256 is not None,
    }


# _verify_if_expected()
# Verifies an artifact when a hash was supplied, or just measures it when not.
# Inputs: path to the artifact and the expected digest or None.
# Output: the artifact's sha256.
# Use this so the two return paths above verify identically.
#
# A spec with no sha256 is still hashed, and the digest is returned and reported
# so a caller can see what it got -- but it cannot be verified against anything.
# ModelSpec.artifact.sha256 is optional for the local-development frame routes;
# a job spec's model always carries one, and the job runner requires it.
def _verify_if_expected(path: Path, expected_sha256: str | None) -> str:

    # With an expectation, a mismatch is fatal.
    if expected_sha256:
        return verify_sha256(path, expected_sha256)

    # Without one, report the digest and let the caller decide.
    return compute_sha256(path)
