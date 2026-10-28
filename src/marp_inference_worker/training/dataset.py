# dataset.py
# Created: 2026-10-19
# Author: Isaac Travers
#
# Training dataset fetch and cache for the MARP Inference Worker.
# A training job names a dataset by locator and hash, exactly as it names a
# model. This file turns that into a directory on local disk that Ultralytics
# can train from: fetched once, verified, unpacked, and shared by every attempt
# that wants the same bytes.
#
# What belongs here: fetching, verifying, unpacking, and making the dataset's
# own YAML true on this machine. What does not: anything about training, and
# anything about the coordinator. The locator arrives already resolved and
# authorized by the caller.

# hashlib verifies the archive against the hash the job spec carried.
import hashlib

# shutil removes a half-unpacked directory when something goes wrong.
import shutil

# tarfile unpacks the archive. The packaging side writes an uncompressed tar
# deliberately -- a dataset is almost all JPEG, so compression costs minutes of
# CPU and saves nothing -- and tarfile is in the standard library, so a worker
# needs no dependency to read one.
import tarfile

# Path types every directory this module hands back.
from pathlib import Path

# Callable types the injected downloader and progress callback.
from typing import Any, Callable


# How many bytes are read at a time when hashing or downloading.
# A dataset archive is measured in gigabytes, so everything here streams.
_CHUNK_BYTES = 1024 * 1024


# The file that says a cache entry is finished and safe to use.
# Written last, after the unpack and the YAML rewrite. Its absence is what makes
# an interrupted fetch retry rather than train on half a dataset -- a directory
# that merely exists proves nothing about what is in it.
_COMPLETE_MARKER = ".marp-complete"


# The YAML the packaging script writes into every archive.
# A fixed name on purpose: the builders produce `classnames.yaml`,
# `frames-classnames.yaml` and `mixed-classnames.yaml`, and choosing between
# them is a decision made once, at packaging time, by somebody who knows what
# the phases were. The worker is handed the answer.
_DATA_YAML = "data.yaml"


# DatasetHashMismatch
# Raised when a fetched archive is not the bytes the job spec asked for.
# A distinct type so a caller can tell "the bytes are wrong" from "the fetch
# failed", and refuse the job rather than retry forever.
class DatasetHashMismatch(RuntimeError):
    pass


# compute_sha256()
# Computes a file's sha256 without loading it into memory.
# Inputs: path to the file.
# Output: lower-case hex digest.
def compute_sha256(path: Path) -> str:

    # Streamed, because the file is gigabytes.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# cache_entry()
# Where a dataset with this hash lives once unpacked.
# Inputs: the cache root and the expected sha256.
# Output: the directory, which may not exist yet.
# Keyed on the hash rather than the dataset name or id, so two jobs wanting the
# same bytes share one copy and a dataset that is rebuilt is a different entry
# rather than a silent overwrite of the old one (R7).
def cache_entry(cache_root: Path, sha256: str) -> Path:

    return Path(cache_root) / sha256.lower()


# _download()
# Streams a locator to a local file.
# Inputs: url, destination, optional headers, optional progress callback.
# Output: none.
# Use this through ensure_dataset(); it is separate so a test can substitute a
# local copy without standing up an HTTP server.
def _download(
    url: str,
    destination: Path,
    headers: dict[str, str] | None = None,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> None:

    # Imported here so that importing this module costs nothing on a worker that
    # never trains.
    import httpx

    # A long read timeout and no total timeout: a multi-gigabyte fetch on a
    # volunteer's connection is slow by nature, and a job killed for being slow
    # would be killed every time on exactly the machines we most want.
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)

    with httpx.stream("GET", url, headers=headers, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0) or None
        received = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(_CHUNK_BYTES):
                handle.write(chunk)
                received += len(chunk)
                if on_progress is not None:
                    on_progress(received, total)


# _rewrite_data_yaml()
# Points the dataset's YAML at where this machine actually unpacked it.
# Inputs: the unpacked directory.
# Output: path to the rewritten YAML.
#
# The archive carries relative paths (`train: train/images`), which is the only
# honest thing to put in a file that travels. Ultralytics resolves a relative
# path against its own settings directory rather than against the YAML, so
# leaving them relative points training at somewhere that does not exist. They
# are made absolute here, against the directory we chose (#67 R12).
#
# A dataset built by the legacy scripts carries absolute paths from the machine
# that built it, which is the same bug seen from the other end; the packaging
# step strips those, and this puts correct ones back.
def _data_yaml_or_none(unpacked: Path) -> Path | None:

    # Named rather than returned blindly: a classification dataset has no
    # data.yaml, and handing back a path to a file that is not there makes
    # every caller test for existence instead of being told.
    yaml_path = unpacked / _DATA_YAML
    return yaml_path if yaml_path.is_file() else None


def _rewrite_data_yaml(unpacked: Path) -> Path | None:

    yaml_path = unpacked / _DATA_YAML
    if not yaml_path.is_file():
        # **A classification archive has no data.yaml and is not broken.**
        # Ultralytics reads a classify dataset from a DIRECTORY, taking its
        # class vocabulary from the subfolder names, so there is no YAML to
        # rewrite and nothing missing. This refused every classification
        # dataset outright -- job 21798 died here, before any engine code ran.
        #
        # Still a refusal for a DETECTION archive that lost its yaml, which is
        # what the old message was right about. The two layouts are told apart
        # by `train/images`: a detection dataset splits its train directory into
        # `images/` and `labels/`, while a classification one holds a directory
        # per class. Without this check a detection archive missing its yaml
        # would be accepted as a classification dataset with one class called
        # "images" -- a silently wrong run rather than a refusal.
        if (unpacked / "train").is_dir() and not (unpacked / "train" / "images").is_dir():
            return None

        raise FileNotFoundError(
            f"dataset archive carried no {_DATA_YAML}, and its train/ directory "
            "is not a classification layout either"
        )

    # Rewritten line by line rather than through a YAML library, for the same
    # reason the packaging side builds it that way: the file is a names list, an
    # `nc`, and two paths, all written by us. Anything else in it is something
    # we did not write and should be looked at rather than silently re-emitted.
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    for line in lines:
        key = line.split(":", 1)[0].strip() if ":" in line else ""
        if key in {"train", "val"}:
            relative = line.split(":", 1)[1].strip()
            # Already absolute means somebody else has been here; leave it.
            candidate = Path(relative)
            absolute = candidate if candidate.is_absolute() else (unpacked / relative)
            rewritten.append(f"{key}: {absolute.as_posix()}")
        else:
            rewritten.append(line)

    yaml_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return yaml_path


# ensure_dataset()
# Returns a local directory holding this dataset, fetching it once if needed.
# Inputs: the resolved locator, its expected sha256, the cache root, optional
#   request headers, an optional progress callback, and an injectable download.
# Output: mapping with the unpacked directory, the data YAML, and whether this
#   call did the fetching.
# Use this from the training engine's run(), before building a model.
#
# **Shared, not per attempt.** Two jobs training on one dataset on one machine
# read the same copy. A per-attempt copy would be tens of gigabytes of identical
# bytes, which is the mistake already recorded against the job workspace in #65,
# and a dataset is far larger than what filled that disk.
def ensure_dataset(
    url: str,
    sha256: str,
    cache_root: Path,
    headers: dict[str, str] | None = None,
    on_progress: Callable[[int, int | None], None] | None = None,
    download: Callable[..., None] | None = None,
) -> dict[str, Any]:

    expected = str(sha256).lower()
    if not expected:
        raise ValueError("a training dataset must carry the sha256 it is expected to have")

    cache_root = Path(cache_root)
    unpacked = cache_entry(cache_root, expected)
    marker = unpacked / _COMPLETE_MARKER

    # Already here and finished. The marker is the test, not the directory.
    if marker.is_file():
        return {"path": unpacked, "data_yaml": _data_yaml_or_none(unpacked), "fetched": False}

    # A directory without the marker is the wreckage of an interrupted attempt,
    # and whatever is in it cannot be trusted. Start again rather than resume
    # into it: the archive is content-addressed, so starting again is cheap to
    # reason about and a half-unpacked tree is not.
    if unpacked.exists():
        shutil.rmtree(unpacked, ignore_errors=True)

    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / f"{expected}.tar.partial"
    archive.unlink(missing_ok=True)

    fetch = download or _download
    try:
        fetch(url, archive, headers=headers, on_progress=on_progress)

        # Verified before anything is unpacked. A truncated download that became
        # a training run would produce a model trained on part of a dataset and
        # nothing anywhere would say so.
        actual = compute_sha256(archive)
        if actual != expected:
            raise DatasetHashMismatch(
                f"dataset archive has sha256 {actual}, expected {expected}"
            )

        unpacked.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:") as tar:
            # `filter="data"` refuses absolute paths, parent traversal, links and
            # device nodes. The archive is ours, but it arrives over the network
            # and unpacking is the moment where that stops mattering.
            tar.extractall(unpacked, filter="data")

        _rewrite_data_yaml(unpacked)

        # Last, so a crash anywhere above leaves an entry that is visibly unfinished.
        marker.write_text(expected + "\n", encoding="utf-8")

    except Exception:
        # Leave nothing a later run could mistake for a usable dataset.
        shutil.rmtree(unpacked, ignore_errors=True)
        raise
    finally:
        archive.unlink(missing_ok=True)

    return {"path": unpacked, "data_yaml": _data_yaml_or_none(unpacked), "fetched": True}
