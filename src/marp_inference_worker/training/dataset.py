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

# json records each cache entry's size beside it, so reporting it is not a walk (#320).
import json

# os creates the download lock atomically (#73).
import os

# shutil removes a half-unpacked directory when something goes wrong.
import shutil

# tarfile unpacks the archive. The packaging side writes an uncompressed tar
# deliberately -- a dataset is almost all JPEG, so compression costs minutes of
# CPU and saves nothing -- and tarfile is in the standard library, so a worker
# needs no dependency to read one.
import tarfile

# time paces the wait for another attempt's download (#73).
import time

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


# Beside the marker: what a finished entry holds, written once when it is unpacked.
# Reporting the cache reads this rather than walking 310,520 files per poll (#320).
_ENTRY_INFO = ".marp-entry.json"


# Prefix of the file a process leaves in an entry while it trains on it (#320).
# Eviction skips an entry carrying one whose process is still alive.
_IN_USE_PREFIX = ".marp-in-use-"


# The disk a worker keeps free when MARP names no floor of its own (#320).
# Settled with Isaac on 2026-10-31: 50 GB, adjustable per worker from MARP.
DEFAULT_KEEP_FREE_BYTES = 50 * 1000 ** 3


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


# DatasetNoRoom
# Raised when a dataset cannot be fetched without going under the free-disk floor,
# even after evicting every cached dataset not in use (#320).
class DatasetNoRoom(RuntimeError):
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

    # **Resumed from what is already on disk** (#320). At 16.6 MB/s through the
    # public address, a 20 GB fetch that restarts from zero loses twenty minutes.
    start = destination.stat().st_size if destination.exists() else 0
    request_headers = dict(headers or {})
    if start:
        request_headers["Range"] = f"bytes={start}-"

    with httpx.stream("GET", url, headers=request_headers, timeout=timeout, follow_redirects=True) as response:

        # Nothing past what we hold: the partial is already whole, or is not this
        # archive at all. The hash check after this decides which.
        if start and response.status_code == 416:
            return

        response.raise_for_status()

        # 206 continues the partial; a 200 is the whole archive, so start over.
        resuming = start and response.status_code == 206
        received = start if resuming else 0
        if resuming:
            total = int(str(response.headers.get("content-range", "")).rsplit("/", 1)[-1] or 0) or None
        else:
            total = int(response.headers.get("content-length") or 0) or None

        with destination.open("ab" if resuming else "wb") as handle:
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
        # `test` too: left relative, Ultralytics resolves it against its own settings.
        if key in {"train", "val", "test"}:
            relative = line.split(":", 1)[1].strip()
            # Already absolute means somebody else has been here; leave it.
            candidate = Path(relative)
            absolute = candidate if candidate.is_absolute() else (unpacked / relative)
            rewritten.append(f"{key}: {absolute.as_posix()}")
        else:
            rewritten.append(line)

    yaml_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return yaml_path


# How long a second caller waits for another attempt's download, in seconds.
# A 20 GB archive took 28 minutes to reach training on ABYSS through the public
# address (job 21954); this is generous.
_LOCK_WAIT_S = 2 * 60 * 60


class DatasetBusy(RuntimeError):
    pass


# _dataset_lock()
# Holds one archive's download for this caller, beside its cache entry (#73).
# Inputs: the cache root and the archive's sha256.
# Output: a context manager.
# A lock left by a process that has died is taken over; one held by a live
# process is waited for, and past _LOCK_WAIT_S the wait fails naming it.
class _dataset_lock:

    def __init__(self, cache_root, sha256):
        self._path = Path(cache_root) / f"{sha256}.lock"

    def __enter__(self):
        import psutil

        started = time.monotonic()
        while True:
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    owner = int(self._path.read_text(encoding="utf-8").strip() or 0)
                except (OSError, ValueError):
                    owner = 0

                # Its holder is gone: what it left is not a download in progress.
                if owner and not psutil.pid_exists(owner):
                    self._path.unlink(missing_ok=True)
                    continue

                if time.monotonic() - started > _LOCK_WAIT_S:
                    raise DatasetBusy(
                        f"another attempt (process {owner}) has been downloading this dataset "
                        f"for over {_LOCK_WAIT_S // 60} minutes"
                    )
                time.sleep(1.0)
                continue

            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            return self

    def __exit__(self, *exc):
        self._path.unlink(missing_ok=True)
        return False


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
    size_bytes: int | None = None,
    keep_free_bytes: int | None = None,
) -> dict[str, Any]:

    expected = str(sha256).lower()
    if not expected:
        raise ValueError("a training dataset must carry the sha256 it is expected to have")

    cache_root = Path(cache_root)
    unpacked = cache_entry(cache_root, expected)
    marker = unpacked / _COMPLETE_MARKER

    # Already here and finished. The marker is the test, not the directory.
    if marker.is_file():
        return _use(unpacked, fetched=False)

    cache_root.mkdir(parents=True, exist_ok=True)

    # **One download per archive** (#73). Two attempts of one job on one machine
    # each unlinked and reopened the same `.partial`, and the second died with a
    # Windows sharing violation that said nothing about the first. The lock waits
    # for the other, then uses what it finished.
    with _dataset_lock(cache_root, expected):
        if marker.is_file():
            return _use(unpacked, fetched=False)

        # A directory without the marker is the wreckage of an interrupted attempt,
        # and whatever is in it cannot be trusted. Start again rather than resume
        # into it: the archive is content-addressed, so starting again is cheap to
        # reason about and a half-unpacked tree is not. Only under the lock, so it
        # is never another attempt's unpacking in progress.
        if unpacked.exists():
            shutil.rmtree(unpacked, ignore_errors=True)

        # **Room first** (#320): the rest of the archive plus its unpacked copy,
        # above the floor. Evicting after the fact would mean a full disk mid-fetch.
        if size_bytes:
            partial = cache_root / f"{expected}.tar.partial"
            held = partial.stat().st_size if partial.exists() else 0
            needed = max(0, int(size_bytes) - held) + int(size_bytes)
        else:
            needed = 0
        floor = DEFAULT_KEEP_FREE_BYTES if keep_free_bytes is None else int(keep_free_bytes)
        make_room(cache_root, needed, floor, keep={expected})

        _fetch_and_unpack(url, expected, cache_root, unpacked, marker, headers, on_progress, download)
        return _use(unpacked, fetched=True)


# _fetch_and_unpack()
# Downloads, verifies and unpacks one archive into its cache entry.
# Inputs: as ensure_dataset(), resolved.
# Output: the same mapping ensure_dataset() returns.
# Use this only while holding the archive's lock.
def _fetch_and_unpack(url, expected, cache_root, unpacked, marker, headers, on_progress, download):

    # Kept between attempts so the next one resumes it (#320). Only a hash
    # mismatch or a finished unpack removes it.
    archive = cache_root / f"{expected}.tar.partial"

    fetch = download or _download
    try:
        fetch(url, archive, headers=headers, on_progress=on_progress)

        # Verified before anything is unpacked. A truncated download that became
        # a training run would produce a model trained on part of a dataset and
        # nothing anywhere would say so.
        actual = compute_sha256(archive)
        if actual != expected:
            # Not these bytes: resuming into it would only repeat the mismatch.
            archive.unlink(missing_ok=True)
            raise DatasetHashMismatch(
                f"dataset archive has sha256 {actual}, expected {expected}"
            )

        unpacked.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:") as tar:
            members = tar.getmembers()
            # `filter="data"` refuses absolute paths, parent traversal, links and
            # device nodes. The archive is ours, but it arrives over the network
            # and unpacking is the moment where that stops mattering.
            tar.extractall(unpacked, filter="data")

        _rewrite_data_yaml(unpacked)

        # What the entry holds, so the cache can be reported without a walk.
        _write_entry_info(unpacked, sum(m.size for m in members if m.isfile()),
                          sum(1 for m in members if m.isfile()))

        # Last, so a crash anywhere above leaves an entry that is visibly unfinished.
        marker.write_text(expected + "\n", encoding="utf-8")

    except Exception:
        # Leave nothing a later run could mistake for a usable dataset.
        shutil.rmtree(unpacked, ignore_errors=True)
        raise

    # Unpacked and marked: the archive has done its job.
    archive.unlink(missing_ok=True)


# _use()
# Records that this process is training on a cache entry, and returns it.
# Inputs: the entry directory, and whether this call fetched it.
# Output: the mapping ensure_dataset() returns.
# The marker's time is the entry's last use, which is what eviction orders by;
# the in-use file keeps it from being evicted while this process lives (#320).
def _use(unpacked: Path, fetched: bool) -> dict[str, Any]:

    marker = unpacked / _COMPLETE_MARKER
    os.utime(marker)
    (unpacked / f"{_IN_USE_PREFIX}{os.getpid()}").write_text("", encoding="utf-8")
    return {"path": unpacked, "data_yaml": _data_yaml_or_none(unpacked), "fetched": fetched}


# _write_entry_info()
# Records a finished entry's size beside it (#320).
# Inputs: the entry directory, its bytes and its file count.
# Output: none.
def _write_entry_info(unpacked: Path, size: int, files: int) -> None:

    (unpacked / _ENTRY_INFO).write_text(
        json.dumps({"bytes": int(size), "files": int(files)}), encoding="utf-8"
    )


# entry_bytes()
# How many bytes a finished cache entry holds.
# Inputs: the entry directory; whether to walk it when nothing is recorded.
# Output: bytes, or None when unrecorded and walk is False.
# Entries unpacked before #320 carry no record; one walk writes it, and the
# runner does that in the background rather than in a poll.
def entry_bytes(unpacked: Path, walk: bool = False) -> int | None:

    try:
        return int(json.loads((unpacked / _ENTRY_INFO).read_text(encoding="utf-8"))["bytes"])
    except (OSError, ValueError, KeyError, TypeError):
        pass

    if not walk:
        return None

    size = 0
    files = 0
    for path in unpacked.rglob("*"):
        if path.is_file() and not path.name.startswith(".marp-"):
            size += path.stat().st_size
            files += 1
    _write_entry_info(unpacked, size, files)
    return size


# _in_use()
# Whether a live process is training on this entry (#320).
# Inputs: the entry directory.
# Output: True when any in-use file names a process that is still running.
# Files left by processes that have exited are removed on the way.
def _in_use(unpacked: Path) -> bool:

    import psutil

    busy = False
    for path in unpacked.glob(f"{_IN_USE_PREFIX}*"):
        try:
            pid = int(path.name[len(_IN_USE_PREFIX):])
        except ValueError:
            pid = 0
        if pid and psutil.pid_exists(pid):
            busy = True
        else:
            path.unlink(missing_ok=True)
    return busy


# _disk()
# Free and total bytes on the disk the cache is on.
# Inputs: the cache root, which may not exist yet.
# Output: {"total_bytes", "free_bytes"}.
def _disk(cache_root: Path) -> dict[str, int]:

    probe = Path(cache_root)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {"total_bytes": int(usage.total), "free_bytes": int(usage.free)}


# _entries()
# The cache's finished entries, least recently used first.
# Inputs: the cache root.
# Output: list of entry directories.
def _entries(cache_root: Path) -> list[Path]:

    root = Path(cache_root)
    if not root.is_dir():
        return []
    found = [p for p in root.iterdir() if p.is_dir() and (p / _COMPLETE_MARKER).is_file()]
    return sorted(found, key=lambda p: (p / _COMPLETE_MARKER).stat().st_mtime)


# _evict()
# Removes one cache entry, returning the bytes it held.
def _evict(unpacked: Path) -> int:

    size = entry_bytes(unpacked) or 0
    # The marker first, so a removal that fails part way leaves an unfinished
    # entry the next fetch clears, never a finished one with files missing.
    (unpacked / _COMPLETE_MARKER).unlink(missing_ok=True)
    shutil.rmtree(unpacked, ignore_errors=True)
    return size


# make_room()
# Evicts least-recently-used datasets until `needed` bytes fit above the floor (#320).
# Inputs: the cache root, the bytes about to be written, the free-disk floor, and
#   hashes never to evict.
# Output: the hashes evicted.
# Raises DatasetNoRoom when even an empty cache would not make it fit; an entry
# a live process is training on is never evicted.
def make_room(cache_root: Path, needed: int, keep_free_bytes: int, keep: set[str] | None = None) -> list[str]:

    keep = keep or set()
    evicted: list[str] = []
    for unpacked in _entries(cache_root):
        if _disk(cache_root)["free_bytes"] - needed >= keep_free_bytes:
            break
        if unpacked.name in keep or _in_use(unpacked):
            continue
        _evict(unpacked)
        evicted.append(unpacked.name)

    # With no size to go on (a spec from before #320), evicting is all it can do.
    free = _disk(cache_root)["free_bytes"]
    if needed and free - needed < keep_free_bytes:
        raise DatasetNoRoom(
            f"this dataset needs {needed / 1e9:.1f} GB, and fetching it would leave "
            f"{max(0, free - needed) / 1e9:.1f} GB free on the worker's disk, under its "
            f"{keep_free_bytes / 1e9:.0f} GB floor, with every unused dataset already evicted"
        )
    return evicted


# clear_cache()
# Evicts every cached dataset not in use, when MARP asks (#320).
# Inputs: the cache root.
# Output: {"evicted": [...], "freed_bytes": n, "kept_in_use": [...]}.
# Leftover partial downloads are removed too, unless their download is running.
def clear_cache(cache_root: Path) -> dict[str, Any]:

    import psutil

    root = Path(cache_root)
    evicted: list[str] = []
    kept: list[str] = []
    freed = 0
    for unpacked in _entries(root):
        if _in_use(unpacked):
            kept.append(unpacked.name)
            continue
        freed += _evict(unpacked)
        evicted.append(unpacked.name)

    if root.is_dir():
        for partial in root.glob("*.tar.partial"):
            lock = root / partial.name.replace(".tar.partial", ".lock")
            try:
                owner = int(lock.read_text(encoding="utf-8").strip() or 0) if lock.exists() else 0
            except (OSError, ValueError):
                owner = 0
            if owner and psutil.pid_exists(owner):
                continue
            freed += partial.stat().st_size
            partial.unlink(missing_ok=True)

    return {"evicted": evicted, "freed_bytes": freed, "kept_in_use": kept}


# cache_report()
# What the dataset cache holds and the disk it is on, for MARP (#320).
# Inputs: the cache root, and the floor this worker keeps.
# Output: JSON-safe mapping.
# Cheap enough for every poll: it reads each entry's record and never walks one.
def cache_report(cache_root: Path, keep_free_bytes: int | None = None) -> dict[str, Any]:

    from datetime import datetime, timezone

    entries = []
    for unpacked in reversed(_entries(cache_root)):
        used = (unpacked / _COMPLETE_MARKER).stat().st_mtime
        entries.append({
            "sha256": unpacked.name,
            "bytes": entry_bytes(unpacked),
            "last_used_at": datetime.fromtimestamp(used, timezone.utc).isoformat(),
            "in_use": _in_use(unpacked),
        })

    root = Path(cache_root)
    partial = sum(p.stat().st_size for p in root.glob("*.tar.partial")) if root.is_dir() else 0
    return {
        "bytes": sum(e["bytes"] or 0 for e in entries) + partial,
        "partial_bytes": partial,
        "entries": entries,
        "disk": _disk(root),
        "keep_free_bytes": DEFAULT_KEEP_FREE_BYTES if keep_free_bytes is None else int(keep_free_bytes),
    }
