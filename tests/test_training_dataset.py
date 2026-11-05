# test_training_dataset.py
# Created: 2026-10-19
# Author: Isaac Travers
#
# Training dataset cache tests for the MARP Inference Worker.
# These build a real tar and unpack it, rather than mocking tarfile, because
# every interesting failure here is about what ends up on disk: whether the YAML
# names somewhere that exists, whether a second job re-fetches four gigabytes it
# already has, and whether a wrong archive leaves something behind that the next
# run would train on.
#
# No network and no GPU: the download is injected.

# hashlib computes the hash the fixtures are addressed by.
import hashlib

# tarfile builds the fixture archives.
import tarfile

# pytest provides raises and tmp_path.
import pytest

# Path types the fixture locations.
from pathlib import Path

# The module under test.
from marp_inference_worker.training import dataset as dataset_module
from marp_inference_worker.training.dataset import DatasetHashMismatch, ensure_dataset


# build_archive()
# Writes a small but structurally real dataset archive.
# Inputs: where to write it, and optional extra members.
# Output: (path to the tar, its sha256).
# Use this rather than a handcrafted byte string: the thing under test is the
# unpack, and a fixture that is not a tar proves nothing about it.
def build_archive(tmp_path: Path, name: str = "ds.tar", data_yaml: str | None = None) -> tuple[Path, str]:

    staging = tmp_path / f"staging-{name}"
    (staging / "train" / "images").mkdir(parents=True)
    (staging / "train" / "labels").mkdir(parents=True)
    (staging / "eval" / "images").mkdir(parents=True)
    (staging / "eval" / "labels").mkdir(parents=True)

    (staging / "train" / "images" / "a.jpg").write_bytes(b"not-really-a-jpeg")
    (staging / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (staging / "eval" / "images" / "b.jpg").write_bytes(b"not-really-a-jpeg")
    (staging / "eval" / "labels" / "b.txt").write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")

    # Relative, exactly as the packaging script writes it.
    (staging / "data.yaml").write_text(
        data_yaml if data_yaml is not None
        else "names:\n- Vermilion Rockfish\nnc: 1\ntrain: train/images\nval: eval/images\n",
        encoding="utf-8",
    )

    archive = tmp_path / name
    with tarfile.open(archive, "w") as tar:
        for member in ("train", "eval", "data.yaml"):
            tar.add(staging / member, arcname=member)

    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


# fake_download()
# Returns a download that copies a local archive, and counts its calls.
# Inputs: the archive to serve.
# Output: (callable, list the calls are appended to).
def fake_download(archive: Path):

    calls: list[str] = []

    def download(url, destination, headers=None, on_progress=None):
        calls.append(url)
        destination.write_bytes(Path(archive).read_bytes())
        if on_progress is not None:
            on_progress(destination.stat().st_size, destination.stat().st_size)

    return download, calls


# test_a_dataset_is_fetched_once_and_then_shared()
# Verifies a second job on the same machine reuses the first job's copy.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# This is R7, and it is the one with a cost attached: a per-attempt copy of
# rockfish5 is 4.9 GB, and #65 is the record of what per-attempt copies of
# shared bytes already did to a volunteer's disk.
def test_a_dataset_is_fetched_once_and_then_shared(tmp_path) -> None:

    archive, sha = build_archive(tmp_path)
    download, calls = fake_download(archive)
    cache = tmp_path / "cache"

    first = ensure_dataset("http://marp.invalid/d/1/archive", sha, cache, download=download)
    second = ensure_dataset("http://marp.invalid/d/1/archive", sha, cache, download=download)

    assert first["fetched"] is True
    assert second["fetched"] is False
    assert len(calls) == 1
    assert first["path"] == second["path"]


# test_the_yaml_points_at_where_the_dataset_actually_is()
# Verifies the archive's relative paths are made true on this machine.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# Ultralytics resolves a relative dataset path against its own settings
# directory rather than against the YAML, so shipping relative paths and leaving
# them alone points training at somewhere that does not exist (R12).
def test_the_yaml_points_at_where_the_dataset_actually_is(tmp_path) -> None:

    archive, sha = build_archive(tmp_path)
    download, _ = fake_download(archive)

    result = ensure_dataset("http://marp.invalid/d/1/archive", sha, tmp_path / "cache", download=download)
    text = Path(result["data_yaml"]).read_text(encoding="utf-8")

    train = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("train:"))
    val = next(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("val:"))

    assert Path(train).is_absolute() and Path(train).is_dir()
    assert Path(val).is_absolute() and Path(val).is_dir()
    assert (Path(train) / "a.jpg").is_file()

    # The class names survive the rewrite untouched.
    assert "Vermilion Rockfish" in text


# test_a_wrong_archive_is_refused_and_leaves_nothing_behind()
# Verifies a mismatched download cannot become a training run.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# And that the refusal is clean: a cache entry left behind would be picked up by
# the next attempt, which would train on it without ever re-checking the hash.
def test_a_wrong_archive_is_refused_and_leaves_nothing_behind(tmp_path) -> None:

    archive, _ = build_archive(tmp_path)
    download, _ = fake_download(archive)
    cache = tmp_path / "cache"
    claimed = "f" * 64

    with pytest.raises(DatasetHashMismatch):
        ensure_dataset("http://marp.invalid/d/1/archive", claimed, cache, download=download)

    assert not dataset_module.cache_entry(cache, claimed).exists()
    assert list(cache.glob("*.partial")) == []


# test_an_interrupted_unpack_is_not_mistaken_for_a_dataset()
# Verifies a directory without the completion marker is rebuilt.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# A worker killed part way through leaves a directory holding some of a dataset.
# It is the marker that says an entry is finished, never the directory's
# existence, because the second is true long before the first.
def test_an_interrupted_unpack_is_not_mistaken_for_a_dataset(tmp_path) -> None:

    archive, sha = build_archive(tmp_path)
    download, calls = fake_download(archive)
    cache = tmp_path / "cache"

    # Half a dataset, exactly as a killed worker would leave it.
    wreckage = dataset_module.cache_entry(cache, sha)
    (wreckage / "train" / "images").mkdir(parents=True)
    (wreckage / "train" / "images" / "stale.jpg").write_bytes(b"old")

    result = ensure_dataset("http://marp.invalid/d/1/archive", sha, cache, download=download)

    assert result["fetched"] is True
    assert len(calls) == 1
    assert not (Path(result["path"]) / "train" / "images" / "stale.jpg").exists()
    assert (Path(result["path"]) / "train" / "images" / "a.jpg").is_file()


# test_an_archive_without_a_data_yaml_is_refused()
# Verifies a dataset that cannot describe itself fails here, not in training.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
def test_an_archive_without_a_data_yaml_is_refused(tmp_path) -> None:

    staging = tmp_path / "bare"
    (staging / "train" / "images").mkdir(parents=True)
    (staging / "train" / "images" / "a.jpg").write_bytes(b"x")

    archive = tmp_path / "bare.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(staging / "train", arcname="train")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    download, _ = fake_download(archive)
    cache = tmp_path / "cache"

    with pytest.raises(FileNotFoundError):
        ensure_dataset("http://marp.invalid/d/1/archive", sha, cache, download=download)

    assert not dataset_module.cache_entry(cache, sha).exists()


# test_an_archive_cannot_write_outside_its_cache_entry()
# Verifies a traversal member is refused rather than unpacked.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# The archive is ours and arrives from our own API, which is exactly the
# reasoning that makes this worth a test: the unpack is where "ours" stops being
# a property anybody has checked.
def test_an_archive_cannot_write_outside_its_cache_entry(tmp_path) -> None:

    outside = tmp_path / "outside.txt"
    payload = tmp_path / "payload.txt"
    payload.write_text("escaped", encoding="utf-8")

    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname="../outside.txt")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    download, _ = fake_download(archive)

    with pytest.raises(Exception):
        ensure_dataset("http://marp.invalid/d/1/archive", sha, tmp_path / "cache", download=download)

    assert not outside.exists()


# The archive's own name for its YAML, so the test does not hard-code a
# second spelling of something the module already names.
_DATA_YAML_NAME = dataset_module._DATA_YAML


# a_classification_archive()
# A tar in the shape MARP's builder actually produces for classification.
# Inputs: tmp_path.
# Output: (archive path, sha256).
#
# Modelled on dataset 390: `train`/`val`/`test`, a directory per class named
# `<species_id>_<slug>`, and a `classes.json` beside the splits mapping those
# handles back to species. **No data.yaml**, because Ultralytics reads a
# classify dataset from the directory and takes its vocabulary from the folder
# names -- there is no YAML to carry.
def a_classification_archive(tmp_path):

    staging = tmp_path / "cls-staging"
    for split in ("train", "val", "test"):
        for handle in ("145_blue-deacon-rockfish", "8_lingcod"):
            folder = staging / split / handle
            folder.mkdir(parents=True)
            (folder / "crop0.jpg").write_bytes(b"x")
    (staging / "classes.json").write_text(
        '{"145_blue-deacon-rockfish": {"species_id": 145, "comname": "Blue/Deacon Rockfish"},'
        ' "8_lingcod": {"species_id": 8, "comname": "Lingcod"}}',
        encoding="utf-8",
    )

    archive = tmp_path / "cls.tar"
    with tarfile.open(archive, "w") as tar:
        for entry in sorted(staging.iterdir()):
            tar.add(entry, arcname=entry.name)

    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


# test_a_classification_archive_unpacks_without_a_data_yaml()
# Verifies the whole fetch-and-unpack path accepts a real classification shape.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# **This is the defect job 21798 died on.** `_rewrite_data_yaml` ran after every
# extraction and raised when there was no data.yaml, so every classification
# dataset was refused before any engine code was reached. The engine's own
# classify tests passed throughout, because they hand it a directory that was
# built by hand rather than one that came through this path.
def test_a_classification_archive_unpacks_without_a_data_yaml(tmp_path) -> None:

    archive, sha = a_classification_archive(tmp_path)
    download, _ = fake_download(archive)

    result = ensure_dataset(
        "http://marp.invalid/d/390/archive", sha, tmp_path / "cache", download=download
    )

    # No data.yaml, and the caller is told so rather than handed a path to a
    # file that is not there.
    assert result["data_yaml"] is None

    unpacked = Path(result["path"])
    assert (unpacked / "train" / "145_blue-deacon-rockfish" / "crop0.jpg").is_file()
    assert (unpacked / "val").is_dir() and (unpacked / "test").is_dir()

    # The vocabulary file travels with it; the worker does not read it, but
    # anything unpacking the archive outside MARP needs it.
    assert (unpacked / "classes.json").is_file()


# test_a_cached_classification_dataset_is_also_reported_without_a_yaml()
# Verifies the second fetch answers the same way as the first.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# The cached branch returns early and built its own answer, so it could have
# kept claiming a data.yaml that the fresh path had learned not to.
def test_a_cached_classification_dataset_is_also_reported_without_a_yaml(tmp_path) -> None:

    archive, sha = a_classification_archive(tmp_path)
    download, _ = fake_download(archive)
    cache = tmp_path / "cache"

    first = ensure_dataset("http://marp.invalid/d/390/archive", sha, cache, download=download)
    second = ensure_dataset("http://marp.invalid/d/390/archive", sha, cache, download=download)

    assert first["fetched"] is True and second["fetched"] is False
    assert second["data_yaml"] is None
    assert second["path"] == first["path"]


# test_a_detection_archive_still_gets_its_paths_rewritten()
# The control: admitting classification must not stop detection working.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
def test_a_detection_archive_still_gets_its_paths_rewritten(tmp_path) -> None:

    staging = tmp_path / "det-staging"
    (staging / "train" / "images").mkdir(parents=True)
    (staging / "train" / "images" / "a.jpg").write_bytes(b"x")
    (staging / "val" / "images").mkdir(parents=True)
    (staging / _DATA_YAML_NAME).write_text(
        "names:\n- Lingcod\nnc: 1\ntrain: train/images\nval: val/images\n", encoding="utf-8"
    )

    archive = tmp_path / "det.tar"
    with tarfile.open(archive, "w") as tar:
        for entry in sorted(staging.iterdir()):
            tar.add(entry, arcname=entry.name)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    download, _ = fake_download(archive)
    result = ensure_dataset(
        "http://marp.invalid/d/1/archive", sha, tmp_path / "cache", download=download
    )

    assert result["data_yaml"] is not None
    text = Path(result["data_yaml"]).read_text(encoding="utf-8")
    assert "train: " + Path(result["path"]).as_posix() + "/train/images" in text


# test_two_attempts_fetching_one_dataset_download_it_once()
# Verifies concurrent fetches of one archive are serialised (#73).
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# Job 21953 on ABYSS: three attempts of one job each unlinked and reopened the
# same `.partial`, and the later ones died with a Windows sharing violation.
# The download is slowed so the two certainly overlap.
def test_two_attempts_fetching_one_dataset_download_it_once(tmp_path: Path) -> None:

    import threading
    import time

    archive, sha = build_archive(tmp_path)
    copy, calls = fake_download(archive)

    def slow(url, destination, headers=None, on_progress=None):
        time.sleep(0.5)
        copy(url, destination, headers, on_progress)

    cache = tmp_path / "cache"
    results: list[dict] = []
    errors: list[BaseException] = []

    def fetch():
        try:
            results.append(ensure_dataset("http://marp.invalid/d/1/archive", sha, cache, download=slow))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=fetch) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert len(calls) == 1
    assert sorted(result["fetched"] for result in results) == [False, True]
    assert results[0]["path"] == results[1]["path"]

    # And the lock is gone once both have finished.
    assert list(cache.glob("*.lock")) == []


# test_a_lock_left_by_a_dead_process_is_taken_over()
# Verifies a crashed attempt's lock does not block the next one forever (#73).
# Inputs: tmp_path.
# Output: pytest pass/fail result.
def test_a_lock_left_by_a_dead_process_is_taken_over(tmp_path: Path) -> None:

    import psutil

    archive, sha = build_archive(tmp_path)
    download, calls = fake_download(archive)
    cache = tmp_path / "cache"
    cache.mkdir()

    # A pid nothing is running as.
    dead = 4_000_000
    while psutil.pid_exists(dead):
        dead += 1
    (cache / f"{sha}.lock").write_text(str(dead), encoding="utf-8")

    result = ensure_dataset("http://marp.invalid/d/1/archive", sha, cache, download=download)

    assert result["fetched"] is True
    assert len(calls) == 1


# _FakeStream
# Stands in for `httpx.stream`, serving an archive and honouring `Range` (#320).
# Records each request's headers so a test can see what was asked for.
class _FakeStream:

    def __init__(self, payload: bytes, fail_after: int | None = None):
        self.payload = payload
        self.fail_after = fail_after
        self.requests: list[dict] = []

    def __call__(self, method, url, headers=None, **kwargs):
        self.requests.append(dict(headers or {}))
        outer = self

        class _Response:
            def __init__(self):
                rng = (headers or {}).get("Range")
                first = int(rng.split("=")[1].split("-")[0]) if rng else 0
                if rng and first >= len(outer.payload):
                    self.status_code, self.body, self.headers = 416, b"", {}
                elif rng:
                    self.status_code = 206
                    self.body = outer.payload[first:]
                    self.headers = {"content-range": f"bytes {first}-{len(outer.payload) - 1}/{len(outer.payload)}"}
                else:
                    self.status_code, self.body = 200, outer.payload
                    self.headers = {"content-length": str(len(outer.payload))}

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(self.status_code)

            def iter_bytes(self, size):
                sent = 0
                for i in range(0, len(self.body), 64):
                    if outer.fail_after is not None and sent >= outer.fail_after:
                        outer.fail_after = None
                        raise ConnectionError("connection dropped")
                    chunk = self.body[i:i + 64]
                    sent += len(chunk)
                    yield chunk

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Response()


# test_an_interrupted_fetch_resumes_from_what_it_had()
# Verifies the second attempt asks for the rest, not the whole archive (#320).
# Inputs: tmp_path, monkeypatch.
# Output: pytest pass/fail result.
#
# Through the public address ABYSS fetches at 16.6 MB/s; a 20 GB dataset that
# restarted from zero after a dropped connection lost twenty minutes.
def test_an_interrupted_fetch_resumes_from_what_it_had(tmp_path: Path, monkeypatch) -> None:

    import httpx

    archive, sha = build_archive(tmp_path)
    payload = archive.read_bytes()
    stream = _FakeStream(payload, fail_after=1024)
    monkeypatch.setattr(httpx, "stream", stream)
    cache = tmp_path / "cache"

    with pytest.raises(ConnectionError):
        ensure_dataset("http://marp.invalid/d/1/archive", sha, cache)

    # The partial survived the failure, holding what arrived.
    partial = cache / f"{sha}.tar.partial"
    assert 0 < partial.stat().st_size < len(payload)
    held = partial.stat().st_size

    result = ensure_dataset("http://marp.invalid/d/1/archive", sha, cache)

    assert result["fetched"] is True
    assert stream.requests[-1]["Range"] == f"bytes={held}-"
    assert not partial.exists()
    assert (Path(result["path"]) / "train" / "images" / "a.jpg").is_file()


# test_a_whole_partial_is_used_rather_than_fetched_again()
# Verifies a partial the server has nothing beyond is checked and unpacked (#320).
# Inputs: tmp_path, monkeypatch.
# Output: pytest pass/fail result.
def test_a_whole_partial_is_used_rather_than_fetched_again(tmp_path: Path, monkeypatch) -> None:

    import httpx

    archive, sha = build_archive(tmp_path)
    stream = _FakeStream(archive.read_bytes())
    monkeypatch.setattr(httpx, "stream", stream)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{sha}.tar.partial").write_bytes(archive.read_bytes())

    result = ensure_dataset("http://marp.invalid/d/1/archive", sha, cache)

    assert result["fetched"] is True
    assert stream.requests == [{"Range": f"bytes={len(archive.read_bytes())}-"}]


# _cached()
# Puts a finished dataset in the cache, last used `age` seconds ago.
# Inputs: the cache, tmp_path, an archive name and an age in seconds.
# Output: the dataset's sha256.
def _cached(cache: Path, tmp_path: Path, name: str, age: float) -> str:

    import os
    import time

    archive, sha = build_archive(tmp_path, name=name, data_yaml=f"names:\n- {name}\nnc: 1\ntrain: train/images\nval: eval/images\n")
    download, _ = fake_download(archive)
    result = ensure_dataset("http://marp.invalid/d/archive", sha, cache, download=download)

    # Not in use any more: the test process is the one that marked it.
    for marker in Path(result["path"]).glob(".marp-in-use-*"):
        marker.unlink()
    when = time.time() - age
    os.utime(Path(result["path"]) / ".marp-complete", (when, when))
    return sha


# _disk_of()
# A disk whose free space is a fixed amount less whatever the cache holds.
def _disk_of(cache: Path, free_without_cache: int):

    def disk(_root):
        held = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
        return {"total_bytes": 10 ** 12, "free_bytes": free_without_cache - held}

    return disk


# test_room_is_made_by_evicting_the_least_recently_used()
# Verifies eviction order, and that it stops once the new dataset fits (#320).
# Inputs: tmp_path, monkeypatch.
# Output: pytest pass/fail result.
def test_room_is_made_by_evicting_the_least_recently_used(tmp_path: Path, monkeypatch) -> None:

    cache = tmp_path / "cache"
    oldest = _cached(cache, tmp_path, "oldest.tar", age=300)
    middle = _cached(cache, tmp_path, "middle.tar", age=200)
    newest = _cached(cache, tmp_path, "newest.tar", age=100)

    held = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
    one = held // 3

    # Room for the new dataset only once one old one has gone.
    floor = 1_000_000
    monkeypatch.setattr(dataset_module, "_disk", _disk_of(cache, floor + held + one // 2))

    evicted = dataset_module.make_room(cache, needed=one, keep_free_bytes=floor)

    assert evicted == [oldest]
    assert dataset_module.cache_entry(cache, middle).is_dir()
    assert dataset_module.cache_entry(cache, newest).is_dir()


# test_a_dataset_in_use_is_never_evicted()
# Verifies an entry a live process trains on is skipped, and the next one goes (#320).
# Inputs: tmp_path, monkeypatch.
# Output: pytest pass/fail result.
def test_a_dataset_in_use_is_never_evicted(tmp_path: Path, monkeypatch) -> None:

    import os

    cache = tmp_path / "cache"
    oldest = _cached(cache, tmp_path, "oldest.tar", age=300)
    middle = _cached(cache, tmp_path, "middle.tar", age=200)
    entry = dataset_module.cache_entry(cache, oldest)
    (entry / f".marp-in-use-{os.getpid()}").write_text("", encoding="utf-8")

    held = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
    one = held // 2
    floor = 1_000_000
    monkeypatch.setattr(dataset_module, "_disk", _disk_of(cache, floor + held + one // 2))

    assert dataset_module.make_room(cache, needed=one, keep_free_bytes=floor) == [middle]
    assert entry.is_dir()


# test_no_room_is_refused_with_the_numbers()
# Verifies a fetch that cannot fit above the floor is refused before downloading (#320).
# Inputs: tmp_path, monkeypatch.
# Output: pytest pass/fail result.
def test_no_room_is_refused_with_the_numbers(tmp_path: Path, monkeypatch) -> None:

    archive, sha = build_archive(tmp_path)
    download, calls = fake_download(archive)
    cache = tmp_path / "cache"
    monkeypatch.setattr(dataset_module, "_disk", lambda _root: {"total_bytes": 10 ** 12, "free_bytes": 60 * 10 ** 9})

    with pytest.raises(dataset_module.DatasetNoRoom, match="50 GB floor"):
        ensure_dataset("http://marp.invalid/d/1/archive", sha, cache, download=download,
                       size_bytes=20 * 10 ** 9)

    assert calls == []


# test_clearing_keeps_what_is_in_use()
# Verifies clear_cache removes idle datasets and leftover partials only (#320).
# Inputs: tmp_path.
# Output: pytest pass/fail result.
def test_clearing_keeps_what_is_in_use(tmp_path: Path) -> None:

    import os

    cache = tmp_path / "cache"
    idle = _cached(cache, tmp_path, "idle.tar", age=300)
    busy = _cached(cache, tmp_path, "busy.tar", age=200)
    (dataset_module.cache_entry(cache, busy) / f".marp-in-use-{os.getpid()}").write_text("", encoding="utf-8")
    (cache / f"{'0' * 64}.tar.partial").write_bytes(b"x" * 100)

    result = dataset_module.clear_cache(cache)

    assert result["evicted"] == [idle]
    assert result["kept_in_use"] == [busy]
    assert not dataset_module.cache_entry(cache, idle).exists()
    assert list(cache.glob("*.partial")) == []


# test_the_report_lists_entries_newest_first_with_sizes()
# Verifies what MARP is told about the cache (#320).
# Inputs: tmp_path.
# Output: pytest pass/fail result.
def test_the_report_lists_entries_newest_first_with_sizes(tmp_path: Path) -> None:

    cache = tmp_path / "cache"
    older = _cached(cache, tmp_path, "older.tar", age=300)
    newer = _cached(cache, tmp_path, "newer.tar", age=100)

    report = dataset_module.cache_report(cache, keep_free_bytes=7)

    assert [e["sha256"] for e in report["entries"]] == [newer, older]
    assert all(e["bytes"] and e["bytes"] > 0 for e in report["entries"])
    assert report["bytes"] == sum(e["bytes"] for e in report["entries"])
    assert report["keep_free_bytes"] == 7
    assert report["disk"]["total_bytes"] > 0


# test_the_test_split_path_is_made_absolute_too()
# Verifies `test:` is rewritten with `train:` and `val:` (#320).
# Inputs: tmp_path.
# Output: pytest pass/fail result.
def test_the_test_split_path_is_made_absolute_too(tmp_path: Path) -> None:

    archive, sha = build_archive(
        tmp_path, data_yaml="names:\n- A\nnc: 1\ntrain: train/images\nval: eval/images\ntest: test/images\n"
    )
    download, _ = fake_download(archive)

    result = ensure_dataset("http://marp.invalid/d/1/archive", sha, tmp_path / "cache", download=download)
    text = Path(result["data_yaml"]).read_text(encoding="utf-8")

    assert "test: " + Path(result["path"]).as_posix() + "/test/images" in text


# test_a_stop_during_the_fetch_stops_it_and_keeps_what_arrived()
# Verifies Pause now no longer waits for a dataset to finish downloading (MARP_API#319).
# Inputs: tmp_path, monkeypatch.
# Output: pytest pass/fail result.
#
# Found live on 2026-11-05: Pause now at 1,125 of 1,964 MB took 73 s, the fetch
# running on until the 60 s kill, because nothing checked the stop file before
# training began. The partial it left is what the next attempt resumes from.
def test_a_stop_during_the_fetch_stops_it_and_keeps_what_arrived(tmp_path: Path, monkeypatch) -> None:

    from marp_inference_worker.jobs import child_main

    archive, sha = build_archive(tmp_path)
    payload = archive.read_bytes()

    def slow(url, destination, headers=None, on_progress=None):
        with destination.open("wb") as handle:
            for i in range(0, len(payload), 64):
                handle.write(payload[i:i + 64])
                handle.flush()
                on_progress(i + 64, len(payload))

    monkeypatch.setattr(dataset_module, "_download", slow)

    class _Ctx:
        def __init__(self):
            self.calls = 0

        def report_progress(self, *args, **kwargs):
            self.calls += 1

        def should_stop(self):
            return self.calls > 3

        def log(self, *args, **kwargs):
            pass

    cache = tmp_path / "cache"
    with pytest.raises(child_main._FetchStopped):
        child_main._fetch_dataset_if_named(_Ctx(), {"params": {
            "dataset_url": "http://marp.invalid/d", "dataset_sha256": sha, "dataset_cache": str(cache),
        }})

    partial = cache / f"{sha}.tar.partial"
    assert 0 < partial.stat().st_size < len(payload), "what arrived is kept for the next attempt"
    assert not dataset_module.cache_entry(cache, sha).exists()
    assert not (cache / f"{sha}.lock").exists(), "and the lock is let go"
