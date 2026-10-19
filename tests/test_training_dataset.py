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
