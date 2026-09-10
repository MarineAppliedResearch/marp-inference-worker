# test_job_context.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Tests for the job context in the MARP Inference Worker.
# The context is the whole boundary between an engine and MARP (R7). These
# verify the six calls it offers, that each produces an event the parent can
# read, and that cancellation reaches an engine through should_stop().
#
# What these cannot see: a job actually being cancelled mid-inference. That is
# a runner-level behaviour and is covered in test_job_runner.py, which runs a
# real child process. A unit test on this class can only prove the mechanism,
# not that it is wired up.

# json parses the event lines the context writes.
import json

# StringIO stands in for the child's stdout.
from io import StringIO

# Path types the workspace paths.
from pathlib import Path

# Pytest checks the expected refusal of a missing artifact.
import pytest

from marp_inference_worker.jobs.context import ChildJobContext, hash_file


# _make_context()
# Builds a context writing into a string buffer.
# Inputs: the pytest temporary directory and optional params.
# Output: the context and the buffer its events land in.
# Use this so a test can read exactly what an engine's calls produced.
def _make_context(tmp_path: Path, params: dict | None = None):

    stream = StringIO()
    ctx = ChildJobContext(
        params=params or {},
        checkpoint_dir=tmp_path / "work",
        stop_file=tmp_path / "STOP",
        stream=stream,
    )
    return ctx, stream


# _events(stream)
# Parses the events written to a buffer.
# Inputs: the stream.
# Output: list of decoded event mappings.
def _events(stream: StringIO) -> list[dict]:

    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# test_context_offers_exactly_the_six_documented_calls()
# Verifies the context's public surface.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R7's "exactly" clause. An engine is meant to have these and no more; a
# seventh public call would be a way for engine code to learn something about
# MARP, which is the thing the contract exists to prevent. emit_terminal is
# excluded deliberately -- it belongs to the child entry point, not the engine.
def test_context_offers_exactly_the_six_documented_calls() -> None:

    public = {
        name
        for name in dir(ChildJobContext)
        if not name.startswith("_") and name != "emit_terminal"
    }

    assert public == {
        "params",
        "log",
        "report_progress",
        "report_metrics",
        "checkpoint_dir",
        "resume_from",
        "publish_artifact",
        "should_stop",
    }


# test_context_satisfies_the_job_context_protocol()
# Verifies the context is what engines are typed against.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves that the concrete implementation and the protocol engines depend on
# have not drifted apart -- which they can, since a Protocol is structural.
def test_context_satisfies_the_job_context_protocol(tmp_path: Path) -> None:

    from marp_inference_worker.engines.base_engine import JobContext

    ctx, _ = _make_context(tmp_path)
    assert isinstance(ctx, JobContext)


# test_params_are_the_jobs_own_settings(tmp_path)
# Verifies params reach the engine unchanged.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves the first of the six: an engine reads its settings from the job, not
# from worker configuration.
def test_params_are_the_jobs_own_settings(tmp_path: Path) -> None:

    ctx, _ = _make_context(tmp_path, params={"confidence": 0.15, "data_type": "Fish"})

    assert ctx.params["confidence"] == 0.15
    assert ctx.params["data_type"] == "Fish"


# test_checkpoint_dir_exists_before_the_engine_writes(tmp_path)
# Verifies the workspace is created up front.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves an engine can write to checkpoint_dir immediately, without having to
# create it -- which is what every engine would otherwise duplicate.
def test_checkpoint_dir_exists_before_the_engine_writes(tmp_path: Path) -> None:

    ctx, _ = _make_context(tmp_path)

    assert ctx.checkpoint_dir.is_dir()

    # And Milestone 1 hands no checkpoint to resume from.
    assert ctx.resume_from is None


# test_log_progress_and_metrics_each_produce_a_readable_event(tmp_path)
# Verifies the three reporting calls.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves the events the parent folds into heartbeats and event batches exist and
# carry what the coordinator needs.
def test_log_progress_and_metrics_each_produce_a_readable_event(tmp_path: Path) -> None:

    ctx, stream = _make_context(tmp_path)

    ctx.log("opened video", level="info")
    ctx.report_progress(120, 1800, "frames")
    ctx.report_metrics(step=120, phase="inference", metrics={"active_tracks": 3})

    events = _events(stream)
    assert [event["kind"] for event in events] == ["log", "progress", "metrics"]

    # Each event carries its own payload.
    assert events[0]["message"] == "opened video"
    assert events[1]["done"] == 120 and events[1]["total"] == 1800
    assert events[2]["metrics"] == {"active_tracks": 3}

    # Sequence numbers are assigned in order, so the coordinator can drop
    # duplicates when a batch is resent after a network failure.
    assert [event["seq"] for event in events] == [0, 1, 2]


# test_metrics_mapping_is_not_schemad(tmp_path)
# Verifies that arbitrary metric names pass through.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves a deliberate decision rather than an accident. Ultralytics' metric
# names differ by task and change between versions, so a fixed set of fields
# here would be wrong within a release. Anything JSON-safe must survive.
def test_metrics_mapping_is_not_schemad(tmp_path: Path) -> None:

    ctx, stream = _make_context(tmp_path)

    ctx.report_metrics(
        step=1,
        phase="train",
        metrics={
            "metrics/mAP50-95(B)": 0.42,
            "train/box_loss": 1.03,
            "lr/pg0": 0.0001,
            "anything_at_all": [1, 2, 3],
        },
    )

    metrics = _events(stream)[0]["metrics"]
    assert metrics["metrics/mAP50-95(B)"] == 0.42
    assert metrics["anything_at_all"] == [1, 2, 3]


# test_publish_artifact_hashes_the_real_bytes(tmp_path)
# Verifies the artifact hand-off.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R11's mechanism: the results file is hashed by the process that wrote
# it, so the hash describes exactly the bytes on disk, and the parent is told
# where it is rather than being handed its contents.
def test_publish_artifact_hashes_the_real_bytes(tmp_path: Path) -> None:

    ctx, stream = _make_context(tmp_path)

    results_path = ctx.checkpoint_dir / "observations.jsonl"
    results_path.write_text('{"frame": 1}\n', encoding="utf-8")

    returned_sha256 = ctx.publish_artifact(results_path, "observations")

    # The returned hash is the file's real hash.
    assert returned_sha256 == hash_file(results_path)

    # The event names the role, the path, the hash and the size -- and no
    # detections travel in it.
    event = _events(stream)[0]
    assert event["kind"] == "artifact"
    assert event["role"] == "observations"
    assert event["sha256"] == returned_sha256
    assert event["size_bytes"] == results_path.stat().st_size
    assert "detections" not in event


# test_publish_artifact_refuses_a_missing_file(tmp_path)
# Verifies that a nonexistent artifact is not reported as a result.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves an engine bug is caught here rather than becoming a successful job with
# no result behind it.
def test_publish_artifact_refuses_a_missing_file(tmp_path: Path) -> None:

    ctx, _ = _make_context(tmp_path)

    with pytest.raises(FileNotFoundError):
        ctx.publish_artifact(ctx.checkpoint_dir / "not-there.jsonl", "observations")


# test_should_stop_is_false_until_the_stop_file_appears(tmp_path)
# Verifies the cancellation mechanism.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves the engine-side half of R5. The parent creates the stop file when a
# heartbeat says cancel or abandon, and the engine notices on its next check.
#
# What this does NOT prove is that a running job is actually cancelled: this is
# a unit test and cannot observe a child process unwinding mid-inference. See
# test_job_runner.py for the tier that can.
def test_should_stop_is_false_until_the_stop_file_appears(tmp_path: Path) -> None:

    ctx, _ = _make_context(tmp_path)
    stop_file = tmp_path / "STOP"

    # Nothing has asked it to stop.
    assert ctx.should_stop() is False

    # The parent asks.
    stop_file.touch()

    # The answer is cached for a fraction of a second to keep a per-frame call
    # cheap, so the cache has to be stepped past for the test to be honest
    # about what it is checking.
    ctx._stop_checked_at = 0.0
    assert ctx.should_stop() is True


# test_should_stop_stays_true_once_set(tmp_path)
# Verifies that a stop is not retractable.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves an engine part way through unwinding cannot be told to carry on by the
# stop file being removed, which would leave it in a half-finished state.
def test_should_stop_stays_true_once_set(tmp_path: Path) -> None:

    ctx, _ = _make_context(tmp_path)
    stop_file = tmp_path / "STOP"

    stop_file.touch()
    ctx._stop_checked_at = 0.0
    assert ctx.should_stop() is True

    # The file goes away; the answer does not change.
    stop_file.unlink()
    assert ctx.should_stop() is True
