# test_job_runner.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Tests for the job runner in the MARP Inference Worker.
# These run the real loop against a fake coordinator: a real child process, real
# event lines over a real pipe, real heartbeats, real cancellation, real terminal
# reports. The engine is the mock one, so no GPU and no model file are needed.
#
# This is the tier that can observe the runner. The doctrine matters here: a unit
# test on the context class can prove that should_stop() reads a file, but only
# this tier can prove that a heartbeat answering `cancel` actually stops a job
# that is running -- the mechanism and the wiring are different things, and this
# repository has had defects reported twice because the first fix was verified at
# a tier that could not see the bug.
#
# What is still NOT covered here, and cannot be: real YOLO inference, real
# ByteTrack over real frames, GPU device selection, and a real Jellyfin stream.
# Those need a GPU and are listed as uncovered in .marp/verification.md.

# time waits for the child, with a deadline rather than a fixed sleep.
import time

# Path types the state directory.
from pathlib import Path

# Any types the fake coordinator's payloads.
from typing import Any

from marp_inference_worker.jobs.runner import JobRunner
from marp_inference_worker.jobs.worker_state import WorkerState


# How long a test will wait for a child process to reach a state.
# Generous, because launching a Python interpreter on Windows is not fast.
_DEADLINE_S = 60.0


# FakeCoordinator
# Stands in for MARP's coordinator routes.
# Records every call so a test can assert what the worker actually sent, and
# lets a test script the heartbeat answers -- which is how cancel, pause and
# abandon reach a worker (R5).
class FakeCoordinator:

    # __init__()
    # Builds a coordinator that offers a fixed list of jobs.
    # Inputs: the attempt envelopes to offer, in order.
    # Output: initialized FakeCoordinator.
    def __init__(self, offers: list[dict[str, Any]] | None = None) -> None:

        # Offered one per poll, then nothing.
        self._offers = list(offers or [])

        # Everything the worker sent, for assertions.
        self.enrolments: list[dict[str, Any]] = []
        self.polls: list[dict[str, Any]] = []
        self.heartbeats: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.artifact_checks: list[dict[str, Any]] = []
        self.uploads: list[Any] = []
        self.results: list[dict[str, Any]] = []

        # What each heartbeat should answer. A callable receives the heartbeat
        # count and returns an action, so a test can cancel on the third one.
        self.heartbeat_action = "continue"

        # Whether check_artifact claims to already hold the artifact.
        self.already_have_artifact = True

    # enrol()
    # Records the enrolment and assigns a worker id.
    def enrol(
        self,
        local_id: str,
        capabilities: dict[str, Any],
        name: str,
        slot_count: int,
        worker_version: str | None = None,
    ) -> dict[str, Any]:

        # Signature mirrors CoordinatorClient.enrol deliberately. It used to
        # take only (local_id, capabilities) and every test passed, while the
        # real coordinator answered 400 because `name` is required -- a fake
        # that has drifted from the contract tests nothing about it. Keyword
        # names matter here: the runner calls them by keyword.
        self.enrolments.append({
            "local_id": local_id,
            "capabilities": capabilities,
            "name": name,
            "slot_count": slot_count,
            "worker_version": worker_version,
        })
        return {"worker_id": f"worker-for-{local_id[:8]}"}

    # poll()
    # Hands out the next offer, or None once they are exhausted.
    def poll(self, worker_id: str, free_slots: int, engines: list[str]):

        self.polls.append({"worker_id": worker_id, "free_slots": free_slots, "engines": engines})
        if self._offers:
            return self._offers.pop(0)
        return None

    # heartbeat()
    # Records the heartbeat and returns the scripted action.
    def heartbeat(self, attempt_id, worker_id, lease_epoch, progress) -> dict[str, Any]:

        self.heartbeats.append(
            {
                "attempt_id": attempt_id,
                "worker_id": worker_id,
                "lease_epoch": lease_epoch,
                "progress": progress,
            }
        )

        # Callable actions let a test change its answer over time.
        action = self.heartbeat_action
        if callable(action):
            action = action(len(self.heartbeats))
        return {"action": action}

    # post_events()
    # Records a batch of log and metric events.
    def post_events(self, attempt_id, worker_id, lease_epoch, events) -> None:

        for event in events:
            self.events.append({"attempt_id": attempt_id, **event})

    # check_artifact()
    # Answers whether the artifact is already held.
    def check_artifact(self, attempt_id, worker_id, lease_epoch, sha256, size_bytes, kind):

        self.artifact_checks.append(
            {"attempt_id": attempt_id, "sha256": sha256, "size_bytes": size_bytes, "kind": kind}
        )
        if self.already_have_artifact:
            return {"already_have": True}
        return {"already_have": False, "upload": {"url": "http://fake/upload", "method": "PUT"}}

    # upload_artifact()
    # Records that an upload happened.
    def upload_artifact(self, upload_target, path) -> None:

        self.uploads.append({"target": upload_target, "path": str(path), "bytes": path.stat().st_size})

    # report_result()
    # Records the terminal report.
    def report_result(self, attempt_id, worker_id, lease_epoch, outcome, payload) -> None:

        self.results.append(
            {
                "attempt_id": attempt_id,
                "worker_id": worker_id,
                "lease_epoch": lease_epoch,
                "outcome": outcome,
                "result": payload,
            }
        )

    # close()
    # Present so the runner can shut the client down.
    def close(self) -> None:
        pass


# _fake_model_file()
# Writes a stand-in model artifact and returns its path and real sha256.
# Inputs: the directory to write into and unique contents.
# Output: the path and its lower-case hex sha256.
#
# A real file with a real hash, not a placeholder. The runner fetches and
# verifies every job's model before launching anything (R13), so a job spec
# carrying a hash that does not match its bytes is correctly refused -- which
# means a test that wants a job to run has to supply a matching pair.
def _fake_model_file(
    directory: Path,
    contents: bytes = b"stand-in model bytes",
    name: str = "test-model.pt",
):

    import hashlib

    # A distinct filename per job, deliberately. An earlier version of this
    # helper reused one path, so building three offers up front left the file
    # holding the third job's bytes -- and the runner then refused the first two
    # on a hash mismatch, correctly. The check was right and the fixture was
    # wrong, which is worth remembering before relaxing anything here.
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / name
    model_path.write_bytes(contents)
    return model_path, hashlib.sha256(contents).hexdigest()


# _mock_job()
# Builds an attempt envelope for the mock engine.
# Inputs: attempt id, the model file and its hash, the frame count, and an
# optional per-frame delay.
# Output: the offer mapping a coordinator would return from /poll.
# Use a delay when the test needs the job still running so it can be cancelled.
def _mock_job(
    attempt_id: str,
    model_path: Path,
    model_sha256: str,
    frames: int = 5,
    frame_delay_s: float = 0.0,
    start_frame: int = 1000,
) -> dict[str, Any]:

    return {
        "attempt_id": attempt_id,
        "worker_id": "worker-for-test",
        "lease_epoch": 3,
        "spec": {
            "engine": "mock",
            # A local path rather than an http url: the fetch path is a copy,
            # so the test needs no network, and the hash is still verified.
            "model": {
                "name": f"test-model-{attempt_id}",
                "sha256": model_sha256,
                "url": str(model_path),
            },
            "video": {"jellyfin_item_id": "item-1", "source_name": "20240727_185645 Fwd.mp4"},
            # A range is always present, even for a whole video (R9), and it is
            # half-open: this job owns start_frame up to but NOT end_frame.
            "range": {"start_frame": start_frame, "end_frame": start_frame + frames},
            "params": {
                "frame_delay_s": frame_delay_s,
                # Supplied so the runner needs no Jellyfin server. Resolution is
                # exercised separately and is not what these tests are about.
                "video_source_url": "not-used-by-the-mock-engine",
            },
            "reduction": {"name": "v3_dirpad", "version": "1"},
        },
    }


# _job_for()
# Builds a mock job with a fresh model file under the test's own directory.
# Inputs: the pytest temporary directory, attempt id, frames and delay.
# Output: the offer mapping.
# Use this in tests that do not care about the model, which is most of them.
def _job_for(
    tmp_path: Path,
    attempt_id: str,
    frames: int = 5,
    frame_delay_s: float = 0.0,
    start_frame: int = 1000,
) -> dict[str, Any]:

    # Contents keyed by attempt id, so each job gets its own cache entry and
    # tests cannot pass by reusing a previous run's cached artifact.
    model_path, model_sha256 = _fake_model_file(
        tmp_path / "models",
        contents=f"model for {attempt_id}".encode(),
        name=f"{attempt_id}.pt",
    )
    return _mock_job(
        attempt_id, model_path, model_sha256, frames, frame_delay_s, start_frame
    )


# _runner()
# Builds a runner against a fake coordinator.
# Inputs: the fake coordinator, the temporary state directory, and slot count.
# Output: the runner and its worker state.
def _runner(coordinator: FakeCoordinator, tmp_path: Path, slots: int = 1):

    state = WorkerState()
    runner = JobRunner(
        client=coordinator,
        state_dir=tmp_path / "state",
        worker_state=state,
        slot_count=slots,
    )
    return runner, state


# _wait_until()
# Waits for a condition, servicing the runner while it waits.
# Inputs: the runner, the predicate, and a deadline.
# Output: True when the condition became true.
# Use this rather than a fixed sleep: a fixed sleep either flakes on a slow
# machine or wastes time on a fast one.
def _wait_until(runner: JobRunner, predicate, deadline_s: float = _DEADLINE_S) -> bool:

    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        runner._service_running_jobs()
        if predicate():
            return True
        time.sleep(0.05)
    return False


# test_worker_enrols_with_its_real_discovered_hardware()
# Verifies enrolment.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R2 and R3. The worker is configured by a token alone -- nothing about
# its hardware is passed in -- and what it reports is read off the machine.
def test_worker_enrols_with_its_real_discovered_hardware(tmp_path: Path) -> None:

    coordinator = FakeCoordinator()
    runner, state = _runner(coordinator, tmp_path)

    runner.ensure_enrolled()

    # Exactly one enrolment, carrying discovered capabilities.
    assert len(coordinator.enrolments) == 1
    capabilities = coordinator.enrolments[0]["capabilities"]

    # Slot count, CUDA devices, engines and reductions are all discovered.
    assert capabilities["slots"] == 1
    assert "cuda_device_count" in capabilities
    assert {"name": "v3_dirpad", "version": "1"} in capabilities["reductions"]
    assert any(entry["engine"] == "marp_tracking" for entry in capabilities["engines"])

    # The real hardware snapshot is included, not a description of it.
    assert "cpu" in capabilities["resources"]
    assert "memory" in capabilities["resources"]

    # The assigned id is recorded in the worker's state.
    assert state.describe()["enrolled"] is True


# test_identity_survives_a_restart(tmp_path)
# Verifies that a restarted worker is the same worker.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R3's persistence clause. A worker that re-enrolled on every restart
# would appear in MARP as a new machine each time the service bounced.
def test_identity_survives_a_restart(tmp_path: Path) -> None:

    first = FakeCoordinator()
    runner_one, _ = _runner(first, tmp_path)
    runner_one.ensure_enrolled()
    first_identity = runner_one.identity.local_id
    first_worker_id = runner_one.identity.worker_id

    # A second runner over the same state directory is a restarted worker.
    second = FakeCoordinator()
    runner_two, _ = _runner(second, tmp_path)
    runner_two.ensure_enrolled()

    # Same identity, and it did not enrol again.
    assert runner_two.identity.local_id == first_identity
    assert runner_two.identity.worker_id == first_worker_id
    assert second.enrolments == []


# test_re_enrolment_keeps_the_local_id(tmp_path)
# Verifies re-enrolment when the coordinator has lost the worker's record.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves R3's second clause. The machine-local id is stable, so MARP can
# recognize the same machine coming back rather than accumulating duplicates.
def test_re_enrolment_keeps_the_local_id(tmp_path: Path) -> None:

    coordinator = FakeCoordinator()
    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()
    local_id = runner.identity.local_id

    # Forced, as it would be when a call is answered "unknown worker".
    runner.ensure_enrolled(force=True)

    assert len(coordinator.enrolments) == 2
    assert coordinator.enrolments[0]["local_id"] == local_id
    assert coordinator.enrolments[1]["local_id"] == local_id


# test_job_runs_in_a_child_process_and_reports_success(tmp_path)
# Verifies the whole happy path.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R4, R6, R7 and R11 together, at the only tier that can see them: a real
# child process runs the engine through run(ctx, spec), its progress arrives over
# the pipe, its results file is handed over by hash, and the outcome is reported.
def test_job_runs_in_a_child_process_and_reports_success(tmp_path: Path) -> None:

    coordinator = FakeCoordinator(offers=[_job_for(tmp_path, "attempt-happy", frames=5)])
    runner, state = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()

    # Take the job.
    assert runner._poll_once() is True

    # It is pinned to a slot, and /status can see it.
    assert len(state.describe()["active_jobs"]) == 1
    assert state.describe()["active_jobs"][0]["slot_index"] == 0

    # Wait for it to finish, servicing heartbeats as a real loop would.
    assert _wait_until(runner, lambda: len(coordinator.results) == 1), "job never reported"

    # It succeeded.
    result = coordinator.results[0]
    assert result["outcome"] == "succeeded"
    assert result["attempt_id"] == "attempt-happy"

    # The lease fields went with the report, all three of them.
    assert result["worker_id"] == "worker-for-test"
    assert result["lease_epoch"] == 3

    # The engine processed the whole range.
    summary = result["result"]["summary"]
    assert summary["frames_expected"] == 5
    assert summary["frames_processed"] == 5
    assert summary["stopped_early"] is False

    # The results file was offered by hash, and no detections travelled inline.
    assert len(coordinator.artifact_checks) == 1
    check = coordinator.artifact_checks[0]
    assert len(check["sha256"]) == 64
    assert check["size_bytes"] > 0
    assert "detections" not in result["result"]

    # The slot is free again.
    assert state.describe()["active_jobs"] == []
    assert runner.free_slots() == 1


# test_artifact_is_uploaded_only_when_the_coordinator_asks(tmp_path)
# Verifies the two-step artifact hand-off.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R11's point. The hash is offered first; the bytes move only if the
# coordinator does not already have them. A worker that always uploaded would
# resend the same results file for every retried attempt.
def test_artifact_is_uploaded_only_when_the_coordinator_asks(tmp_path: Path) -> None:

    coordinator = FakeCoordinator(offers=[_job_for(tmp_path, "attempt-upload", frames=3)])
    coordinator.already_have_artifact = False

    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()
    runner._poll_once()
    assert _wait_until(runner, lambda: len(coordinator.results) == 1)

    # Checked, then uploaded, because the coordinator said it did not have it.
    assert len(coordinator.artifact_checks) == 1
    assert len(coordinator.uploads) == 1
    assert coordinator.uploads[0]["bytes"] > 0

    # And the report says it was delivered.
    artifacts = coordinator.results[0]["result"]["artifacts"]
    assert artifacts["observations"]["delivered"] is True
    assert artifacts["observations"]["upload"] == "sent"


# test_artifact_is_not_uploaded_when_already_held(tmp_path)
# Verifies the other branch of the hand-off.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves the saving is real: nothing is sent when the coordinator says it has it.
def test_artifact_is_not_uploaded_when_already_held(tmp_path: Path) -> None:

    coordinator = FakeCoordinator(offers=[_job_for(tmp_path, "attempt-cached", frames=3)])
    coordinator.already_have_artifact = True

    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()
    runner._poll_once()
    assert _wait_until(runner, lambda: len(coordinator.results) == 1)

    assert coordinator.uploads == []
    artifacts = coordinator.results[0]["result"]["artifacts"]
    assert artifacts["observations"]["upload"] == "not_needed"


# test_cancel_in_a_heartbeat_stops_a_running_job(tmp_path)
# Verifies cancellation of a job that is actually running.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R5, and this is the test the doctrine is about. A unit test can show
# that should_stop() reads a file; only this one shows that a heartbeat
# answering `cancel` reaches a live child process, stops it, and has it report
# itself cancelled with the work it had already done.
#
# The job is paced so it is genuinely mid-run when the cancel lands -- a job
# that had already finished would pass this test while cancelling nothing.
def test_cancel_in_a_heartbeat_stops_a_running_job(tmp_path: Path) -> None:

    # 2000 frames at 5ms each: about ten seconds of work, far longer than the
    # test needs, so the cancel certainly arrives mid-run.
    coordinator = FakeCoordinator(
        offers=[_job_for(tmp_path, "attempt-cancel", frames=2000, frame_delay_s=0.005)]
    )

    # Answer `continue` for the first heartbeat, then `cancel`.
    coordinator.heartbeat_action = lambda count: "continue" if count < 2 else "cancel"

    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()
    runner._poll_once()

    # The job is running before anything is cancelled.
    job = runner._jobs_by_slot[0]
    assert job.is_running()

    # Service until it reports. The runner sends heartbeats, gets `cancel`, and
    # creates the stop file the child is checking.
    assert _wait_until(runner, lambda: len(coordinator.results) == 1), "cancelled job never reported"

    result = coordinator.results[0]

    # Reported as cancelled, not failed and not succeeded.
    assert result["outcome"] == "cancelled"

    # The engine noticed and unwound rather than being killed.
    summary = result["result"]["summary"]
    assert summary["stopped_early"] is True

    # It stopped part way: some work done, but not all of it. This is the
    # assertion that makes the test about cancellation rather than about
    # completion -- 2000 of 2000 frames would mean nothing was cancelled.
    assert 0 < summary["frames_processed"] < 2000

    # And the partial work was still handed over rather than thrown away.
    assert len(coordinator.artifact_checks) == 1
    assert summary["results"]["line_count"] == summary["frames_processed"]


# test_abandon_in_a_heartbeat_also_stops_the_job(tmp_path)
# Verifies the second stop action.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves R5 for `abandon`, which is what a lease-epoch mismatch is answered
# with -- a worker that ignored it would keep running work it no longer owns.
def test_abandon_in_a_heartbeat_also_stops_the_job(tmp_path: Path) -> None:

    coordinator = FakeCoordinator(
        offers=[_job_for(tmp_path, "attempt-abandon", frames=2000, frame_delay_s=0.005)]
    )
    coordinator.heartbeat_action = "abandon"

    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()
    runner._poll_once()
    assert _wait_until(runner, lambda: len(coordinator.results) == 1)

    assert coordinator.results[0]["outcome"] == "cancelled"
    assert coordinator.results[0]["result"]["summary"]["frames_processed"] < 2000


# test_pause_in_a_heartbeat_leaves_the_running_job_alone(tmp_path)
# Verifies that pause is not a stop.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves A5's settled answer. Pause stops the worker taking NEW work; a job
# already running finishes. Conflating the two would mean an operator freeing
# their GPU silently discarded an hour of somebody's inference.
def test_pause_in_a_heartbeat_leaves_the_running_job_alone(tmp_path: Path) -> None:

    coordinator = FakeCoordinator(offers=[_job_for(tmp_path, "attempt-pause", frames=10)])
    coordinator.heartbeat_action = "pause"

    runner, state = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()
    runner._poll_once()

    assert _wait_until(runner, lambda: len(coordinator.results) == 1)

    # The job ran to completion despite the pause.
    assert coordinator.results[0]["outcome"] == "succeeded"
    assert coordinator.results[0]["result"]["summary"]["frames_processed"] == 10

    # And the worker is now paused, so it will not be offered more.
    assert state.is_paused() is True
    assert runner.free_slots() == 0


# test_paused_worker_reports_no_free_slots(tmp_path)
# Verifies how pause reaches the coordinator.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves R1's operator half joins up with R4: a paused worker asks for nothing,
# which needs no separate coordinator concept.
def test_paused_worker_reports_no_free_slots(tmp_path: Path) -> None:

    coordinator = FakeCoordinator()
    runner, state = _runner(coordinator, tmp_path, slots=2)

    assert runner.free_slots() == 2

    state.set_paused(True, reason="operator needs the GPU")
    assert runner.free_slots() == 0

    state.set_paused(False)
    assert runner.free_slots() == 2


# test_unknown_heartbeat_action_is_treated_as_continue(tmp_path)
# Verifies forward compatibility.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves a deliberate choice: an older worker against a newer coordinator keeps
# working rather than stopping every job on an action it does not recognize.
def test_unknown_heartbeat_action_is_treated_as_continue(tmp_path: Path) -> None:

    coordinator = FakeCoordinator(offers=[_job_for(tmp_path, "attempt-future", frames=5)])
    coordinator.heartbeat_action = "throttle_to_half_speed"

    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()
    runner._poll_once()
    assert _wait_until(runner, lambda: len(coordinator.results) == 1)

    assert coordinator.results[0]["outcome"] == "succeeded"


# test_job_that_cannot_run_is_refused_before_it_starts(tmp_path)
# Verifies immediate refusal.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R14. A job naming an engine this worker does not have is reported
# failed with the reason, and the child process still exits cleanly rather than
# the runner losing track of it.
def test_job_that_cannot_run_is_refused_before_it_starts(tmp_path: Path) -> None:

    offer = _job_for(tmp_path, "attempt-no-engine", frames=5)
    offer["spec"]["engine"] = "an_engine_this_worker_does_not_have"

    coordinator = FakeCoordinator(offers=[offer])
    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()
    runner._poll_once()

    assert _wait_until(runner, lambda: len(coordinator.results) == 1)

    result = coordinator.results[0]
    assert result["outcome"] == "failed"

    # And it says why, by name, rather than reporting an opaque failure.
    message = str(result["result"])
    assert "an_engine_this_worker_does_not_have" in message


# test_malformed_spec_is_rejected_without_launching_anything(tmp_path)
# Verifies validation of the coordinator's offer.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves the boundary check in R9. An inverted range is refused at the schema,
# before a child process exists for it -- rather than after a job has seeked to
# a frame past its own end and read nothing.
def test_malformed_spec_is_rejected_without_launching_anything(tmp_path: Path) -> None:

    offer = _job_for(tmp_path, "attempt-bad-range", frames=5)
    offer["spec"]["range"] = {"start_frame": 5000, "end_frame": 1000}

    coordinator = FakeCoordinator(offers=[offer])
    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()

    # Nothing was taken.
    assert runner._poll_once() is False
    assert runner._jobs_by_slot == {}

    # And the coordinator was told, rather than being left to time the lease out.
    assert len(coordinator.results) == 1
    assert coordinator.results[0]["outcome"] == "failed"
    assert coordinator.results[0]["result"]["reason"] == "invalid_spec"


# test_a_job_in_flight_at_restart_is_reported_lost(tmp_path)
# Verifies restart survival.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R15. The worker writes an in-flight record before launching a job and
# clears it when the job finishes, so a record still present at start means the
# worker died mid-job. It is reported `lost` so MARP can re-queue it -- silent
# forgetting is the alternative and it is worse: the job sits leased until the
# lease expires and nobody knows why nothing happened.
def test_a_job_in_flight_at_restart_is_reported_lost(tmp_path: Path) -> None:

    import json

    # Stand in for what the previous run left behind when it was killed.
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "in-flight.json").write_text(
        json.dumps(
            {
                "attempts": [
                    {
                        "attempt_id": "attempt-was-running",
                        "worker_id": "worker-for-test",
                        "lease_epoch": 9,
                        "slot_index": 0,
                        "progress": {"done": 431, "total": 1800, "unit": "frames"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    coordinator = FakeCoordinator()
    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()

    runner.report_lost_job()

    # Reported lost, with the lease it was running under and how far it got.
    assert len(coordinator.results) == 1
    report = coordinator.results[0]
    assert report["outcome"] == "lost"
    assert report["attempt_id"] == "attempt-was-running"
    assert report["lease_epoch"] == 9
    assert report["result"]["progress"]["done"] == 431

    # The record is cleared, so the same job is not reported lost forever.
    assert not (state_dir / "in-flight.json").is_file()

    # And the worker goes on to take work.
    assert runner._poll_once() is False
    assert len(coordinator.polls) == 1


# test_in_flight_record_is_written_and_cleared_around_a_job(tmp_path)
# Verifies the record's lifecycle.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves the other half of R15: a clean run leaves nothing behind, so the next
# start does not report a job lost that actually completed.
def test_in_flight_record_is_written_and_cleared_around_a_job(tmp_path: Path) -> None:

    coordinator = FakeCoordinator(offers=[_job_for(tmp_path, "attempt-lifecycle", frames=200, frame_delay_s=0.002)])
    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()

    inflight = tmp_path / "state" / "in-flight.json"
    assert not inflight.is_file()

    runner._poll_once()

    # Written before the child was launched.
    assert inflight.is_file()

    assert _wait_until(runner, lambda: len(coordinator.results) == 1)

    # Cleared once it finished.
    assert not inflight.is_file()


# test_progress_reaches_the_coordinator_and_advances(tmp_path)
# Verifies that heartbeats carry real progress.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R4's heartbeat clause and one of the acceptance criteria: progress that
# advances. A worker reporting a constant zero would satisfy every other test
# here, so this asserts movement rather than presence.
def test_progress_reaches_the_coordinator_and_advances(tmp_path: Path) -> None:

    coordinator = FakeCoordinator(
        offers=[_job_for(tmp_path, "attempt-progress", frames=600, frame_delay_s=0.005)]
    )
    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()
    runner._poll_once()

    # Wait until progress is actually non-zero. The first heartbeats read zero
    # legitimately -- launching a Python interpreter on Windows takes about a
    # second and the loop heartbeats far faster than that -- so comparing the
    # earliest two would be asserting startup latency, not progress.
    assert _wait_until(
        runner,
        lambda: any(beat["progress"]["done"] > 0 for beat in coordinator.heartbeats),
    ), "progress never became non-zero"

    first_nonzero = next(
        index
        for index, beat in enumerate(coordinator.heartbeats)
        if beat["progress"]["done"] > 0
    )
    baseline = coordinator.heartbeats[first_nonzero]["progress"]["done"]

    # Then wait for it to move past that point.
    assert _wait_until(
        runner,
        lambda: any(
            beat["progress"]["done"] > baseline
            for beat in coordinator.heartbeats[first_nonzero:]
        )
        or len(coordinator.results) == 1,
    ), "progress never advanced"

    dones = [beat["progress"]["done"] for beat in coordinator.heartbeats]

    # It moved. Strictly greater, so a stuck counter fails.
    assert max(dones) > baseline or len(coordinator.results) == 1

    # And each heartbeat says which slot the job is on and how long it has run.
    assert coordinator.heartbeats[0]["progress"]["slot_index"] == 0
    assert coordinator.heartbeats[0]["progress"]["elapsed_s"] >= 0

    # Clean up the still-running job so it does not outlive the test.
    runner._jobs_by_slot[0].request_stop()
    _wait_until(runner, lambda: len(coordinator.results) == 1)


# test_engine_logs_reach_the_coordinator_as_events(tmp_path)
# Verifies the event channel.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
# Proves that ctx.log() from inside a child process ends up at the coordinator,
# batched and keyed by seq -- the path a real job's narrative takes.
def test_engine_logs_reach_the_coordinator_as_events(tmp_path: Path) -> None:

    coordinator = FakeCoordinator(offers=[_job_for(tmp_path, "attempt-events", frames=5)])
    runner, _ = _runner(coordinator, tmp_path)
    runner.ensure_enrolled()
    runner._poll_once()
    assert _wait_until(runner, lambda: len(coordinator.results) == 1)

    # The mock engine logs the range it was given; that line must have arrived.
    logs = [event for event in coordinator.events if event.get("kind") == "log"]
    assert any("1000..1005" in event.get("message", "") for event in logs)

    # Every event carries a seq, so a resent batch can be deduplicated.
    assert all("seq" in event for event in coordinator.events)


# test_slots_limit_how_many_jobs_run_at_once(tmp_path)
# Verifies the slot accounting.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R6's host-level half: the host owns how many jobs run at once, and a
# job is pinned to a distinct slot. Two jobs on one slot would mean two
# processes on one GPU, which is how a worker runs out of VRAM.
def test_slots_limit_how_many_jobs_run_at_once(tmp_path: Path) -> None:

    coordinator = FakeCoordinator(
        offers=[
            _job_for(tmp_path, "attempt-a", frames=400, frame_delay_s=0.005),
            _job_for(tmp_path, "attempt-b", frames=400, frame_delay_s=0.005),
            _job_for(tmp_path, "attempt-c", frames=400, frame_delay_s=0.005),
        ]
    )
    runner, _ = _runner(coordinator, tmp_path, slots=2)
    runner.ensure_enrolled()

    # Two jobs fit.
    assert runner._poll_once() is True
    assert runner._poll_once() is True
    assert runner.free_slots() == 0

    # On distinct slots.
    assert sorted(runner._jobs_by_slot) == [0, 1]

    # The third has nowhere to go, and the loop does not ask for it.
    assert runner.free_slots() == 0

    # Clean up: stop both so nothing outlives the test.
    for job in list(runner._jobs_by_slot.values()):
        job.request_stop()
    assert _wait_until(runner, lambda: len(coordinator.results) == 2)


# test_worker_never_receives_a_push_address(tmp_path)
# Verifies that push is unrepresentable.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R1 structurally rather than behaviourally. The decision was that push
# be unrepresentable -- no host, url or port field anywhere in the contract --
# so this asserts the job spec schema has nowhere to put one. A behavioural test
# cannot show the absence of a capability; a schema check can.
def test_worker_never_receives_a_push_address(tmp_path: Path) -> None:

    from marp_inference_worker.jobs.job_spec import AttemptEnvelope, JobSpec

    # Names that would let MARP say "connect back to me here". None of these may
    # appear anywhere in the contract, at any nesting level.
    forbidden = {
        "host",
        "hostname",
        "port",
        "callback",
        "callback_url",
        "endpoint",
        "worker_url",
        "worker_host",
        "reply_to",
        "webhook",
    }

    for model in (JobSpec, AttemptEnvelope):
        assert not (set(model.model_fields) & forbidden), model.__name__

    # Nor in the nested models the spec is built from.
    for nested in ("model", "video", "range", "reduction"):
        annotation = JobSpec.model_fields[nested].annotation
        assert not (set(annotation.model_fields) & forbidden), nested

    # There is exactly one url in the whole contract, and it is an outbound
    # fetch: where the worker GOES to get a model artifact. That direction is
    # the distinction that matters -- a locator the worker reads from cannot be
    # used to reach the worker, whereas a callback address could.
    model_ref = JobSpec.model_fields["model"].annotation
    assert "url" in model_ref.model_fields

    urls_elsewhere = [
        (name, field)
        for name, field in list(JobSpec.model_fields.items())
        + list(AttemptEnvelope.model_fields.items())
        if "url" in name.lower()
    ]
    assert urls_elsewhere == [], urls_elsewhere

    for nested in ("video", "range", "reduction"):
        annotation = JobSpec.model_fields[nested].annotation
        assert not [name for name in annotation.model_fields if "url" in name.lower()], nested


# test_two_piece_split_covers_every_frame_exactly_once(tmp_path)
# Verifies the half-open range convention across a piece boundary.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R9's convention rather than merely documenting it: a range is
# [start_frame, end_frame), so consecutive pieces share a bound and the union of
# two adjacent pieces is exactly their span with nothing dropped and nothing
# done twice.
#
# This is the test that would have caught a divergence between the worker and
# the coordinator, and until now nothing on either side would have. MARP_API had
# implemented both bounds inclusive, which drops one frame at every piece
# boundary -- silently, because each piece looks complete on its own, and a
# ten-hour video in ten pieces would lose nine frames with every other test
# still green. The convention was settled half-open on 2026-09-09.
#
# Run at the `runner` tier deliberately. The schema can only reject an inverted
# range; whether a job actually processes end_frame is a question about what the
# engine did, so it has to be read off two real jobs' results.
def test_two_piece_split_covers_every_frame_exactly_once(tmp_path: Path) -> None:

    import json

    # One video split at 300. The two pieces share that bound: the first ends
    # where the second begins, and 300 belongs to the second alone.
    pieces = [
        _job_for(tmp_path, "attempt-piece-one", frames=300, start_frame=0),
        _job_for(tmp_path, "attempt-piece-two", frames=300, start_frame=300),
    ]

    # The ranges are adjacent, which is what makes this a split rather than two
    # unrelated jobs. Asserted so a later edit to the fixture cannot quietly
    # turn it into a test of two disjoint ranges.
    assert pieces[0]["spec"]["range"] == {"start_frame": 0, "end_frame": 300}
    assert pieces[1]["spec"]["range"] == {"start_frame": 300, "end_frame": 600}
    assert pieces[0]["spec"]["range"]["end_frame"] == pieces[1]["spec"]["range"]["start_frame"]

    coordinator = FakeCoordinator(offers=list(pieces))

    # Two slots, so both pieces run as they would on two workers.
    runner, _ = _runner(coordinator, tmp_path, slots=2)
    runner.ensure_enrolled()

    assert runner._poll_once() is True
    assert runner._poll_once() is True

    assert _wait_until(runner, lambda: len(coordinator.results) == 2), "both pieces never reported"

    # Both succeeded, and each processed exactly its own count -- which for a
    # half-open range is end minus start, a plain subtraction.
    frames_by_attempt = {}
    for result in coordinator.results:
        assert result["outcome"] == "succeeded", result
        summary = result["result"]["summary"]
        assert summary["frames_expected"] == 300
        assert summary["frames_processed"] == 300
        frames_by_attempt[result["attempt_id"]] = summary

    assert set(frames_by_attempt) == {"attempt-piece-one", "attempt-piece-two"}

    # Now read the frame indices the engine actually wrote, per piece. The
    # summary counts could both be 300 while overlapping or skipping, so the
    # counts alone do not settle it -- the indices do.
    indices_by_attempt: dict[str, list[int]] = {}
    for attempt_id in frames_by_attempt:
        results_path = tmp_path / "state" / "jobs" / attempt_id / "work" / "observations.jsonl"
        assert results_path.is_file(), f"no results file for {attempt_id}"

        indices_by_attempt[attempt_id] = [
            json.loads(line)["frame"]
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    first = indices_by_attempt["attempt-piece-one"]
    second = indices_by_attempt["attempt-piece-two"]

    # Each piece stayed inside its own half-open range. The upper bound is the
    # assertion that matters: the first piece must NOT have touched frame 300.
    assert min(first) == 0 and max(first) == 299
    assert min(second) == 300 and max(second) == 599
    assert 300 not in first, "first piece processed its end_frame; the range is half-open"

    # No frame was done twice, within a piece or across the boundary.
    assert len(first) == len(set(first))
    assert len(second) == len(set(second))
    assert not (set(first) & set(second)), "pieces overlap at the boundary"

    # And the union is exactly the whole span, so nothing was dropped. This
    # single assertion is what an inclusive coordinator would have failed:
    # frame 300 would be missing from both pieces.
    assert sorted(first + second) == list(range(600))
