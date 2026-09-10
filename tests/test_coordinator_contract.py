# test_coordinator_contract.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Wire-level tests for the worker's side of the MARP coordinator contract.
# Every test here runs the real CoordinatorClient against a real HTTP server on
# loopback and asserts on the bytes that actually left the process: the field
# names, the types, and the shape of each body.
#
# This tier exists because nothing below it can see the bugs it guards. The
# runner tests substitute a fake for the client, so they prove the runner's
# decisions and say nothing about what the client sends; the coordinator's own
# suite proves what it accepts. Between the two sat six divergences that both
# suites passed straight through -- a worker id sent as a string and refused as
# not-an-integer, a metric batch under a kind the coordinator does not accept, a
# timestamp sent as a float into a timestamp column, artifacts nested inside a
# field the coordinator never reads. Each of them broke the whole loop, and each
# of them was invisible until the two halves were run together.
#
# A real socket rather than a mocked transport, deliberately: a mock records what
# the client meant, and the failures above were all about what it actually said.

# json decodes the recorded request bodies.
import json

# threading runs the recording server beside the test.
import threading

# The standard library's own HTTP server is enough, and pulls in no dependency.
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Any types the recorded bodies.
from typing import Any

import pytest

from marp_inference_worker.jobs.coordinator_client import (
    CoordinatorClient,
    CoordinatorError,
)
from marp_inference_worker.jobs.job_spec import AttemptEnvelope


# Recorder
# Holds what the server saw and what it should answer.
# Shared between the test and the request handler, which cannot easily be given
# constructor arguments.
class Recorder:

    # __init__()
    # Builds an empty recorder answering `{}` to everything.
    # Inputs: none.
    # Output: initialized Recorder.
    def __init__(self) -> None:

        # One entry per request: method, path, headers and decoded body.
        self.requests: list[dict[str, Any]] = []

        # Path fragment -> (status, body mapping). Matched anywhere in the path,
        # because the upload route's path ends in a hash rather than its name.
        self.responses: dict[str, tuple[int, Any]] = {}

    # answer()
    # Scripts the answer for one route.
    # Inputs: a fragment of the path to match, the status, and the body.
    # Output: none.
    def answer(self, fragment: str, status: int = 200, body: Any = None) -> None:

        self.responses[fragment] = (status, {} if body is None else body)

    # for_path()
    # Returns the recorded requests whose path contains a fragment.
    # Inputs: the path fragment.
    # Output: list of recorded requests, in order.
    def for_path(self, fragment: str) -> list[dict[str, Any]]:

        return [record for record in self.requests if fragment in record["path"]]


# _make_handler()
# Builds a request handler class bound to one recorder.
# Inputs: the recorder to write into.
# Output: a BaseHTTPRequestHandler subclass.
# Use this from the fixture; the handler class is instantiated per request, so
# the recorder has to be closed over rather than passed in.
def _make_handler(recorder: Recorder):

    class Handler(BaseHTTPRequestHandler):

        # Silence the default stderr access log, which would drown the run.
        def log_message(self, *args: Any) -> None:
            pass

        def do_POST(self) -> None:

            # Read exactly what was sent, so nothing is inferred.
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""

            # Keep the raw bytes as well as the decoded body: an upload is not
            # JSON and its length is the thing worth asserting.
            try:
                body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = None

            recorder.requests.append({
                "method": "POST",
                "path": self.path,
                "headers": dict(self.headers),
                "body": body,
                "raw_bytes": len(raw),
            })

            status, payload = 200, {}
            for fragment, scripted in recorder.responses.items():
                if fragment in self.path:
                    status, payload = scripted
                    break

            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


# coordinator (fixture)
# Runs a recording HTTP server and yields (client, recorder).
# Inputs: none.
# Output: a CoordinatorClient pointed at the server, and the recorder.
# The server is shut down after each test so no socket outlives its test.
@pytest.fixture()
def coordinator():

    recorder = Recorder()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(recorder))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address[0], server.server_address[1]
    client = CoordinatorClient(
        base_url=f"http://{host}:{port}",
        service_token="svc_test_token",
        request_timeout_s=10.0,
        long_poll_timeout_s=10.0,
    )

    try:
        yield client, recorder
    finally:
        client.close()
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


# test_every_call_sends_the_worker_id_as_the_integer_marp_issued()
# Verifies the identifier type on every state-changing call.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
#
# The first thing the live round trip found. MARP issues `worker_id` as an
# integer and validates it as one; the worker keeps it as text because it is also
# a directory name, and was sending the text. Every poll, heartbeat, event batch
# and result was answered `400: worker_id is required and must be an integer`,
# and the loop never got past enrolment.
def test_every_call_sends_the_worker_id_as_the_integer_marp_issued(coordinator) -> None:

    client, recorder = coordinator
    recorder.answer("/poll", 200, {"attempt_id": 1})

    client.poll(worker_id="65", free_slot_indexes=[0], capabilities={}, wait_seconds=5)
    client.heartbeat(attempt_id="228", worker_id="65", lease_epoch=1, progress={"done": 1})
    client.post_events(
        attempt_id="228",
        worker_id="65",
        lease_epoch=1,
        events=[{"seq": 0, "kind": "log", "message": "hello"}],
    )
    client.report_result(attempt_id="228", worker_id="65", lease_epoch=1, outcome="succeeded")

    # All four, checked as a set rather than one at a time: a fix applied to one
    # call and forgotten on the others is exactly what happened here.
    assert len(recorder.requests) == 4

    for record in recorder.requests:
        assert record["body"]["worker_id"] == 65, record["path"]
        assert isinstance(record["body"]["worker_id"], int), record["path"]


# test_an_identifier_that_is_not_a_coordinator_id_is_refused()
# Verifies the identifier conversion fails loudly.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
# A silent fallback would send something the coordinator refuses, at which point
# the error is a 400 forty lines away from its cause.
def test_an_identifier_that_is_not_a_coordinator_id_is_refused(coordinator) -> None:

    client, _ = coordinator

    with pytest.raises(CoordinatorError, match="worker_id"):
        client.heartbeat(
            attempt_id="228",
            worker_id="not-a-number",
            lease_epoch=1,
            progress={},
        )


# test_the_lease_offer_validates_with_the_integer_ids_marp_issues()
# Verifies the attempt envelope accepts what a poll actually returns.
# Inputs: none.
# Output: pytest pass/fail result.
#
# MARP's `GpuLease` carries `attempt_id` as an integer and carries no
# `worker_id` at all. The envelope required text and required the worker id, so
# every offered job failed validation and was reported straight back as an
# invalid spec -- the worker refused every job it was given.
def test_the_lease_offer_validates_with_the_integer_ids_marp_issues() -> None:

    # Shaped exactly as the coordinator's poll answers, plus the worker id the
    # runner fills in from its own identity.
    envelope = AttemptEnvelope.model_validate({
        "attempt_id": 228,
        "worker_id": 65,
        "lease_epoch": 1,
        "spec": {
            "engine": "mock",
            "model": {"name": "m.pt", "sha256": "a" * 64},
            "video": {
                "url": "http://media.invalid/v.mp4",
                "source_name": "v.mp4",
                "jellyfin_item_id": "item-1",
            },
            "range": {"start_frame": 0, "end_frame": 300},
            "params": {},
            # An integer, which is how MARP's own published spec documents it.
            "reduction": {"name": "v3_dirpad", "version": 1},
        },
    })

    # Held as text, because the worker uses both as path segments.
    assert envelope.attempt_id == "228"
    assert envelope.worker_id == "65"

    # And the reduction version normalized to the registry's own key type.
    assert envelope.spec.reduction.version == "1"


# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------


# test_poll_names_its_free_slots_and_asks_to_be_held_open()
# Verifies the poll body against MARP's GpuPollRequest.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
#
# Two separate defects in one body. The worker sent `free_slots`, a count, where
# the coordinator reads `slot_indexes` -- so every attempt was recorded on slot
# 0 whatever slot was actually running it. And it sent no `wait_seconds`, which
# defaults to zero: the long poll of R4 was never once exercised, because the
# coordinator answered immediately every single time.
def test_poll_names_its_free_slots_and_asks_to_be_held_open(coordinator) -> None:

    client, recorder = coordinator
    recorder.answer("/poll", 204)

    client.poll(worker_id="65", free_slot_indexes=[1, 2], capabilities={}, wait_seconds=45)

    body = recorder.for_path("/poll")[0]["body"]

    assert body["slot_indexes"] == [1, 2]
    assert body["wait_seconds"] == 45
    assert "free_slots" not in body


# test_poll_sends_the_capabilities_the_attempt_is_snapshotted_from()
# Verifies the machine description travels with the poll.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
# The coordinator copies these onto the attempt it opens, and that snapshot is
# the only record of what the machine was when it ran the job. Sending none left
# every attempt's `capabilities_snapshot` null.
def test_poll_sends_the_capabilities_the_attempt_is_snapshotted_from(coordinator) -> None:

    client, recorder = coordinator
    recorder.answer("/poll", 204)

    client.poll(
        worker_id="65",
        free_slot_indexes=[0],
        capabilities={"slots": 1, "cuda_device_count": 1},
        wait_seconds=0,
    )

    body = recorder.for_path("/poll")[0]["body"]

    assert body["capabilities"] == {"slots": 1, "cuda_device_count": 1}


# test_a_204_from_the_poll_is_no_work_rather_than_an_error()
# Verifies the idle answer.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
def test_a_204_from_the_poll_is_no_work_rather_than_an_error(coordinator) -> None:

    client, recorder = coordinator
    recorder.answer("/poll", 204)

    assert client.poll(worker_id="65", free_slot_indexes=[0], wait_seconds=0) is None


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


# test_heartbeat_says_which_state_the_attempt_is_in()
# Verifies the heartbeat reports the attempt's state.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
#
# The coordinator leaves an attempt in `assigned` until a worker says otherwise,
# and the worker was saying nothing -- so an attempt that ran for an hour still
# read as merely assigned, and nothing could tell a working machine from one
# that leased a job and never started it.
def test_heartbeat_says_which_state_the_attempt_is_in(coordinator) -> None:

    client, recorder = coordinator

    client.heartbeat(attempt_id="228", worker_id="65", lease_epoch=1, progress={"done": 5})
    client.heartbeat(
        attempt_id="228",
        worker_id="65",
        lease_epoch=1,
        progress={"done": 300},
        state="uploading",
    )

    bodies = [record["body"] for record in recorder.for_path("/heartbeat")]

    # Running by default, because that is what a heartbeat during a job means.
    assert bodies[0]["state"] == "running"
    assert bodies[1]["state"] == "uploading"

    # And it is one of the three the coordinator will accept -- a terminal state
    # here is refused, because only the result route may end an attempt.
    for body in bodies:
        assert body["state"] in ("preparing", "running", "uploading")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


# test_events_arrive_as_seq_kind_at_payload()
# Verifies the event envelope.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
#
# The child writes its fields at the top level of the event; the coordinator
# reads them from a `payload` object and stores that column. Sent flat, every log
# line and every metric was recorded with a null payload -- the events arrived,
# were counted, and carried nothing.
def test_events_arrive_as_seq_kind_at_payload(coordinator) -> None:

    client, recorder = coordinator

    client.post_events(
        attempt_id="228",
        worker_id="65",
        lease_epoch=1,
        events=[{"seq": 7, "kind": "log", "at": 1757443200.5, "level": "info", "message": "hi"}],
    )

    event = recorder.for_path("/events")[0]["body"]["events"][0]

    assert set(event) == {"seq", "kind", "at", "payload"}
    assert event["seq"] == 7
    assert event["payload"]["message"] == "hi"
    assert event["payload"]["level"] == "info"


# test_a_metric_batch_uses_the_kind_the_coordinator_accepts()
# Verifies the kind translation.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
#
# The child's context emits `metrics`; the coordinator accepts `metric` or `log`
# and refuses anything else. One refused event fails the whole batch, so a
# single metric took every log line beside it down with it.
def test_a_metric_batch_uses_the_kind_the_coordinator_accepts(coordinator) -> None:

    client, recorder = coordinator

    client.post_events(
        attempt_id="228",
        worker_id="65",
        lease_epoch=1,
        events=[
            {"seq": 0, "kind": "metrics", "step": 3, "phase": "mock", "metrics": {"frames": 4}},
            {"seq": 1, "kind": "something_new", "detail": "from a newer worker"},
        ],
    )

    events = recorder.for_path("/events")[0]["body"]["events"]

    assert events[0]["kind"] == "metric"
    assert events[0]["payload"]["phase"] == "mock"

    # An unknown kind becomes a log rather than being dropped or refused, and
    # says what it was, so a newer worker against an older coordinator loses
    # nothing.
    assert events[1]["kind"] == "log"
    assert events[1]["payload"]["child_kind"] == "something_new"


# test_event_timestamps_are_iso_strings_not_epoch_numbers()
# Verifies the timestamp type.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
#
# The coordinator coalesces `at` into a timestamp column. A float epoch there is
# a Postgres type error, so the events route answered 500 for every batch the
# worker sent -- not a validation message, a server error.
def test_event_timestamps_are_iso_strings_not_epoch_numbers(coordinator) -> None:

    client, recorder = coordinator

    client.post_events(
        attempt_id="228",
        worker_id="65",
        lease_epoch=1,
        events=[{"seq": 0, "kind": "log", "at": 1757443200.5, "message": "x"}],
    )

    event = recorder.for_path("/events")[0]["body"]["events"][0]

    assert isinstance(event["at"], str)
    assert event["at"].startswith("2025-")


# test_a_long_run_of_events_is_split_into_batches()
# Verifies the batch limit is respected.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
#
# The coordinator refuses a batch of more than 500. A tracking job reports a
# metric per frame, so a minute of video passes that -- and the refusal is of
# the whole batch, so the first over-long send would lose every event in it.
def test_a_long_run_of_events_is_split_into_batches(coordinator) -> None:

    client, recorder = coordinator

    events = [{"seq": index, "kind": "metrics", "metrics": {"frames": index}} for index in range(1200)]
    client.post_events(attempt_id="228", worker_id="65", lease_epoch=1, events=events)

    batches = recorder.for_path("/events")

    # Three batches, none over the limit, and every event sent exactly once.
    assert len(batches) == 3
    assert [len(batch["body"]["events"]) for batch in batches] == [500, 500, 200]

    sent = [event["seq"] for batch in batches for event in batch["body"]["events"]]
    assert sent == list(range(1200))


# test_an_event_without_a_sequence_number_is_not_sent()
# Verifies unkeyable events are dropped rather than sent.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
#
# `(attempt_id, seq)` is the event's key and the whole of its replay safety. An
# event with no seq is refused, and refusing one refuses the batch -- so sending
# it would lose the events either side of it too.
def test_an_event_without_a_sequence_number_is_not_sent(coordinator) -> None:

    client, recorder = coordinator

    client.post_events(
        attempt_id="228",
        worker_id="65",
        lease_epoch=1,
        events=[
            {"kind": "log", "message": "no seq at all"},
            {"seq": 4, "kind": "log", "message": "keeps its place"},
        ],
    )

    events = recorder.for_path("/events")[0]["body"]["events"]

    assert [event["seq"] for event in events] == [4]


# test_an_empty_batch_is_not_a_round_trip()
# Verifies nothing is sent when there is nothing to say.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
# The coordinator refuses an empty `events` array, so sending one would turn an
# idle heartbeat into a 400.
def test_an_empty_batch_is_not_a_round_trip(coordinator) -> None:

    client, recorder = coordinator

    client.post_events(attempt_id="228", worker_id="65", lease_epoch=1, events=[])
    client.post_events(
        attempt_id="228",
        worker_id="65",
        lease_epoch=1,
        events=[{"kind": "log", "message": "unkeyable, so nothing is left"}],
    )

    assert recorder.for_path("/events") == []


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


# test_result_names_its_artifacts_at_the_top_level()
# Verifies the terminal report against MARP's GpuResultRequest.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
#
# The one that made a working pipeline lose its results. The worker put the
# artifacts inside a `result` object the coordinator does not read, so every job
# reported success with no artifacts named -- the bytes were staged and then
# orphaned, and `GET /gpu/jobs/:id` showed a succeeded job with nothing behind
# it.
def test_result_names_its_artifacts_at_the_top_level(coordinator) -> None:

    client, recorder = coordinator

    client.report_result(
        attempt_id="228",
        worker_id="65",
        lease_epoch=1,
        outcome="succeeded",
        artifacts=[{"sha256": "b" * 64, "role": "observations"}],
    )

    body = recorder.for_path("/result")[0]["body"]

    assert body["artifacts"] == [{"sha256": "b" * 64, "role": "observations"}]

    # And nothing is hidden in a field the coordinator ignores.
    assert "result" not in body


# test_result_carries_the_reason_a_failure_failed()
# Verifies the failure explanation reaches the attempt row.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
# `failure_reason` is the only part of a failure the coordinator keeps on the
# attempt, so a failure that does not set it is a failure nobody can diagnose.
def test_result_carries_the_reason_a_failure_failed(coordinator) -> None:

    client, recorder = coordinator

    client.report_result(
        attempt_id="228",
        worker_id="65",
        lease_epoch=1,
        outcome="failed",
        failure_reason="preparation_failed: ValueError: no such engine",
    )

    body = recorder.for_path("/result")[0]["body"]

    assert body["outcome"] == "failed"
    assert "no such engine" in body["failure_reason"]


# test_a_successful_result_carries_no_failure_reason()
# Verifies a success says nothing about failing.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
def test_a_successful_result_carries_no_failure_reason(coordinator) -> None:

    client, recorder = coordinator

    client.report_result(attempt_id="228", worker_id="65", lease_epoch=1, outcome="succeeded")

    assert "failure_reason" not in recorder.for_path("/result")[0]["body"]


# test_the_outcome_is_one_the_coordinator_accepts()
# Verifies the outcome vocabulary is the coordinator's.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
#
# The worker used to have a fourth outcome, `lost`, for a job that did not
# survive a restart. The coordinator has three, and answered 400 -- so the job
# stayed leased to a machine that no longer existed until its lease timed out,
# which is exactly the silent loss R15 exists to prevent. A loss is now reported
# as a failure with the loss named in the reason.
def test_the_outcome_is_one_the_coordinator_accepts(coordinator) -> None:

    client, recorder = coordinator

    client.report_result(
        attempt_id="228",
        worker_id="65",
        lease_epoch=1,
        outcome="failed",
        failure_reason="Lost: the worker restarted while this job was running.",
    )

    body = recorder.for_path("/result")[0]["body"]

    assert body["outcome"] in ("succeeded", "failed", "cancelled")
    assert body["failure_reason"].startswith("Lost:")


# ---------------------------------------------------------------------------
# Artifact hand-off
# ---------------------------------------------------------------------------


# test_artifact_check_reports_the_size_under_the_field_the_coordinator_reads()
# Verifies the check body against MARP's GpuArtifactCheckRequest.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
# The worker sent `size_bytes`; the coordinator reads `bytes`, so it recorded
# the size as unknown.
def test_artifact_check_reports_the_size_under_the_field_the_coordinator_reads(coordinator) -> None:

    client, recorder = coordinator
    recorder.answer("/artifacts/check", 200, {"already_have": True})

    client.check_artifact(sha256="c" * 64, size_bytes=10090)

    body = recorder.for_path("/artifacts/check")[0]["body"]

    assert body == {"sha256": "c" * 64, "bytes": 10090}


# test_artifact_upload_posts_to_the_url_the_check_answered_with()
# Verifies the second half of the hand-off.
# Inputs: the coordinator fixture and a temporary directory.
# Output: pytest pass/fail result.
#
# The check answers with a plain `upload_url` string and the route is a POST.
# The worker looked for an `upload` object carrying `url` and `method`, found
# nothing, and reported every artifact as undeliverable -- so no result file had
# ever actually been handed over.
def test_artifact_upload_posts_to_the_url_the_check_answered_with(coordinator, tmp_path) -> None:

    client, recorder = coordinator

    artifact = tmp_path / "observations.jsonl"
    artifact.write_bytes(b'{"frame": 0}\n{"frame": 1}\n')

    client.upload_artifact(f"/api/v2/gpu/artifacts/upload/{'d' * 64}", artifact, attempt_id="228")

    upload = recorder.for_path_upload = recorder.requests[-1]

    # The path the check named, with the attempt quoted as provenance.
    assert upload["path"].startswith("/api/v2/gpu/artifacts/upload/" + "d" * 64)
    assert "attempt_id=228" in upload["path"]

    # The bytes themselves, all of them, and labelled.
    assert upload["raw_bytes"] == artifact.stat().st_size
    assert upload["headers"]["Content-Type"] == "application/octet-stream"


# test_an_upload_the_coordinator_refuses_is_not_a_delivered_result()
# Verifies a failed upload is reported rather than swallowed.
# Inputs: the coordinator fixture and a temporary directory.
# Output: pytest pass/fail result.
def test_an_upload_the_coordinator_refuses_is_not_a_delivered_result(coordinator, tmp_path) -> None:

    client, recorder = coordinator
    recorder.answer("/artifacts/upload", 400, {"error": "hash mismatch"})

    artifact = tmp_path / "observations.jsonl"
    artifact.write_bytes(b"{}\n")

    with pytest.raises(CoordinatorError, match="400"):
        client.upload_artifact(f"/api/v2/gpu/artifacts/upload/{'e' * 64}", artifact)


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


# test_an_error_status_becomes_a_coordinator_error()
# Verifies an HTTP failure is surfaced as the one error type the loop catches.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
def test_an_error_status_becomes_a_coordinator_error(coordinator) -> None:

    client, recorder = coordinator
    recorder.answer("/poll", 400, {"error": {"message": "worker_id is required"}})

    with pytest.raises(CoordinatorError, match="400"):
        client.poll(worker_id="65", free_slot_indexes=[0], wait_seconds=0)


# test_a_coordinator_that_cannot_be_reached_becomes_a_coordinator_error()
# Verifies a transport failure is translated rather than raised through.
# Inputs: none.
# Output: pytest pass/fail result.
#
# **The one that killed the worker outright.** Restarting the coordinator during
# a long poll raised `httpx.RemoteProtocolError`, which is not a
# `CoordinatorError` -- so it went straight through the runner's loop, ended the
# thread, and left a process still serving `/status`, still saying it was idle,
# and never taking work again. On a home connection that is a worker that dies
# the first time its link drops.
def test_a_coordinator_that_cannot_be_reached_becomes_a_coordinator_error() -> None:

    # A port nothing is listening on: connecting refuses at once.
    client = CoordinatorClient(
        base_url="http://127.0.0.1:1",
        service_token="svc_test_token",
        request_timeout_s=2.0,
        long_poll_timeout_s=2.0,
    )

    try:
        with pytest.raises(CoordinatorError, match="could not be reached"):
            client.poll(worker_id="65", free_slot_indexes=[0], wait_seconds=0)
    finally:
        client.close()


# test_an_upload_that_cannot_be_delivered_becomes_a_coordinator_error()
# Verifies the same translation on the upload path.
# Inputs: a temporary directory.
# Output: pytest pass/fail result.
# The upload uses a different httpx call from every other route, so it needs its
# own guard -- and an artifact hand-off is exactly when a link is most loaded.
def test_an_upload_that_cannot_be_delivered_becomes_a_coordinator_error(tmp_path) -> None:

    client = CoordinatorClient(
        base_url="http://127.0.0.1:1",
        service_token="svc_test_token",
        request_timeout_s=2.0,
        long_poll_timeout_s=2.0,
    )

    artifact = tmp_path / "observations.jsonl"
    artifact.write_bytes(b"{}\n")

    try:
        with pytest.raises(CoordinatorError, match="could not be delivered"):
            client.upload_artifact(f"/api/v2/gpu/artifacts/upload/{'f' * 64}", artifact)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Enrolment
# ---------------------------------------------------------------------------


# test_enrolment_sends_the_name_the_coordinator_keys_on()
# Verifies the enrolment body.
# Inputs: the coordinator fixture.
# Output: pytest pass/fail result.
# `name` is required and is what a re-enrolment is matched on. This is the one
# divergence that was already found and fixed; the assertion keeps it fixed.
def test_enrolment_sends_the_name_the_coordinator_keys_on(coordinator) -> None:

    client, recorder = coordinator
    recorder.answer("/workers/enrol", 200, {"worker_id": 65})

    record = client.enrol(
        local_id="a0294ccd-9fca-462e-8ba3-b5a6f5b380e3",
        capabilities={"slots": 1},
        name="SoftwareEngineering-a0294ccd",
        slot_count=1,
        worker_version="0.1.0",
    )

    body = recorder.for_path("/workers/enrol")[0]["body"]

    assert body["name"] == "SoftwareEngineering-a0294ccd"
    assert body["slot_count"] == 1
    assert body["capabilities"] == {"slots": 1}
    assert record["worker_id"] == 65
