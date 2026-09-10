# coordinator_client.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Coordinator transport for the MARP Inference Worker.
# Every call in this file is outbound from the worker: it enrols, long-polls for
# work, heartbeats a lease, posts batched events, offers artifacts by hash and
# reports a terminal result. There is deliberately no inbound listener and no
# host, url or port field anywhere in the job contract, so push is unrepresentable.
# HTTP shape belongs here; scheduling and execution do not.

# httpx makes the outbound calls and is already a project dependency.
import httpx

# datetime turns the child's epoch timestamps into the ISO-8601 the coordinator
# stores events against.
from datetime import datetime, timezone

# Path types the artifact file offered upstream.
from pathlib import Path

# Any and Mapping type the free-form payload bodies and the child's events.
from typing import Any, Mapping


# Route family the worker talks to, from A1 in .marp/task.md.
_API_PREFIX = "/api/v2/gpu"


# Most events the coordinator accepts in one batch.
# Its own limit; a longer run is sent as several batches, which `(attempt_id,
# seq)` makes free. Without this a job reporting a metric per frame overruns the
# limit and the coordinator refuses the WHOLE batch, losing every event in it.
_MAX_EVENTS_PER_BATCH = 500


# The only two event kinds the coordinator accepts from a worker.
# `note` is the coordinator's own kind, for recording why it took a lease away.
_WORKER_EVENT_KINDS = ("metric", "log")


# CoordinatorError
# Raised when a coordinator call fails in a way the caller must see.
# Transient network trouble is retried by the runner's loop, not here.
class CoordinatorError(RuntimeError):
    pass


# coordinator_id()
# Reads back an identifier the coordinator issued, in the type it issued it.
# Inputs: the value as the worker stored it, and a field name for the message.
# Output: the identifier as an integer.
# Use this at every point an id goes onto the wire.
#
# The worker keeps `worker_id` and `attempt_id` as strings because it also uses
# them as directory names and URL segments, but the coordinator issues them as
# integers and validates them as integers. Sending the string was the first
# thing the live round trip found: every poll, heartbeat, event batch and result
# was answered `400: worker_id is required and must be an integer`, and no unit
# test on either side could see it because both sides were self-consistent.
def coordinator_id(value: Any, field: str) -> int:

    # A non-numeric id is a contract violation and must not be sent as a guess.
    try:
        return int(value)
    except (TypeError, ValueError):
        raise CoordinatorError(f"{field} {value!r} is not a coordinator identifier")


# coordinator_event()
# Translates one of the child's stdout events into the coordinator's shape.
# Inputs: the event mapping as the child's context wrote it.
# Output: `{seq, kind, at, payload}`, or None when it cannot be keyed.
# Use this on every event before it is sent.
#
# Three translations, each of which silently lost data before:
# the child's `metrics` kind is the coordinator's `metric`; the child's fields
# sit at the top level where the coordinator reads a `payload` object; and the
# child stamps `at` as an epoch float, which Postgres cannot coalesce into a
# timestamp column -- the events route answered 500 for every batch.
def coordinator_event(event: Mapping[str, Any]) -> dict[str, Any] | None:

    # The coordinator keys events on (attempt_id, seq), so an event with no
    # usable seq cannot be sent at all. Dropped rather than given an invented
    # one, which would collide with a real event and be silently discarded.
    seq = event.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        return None

    # `metrics` is what the child emits; `metric` is what the coordinator takes.
    kind = str(event.get("kind", "log"))
    if kind == "metrics":
        kind = "metric"

    # Anything else becomes a log line rather than being dropped: an unknown
    # kind would be refused, and one refused event fails the whole batch.
    if kind not in _WORKER_EVENT_KINDS:
        kind = "log"

    # Everything that is not envelope becomes the payload.
    payload = {key: value for key, value in event.items() if key not in ("seq", "kind", "at")}

    # The original kind is worth keeping when it was rewritten above.
    if kind != event.get("kind"):
        payload["child_kind"] = event.get("kind")

    return {"seq": seq, "kind": kind, "at": _iso_timestamp(event.get("at")), "payload": payload}


# _iso_timestamp()
# Converts the child's epoch stamp into an ISO-8601 string.
# Inputs: epoch seconds as a float, or anything else.
# Output: ISO-8601 UTC string, or None when there is nothing to convert.
# Use this rather than sending the float: the coordinator writes `at` into a
# timestamp column and a number there is a database error, not a coercion.
def _iso_timestamp(value: Any) -> str | None:

    # A missing stamp is fine; the coordinator falls back to its own clock.
    if value is None:
        return None

    # A string is assumed to be ISO-8601 already.
    if isinstance(value, str):
        return value

    # Epoch seconds, as the child's context writes them.
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


# CoordinatorClient
# The worker's only channel to MARP.
# Holds the base url and the one service token that configures this worker (R2),
# and nothing else -- no per-job credentials, no scoped media tokens.
class CoordinatorClient:

    # __init__()
    # Builds a client against one coordinator and one service token.
    # Inputs: coordinator base url, service token, and request timeouts.
    # Output: initialized CoordinatorClient.
    # Use this once at worker start; it is safe to keep for the process lifetime.
    def __init__(
        self,
        base_url: str,
        service_token: str,
        request_timeout_s: float = 30.0,
        long_poll_timeout_s: float = 90.0,
    ) -> None:

        # Trailing slashes would double up when joined with the route prefix.
        self._base_url = base_url.rstrip("/")

        # The single secret that configures this worker.
        self._service_token = service_token

        # Ordinary calls get a short timeout; long-poll needs a longer one.
        self._request_timeout_s = request_timeout_s
        self._long_poll_timeout_s = long_poll_timeout_s

        # One connection pool for the process, so keep-alive actually helps.
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._service_token}"},
            timeout=request_timeout_s,
        )

    # close()
    # Releases the connection pool.
    # Inputs: none.
    # Output: none.
    # Use this at worker shutdown so sockets do not outlive the process.
    def close(self) -> None:

        # Idempotent in httpx, so shutdown paths may call it twice.
        self._client.close()

    # _post()
    # Posts one JSON body and returns the decoded response.
    # Inputs: route path, JSON body, and optional timeout override.
    # Output: decoded JSON body, or None for a 204.
    # Use this so error translation happens in exactly one place.
    def _post(
        self,
        path: str,
        body: dict[str, Any],
        timeout_s: float | None = None,
    ) -> dict[str, Any] | None:

        # Send the call with the route prefix applied.
        #
        # A transport failure is translated into CoordinatorError rather than
        # allowed out as an httpx exception. It is the same event as an error
        # status as far as the runner is concerned -- the coordinator could not
        # be talked to -- and the runner's loop only catches CoordinatorError.
        # Found live: restarting the coordinator during a long poll raised
        # httpx.RemoteProtocolError straight through the loop, killed the
        # runner's thread outright, and left the worker reporting itself idle
        # with no error at all. On a home connection that is a worker that goes
        # dead the first time its link drops and never comes back.
        try:
            response = self._client.post(
                f"{_API_PREFIX}{path}",
                json=body,
                timeout=timeout_s if timeout_s is not None else self._request_timeout_s,
            )
        except httpx.HTTPError as error:
            raise CoordinatorError(f"{path} could not be reached: {type(error).__name__}: {error}")

        # 204 is the coordinator saying "nothing for you", not an error.
        if response.status_code == 204:
            return None

        # Anything else non-2xx is a failure the caller has to know about.
        if response.status_code >= 300:
            raise CoordinatorError(
                f"{path} returned {response.status_code}: {response.text[:400]}"
            )

        # An empty 2xx body is treated as an empty mapping rather than a crash.
        if not response.content:
            return {}

        return response.json()

    # enrol()
    # Registers this worker and its discovered hardware with the coordinator.
    # Inputs: local worker id and the capabilities snapshot.
    # Output: response body carrying the coordinator's worker id.
    # Use this at start, and again whenever the coordinator says it does not
    # know this worker -- which is how a re-enrolment happens (R3).
    def enrol(
        self,
        local_id: str,
        capabilities: dict[str, Any],
        name: str,
        slot_count: int,
        worker_version: str | None = None,
    ) -> dict[str, Any]:

        # Hardware is discovered and reported, never configured (R2).
        #
        # `name` is required by the coordinator and is what it keys a
        # re-enrolment on, so it has to be stable across restarts AND unique
        # across machines. A bare hostname is stable but collides -- two
        # volunteers both called DESKTOP-1 would share one pool entry and
        # silently steal each other's leases -- so the durable local id is
        # folded in. First discovered by running the two halves against each
        # other: the worker had been sending only `local_id` and enrolment
        # answered 400.
        body = {
            "local_id": local_id,
            "name": name,
            "slot_count": slot_count,
            "capabilities": capabilities,
        }
        if worker_version:
            body["worker_version"] = worker_version

        # Enrolment must return a body; an empty one is a contract violation.
        result = self._post("/workers/enrol", body)
        if result is None:
            raise CoordinatorError("enrol returned 204 with no worker record")
        return result

    # poll()
    # Long-polls for one job.
    # Inputs: worker id, how many slots are free, and the engines available here.
    # Output: attempt envelope mapping, or None when the coordinator had nothing.
    # Use this as the worker's only way of getting work (R4). The coordinator
    # holds the request open and answers 204 when it times out idle, which the
    # runner treats as a cue to poll again rather than as a failure.
    def poll(
        self,
        worker_id: str,
        free_slot_indexes: list[int],
        capabilities: dict[str, Any] | None = None,
        wait_seconds: int = 0,
    ) -> dict[str, Any] | None:

        # Say what this worker can take right now, so the coordinator can match.
        #
        # `slot_indexes` rather than a count: the coordinator records the first
        # one on the attempt so a pool view can say which GPU is busy, and a
        # count cannot say which. `wait_seconds` is what makes this a long poll
        # at all -- omitting it had the coordinator answer immediately every
        # time, so R4's long poll was never actually exercised. The coordinator
        # caps it. `capabilities` is snapshotted onto whatever attempt this poll
        # opens, which is the only record of what the machine was when it ran.
        body: dict[str, Any] = {
            "worker_id": coordinator_id(worker_id, "worker_id"),
            "slot_indexes": free_slot_indexes,
            "wait_seconds": int(wait_seconds),
        }
        if capabilities is not None:
            body["capabilities"] = capabilities

        # A little headroom over the server's hold, so the server times out first.
        return self._post("/poll", body, timeout_s=self._long_poll_timeout_s + 10.0)

    # heartbeat()
    # Holds a lease and collects the coordinator's instruction for that job.
    # Inputs: attempt envelope fields and the current progress snapshot.
    # Output: response mapping carrying `action`.
    # Use this on a fixed interval while a job runs. This is the only place
    # cancel, pause and abandon can reach the worker (R5).
    def heartbeat(
        self,
        attempt_id: str,
        worker_id: str,
        lease_epoch: int,
        progress: dict[str, Any],
        state: str = "running",
    ) -> dict[str, Any]:

        # All three lease fields go on every state-changing call (A1).
        #
        # `state` is sent because the coordinator leaves an attempt in
        # `assigned` until a worker says otherwise, so without it a job that ran
        # for an hour still reads as merely assigned and nothing can tell a
        # working machine from one that leased and never started.
        body = {
            "attempt_id": attempt_id,
            "worker_id": coordinator_id(worker_id, "worker_id"),
            "lease_epoch": lease_epoch,
            "state": state,
            "progress": progress,
        }

        # A 204 here means the coordinator has nothing to add, so continue.
        result = self._post(f"/attempts/{attempt_id}/heartbeat", body)
        return result if result is not None else {"action": "continue"}

    # post_events()
    # Sends a batch of log and metric events for one attempt.
    # Inputs: attempt envelope fields and the event list, each keyed by `seq`.
    # Output: none.
    # Use this so per-line logging does not become per-line HTTP. `seq` lets the
    # coordinator drop duplicates when a retry resends a batch.
    def post_events(
        self,
        attempt_id: str,
        worker_id: str,
        lease_epoch: int,
        events: list[dict[str, Any]],
    ) -> None:

        # Translate to the coordinator's shape, dropping anything it could not
        # key. One unacceptable event fails the whole batch, so this filters
        # rather than hoping.
        shaped = [
            shaped_event
            for shaped_event in (coordinator_event(event) for event in events)
            if shaped_event is not None
        ]

        # Nothing to send is not an error; skip the round trip.
        if not shaped:
            return

        # Batched, and keyed by seq so replay is safe. Split at the
        # coordinator's own batch limit: a job reporting a metric per frame
        # passes it within seconds and the whole batch would be refused.
        for start in range(0, len(shaped), _MAX_EVENTS_PER_BATCH):
            self._post(
                f"/attempts/{attempt_id}/events",
                {
                    "attempt_id": attempt_id,
                    "worker_id": coordinator_id(worker_id, "worker_id"),
                    "lease_epoch": lease_epoch,
                    "events": shaped[start:start + _MAX_EVENTS_PER_BATCH],
                },
            )

    # check_artifact()
    # Offers an artifact by hash and learns whether it needs uploading.
    # Inputs: attempt envelope fields, the sha256, size and role of the file.
    # Output: mapping answering `already_have`, or naming an upload target.
    # Use this before any upload. Per-frame detections never travel inline; the
    # coordinator is handed a hash and asks for the bytes only if it wants them (R11).
    def check_artifact(
        self,
        sha256: str,
        size_bytes: int,
    ) -> dict[str, Any]:

        # The check route takes only this small descriptor and never the file
        # itself. `bytes` is the coordinator's field name; the worker used to
        # send `size_bytes`, which arrived as nothing and left the staging row's
        # size unknown until the upload itself measured it.
        result = self._post(
            "/artifacts/check",
            {"sha256": sha256, "bytes": size_bytes},
        )
        return result if result is not None else {}

    # upload_artifact()
    # Sends the artifact bytes to the target the check call named.
    # Inputs: the upload path the check answered with, the local file, and the
    # attempt handing it over.
    # Output: none.
    # Use this only when check_artifact() did not answer `already_have`.
    #
    # The check answers with a plain `upload_url` path and the route is a POST.
    # The worker had been looking for an `upload` object carrying `url`,
    # `method` and `headers`, found nothing, and reported every result as
    # "coordinator neither had the artifact nor named an upload target" -- so no
    # artifact had ever actually been handed over.
    def upload_artifact(self, upload_url: str, path: Path, attempt_id: str | None = None) -> None:

        # Nothing to upload to is a failure, not a silent skip.
        if not upload_url:
            raise CoordinatorError("upload target carried no url")

        # The attempt rides in the query string as provenance, which is where
        # the upload route reads it from.
        url = upload_url
        if attempt_id:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}attempt_id={coordinator_id(attempt_id, 'attempt_id')}"

        # Stream from disk so a large results file is never held in memory.
        # Transport failures are translated here too, for the same reason.
        try:
            with path.open("rb") as file_handle:
                response = self._client.request(
                    "POST",
                    url,
                    content=file_handle,
                    # Recorded against the staged bytes and handed back later,
                    # so a reader knows what it is looking at rather than
                    # guessing.
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self._long_poll_timeout_s,
                )
        except httpx.HTTPError as error:
            raise CoordinatorError(
                f"artifact upload could not be delivered: {type(error).__name__}: {error}"
            )

        # A failed upload must not be mistaken for a delivered result.
        if response.status_code >= 300:
            raise CoordinatorError(
                f"artifact upload returned {response.status_code}: {response.text[:400]}"
            )

    # report_result()
    # Reports the attempt's terminal outcome.
    # Inputs: attempt envelope fields, the outcome name, the artifacts handed
    # over for it, and why it failed if it did.
    # Output: the coordinator's ack, which says whether this attempt published.
    # Use this exactly once per attempt. The route is idempotent, so a retry
    # after a network failure is safe and is the right response (A1).
    def report_result(
        self,
        attempt_id: str,
        worker_id: str,
        lease_epoch: int,
        outcome: str,
        artifacts: list[dict[str, Any]] | None = None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:

        # Outcome is the worker's verdict: succeeded, failed or cancelled.
        #
        # `artifacts` is what actually ties a result to its bytes. The worker
        # had been putting them inside a `result` object the coordinator does
        # not read, so every job succeeded with nothing recorded against it --
        # the results file was staged and then orphaned. Each entry must already
        # be staged or the report is refused, which is why the hand-off runs
        # first.
        body: dict[str, Any] = {
            "attempt_id": attempt_id,
            "worker_id": coordinator_id(worker_id, "worker_id"),
            "lease_epoch": lease_epoch,
            "outcome": outcome,
            "artifacts": artifacts or [],
        }

        # Only for a failure or a cancellation; a success has nothing to explain.
        if failure_reason:
            body["failure_reason"] = failure_reason

        result = self._post(f"/attempts/{attempt_id}/result", body)
        return result if result is not None else {}
