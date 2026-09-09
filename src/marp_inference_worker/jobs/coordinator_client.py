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

# Path types the artifact file offered upstream.
from pathlib import Path

# Any types the free-form payload bodies.
from typing import Any


# Route family the worker talks to, from A1 in .marp/task.md.
_API_PREFIX = "/api/v2/gpu"


# CoordinatorError
# Raised when a coordinator call fails in a way the caller must see.
# Transient network trouble is retried by the runner's loop, not here.
class CoordinatorError(RuntimeError):
    pass


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
        response = self._client.post(
            f"{_API_PREFIX}{path}",
            json=body,
            timeout=timeout_s if timeout_s is not None else self._request_timeout_s,
        )

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
        free_slots: int,
        engines: list[str],
    ) -> dict[str, Any] | None:

        # Say what this worker can take right now, so the coordinator can match.
        body = {"worker_id": worker_id, "free_slots": free_slots, "engines": engines}

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
    ) -> dict[str, Any]:

        # All three lease fields go on every state-changing call (A1).
        body = {
            "attempt_id": attempt_id,
            "worker_id": worker_id,
            "lease_epoch": lease_epoch,
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

        # Nothing to send is not an error; skip the round trip.
        if not events:
            return

        # Batched, and keyed by seq so replay is safe.
        self._post(
            f"/attempts/{attempt_id}/events",
            {
                "attempt_id": attempt_id,
                "worker_id": worker_id,
                "lease_epoch": lease_epoch,
                "events": events,
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
        attempt_id: str,
        worker_id: str,
        lease_epoch: int,
        sha256: str,
        size_bytes: int,
        kind: str,
    ) -> dict[str, Any]:

        # The check route sits outside the coordinator's body parser, so it takes
        # only this small descriptor and never the file itself.
        result = self._post(
            "/artifacts/check",
            {
                "attempt_id": attempt_id,
                "worker_id": worker_id,
                "lease_epoch": lease_epoch,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "kind": kind,
            },
        )
        return result if result is not None else {}

    # upload_artifact()
    # Sends the artifact bytes to the target the check call named.
    # Inputs: the upload target mapping and the local file path.
    # Output: none.
    # Use this only when check_artifact() did not answer `already_have`.
    def upload_artifact(self, upload_target: dict[str, Any], path: Path) -> None:

        # The coordinator names both where to put it and how.
        url = upload_target.get("url")
        if not url:
            raise CoordinatorError("upload target carried no url")

        # Stream from disk so a large results file is never held in memory.
        with path.open("rb") as file_handle:
            response = self._client.request(
                upload_target.get("method", "PUT"),
                url,
                content=file_handle,
                headers=upload_target.get("headers") or {},
                timeout=self._long_poll_timeout_s,
            )

        # A failed upload must not be mistaken for a delivered result.
        if response.status_code >= 300:
            raise CoordinatorError(
                f"artifact upload returned {response.status_code}: {response.text[:400]}"
            )

    # report_result()
    # Reports the attempt's terminal outcome.
    # Inputs: attempt envelope fields, outcome name, and the result payload.
    # Output: none.
    # Use this exactly once per attempt. The route is idempotent, so a retry
    # after a network failure is safe and is the right response (A1).
    def report_result(
        self,
        attempt_id: str,
        worker_id: str,
        lease_epoch: int,
        outcome: str,
        payload: dict[str, Any],
    ) -> None:

        # Outcome is the worker's verdict: succeeded, failed, cancelled or lost.
        self._post(
            f"/attempts/{attempt_id}/result",
            {
                "attempt_id": attempt_id,
                "worker_id": worker_id,
                "lease_epoch": lease_epoch,
                "outcome": outcome,
                "result": payload,
            },
        )
