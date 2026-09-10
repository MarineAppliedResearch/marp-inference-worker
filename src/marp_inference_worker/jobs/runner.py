# runner.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Job runner for the MARP Inference Worker.
# This is the loop the worker actually is: enrol, long-poll for work, lease a
# job, launch it in a child process pinned to a slot, heartbeat while it runs,
# act on what the heartbeat says, and report the terminal outcome. Nothing in
# here listens; every call is outbound (R1).
#
# It also owns the two things above a single job: how many jobs this host runs at
# once, and what happens to a job that was in flight when the worker restarted.
#
# Coordinator HTTP shape belongs in coordinator_client.py, and process handling
# in job_process.py. What belongs here is the decisions.

# json reads and writes the in-flight record used to report a lost job.
import json
import socket

# time paces the loop and the heartbeat interval.
import time

from .worker_state import worker_version

# Path types the state and workspace directories.
from pathlib import Path

# Any types the specs and payloads.
from typing import Any

# The engine registry supplies the engine list and resolves a model's needs.
from marp_inference_worker.engines import engine_registry

# The coordinator transport, and its error type.
from marp_inference_worker.jobs.coordinator_client import CoordinatorClient, CoordinatorError

# Worker identity, which has to survive a restart.
from marp_inference_worker.jobs.identity import (
    WorkerIdentity,
    load_or_create_identity,
    save_identity,
)

# One running job.
from marp_inference_worker.jobs.job_process import JobProcess

# The job spec schema, and how a heartbeat action is read.
from marp_inference_worker.jobs.job_spec import AttemptEnvelope, is_stop_action

# Model caching, which verifies the artifact hash before use.
from marp_inference_worker.models import model_cache
from marp_inference_worker.models.model_spec import ArtifactSpec, ModelSpec

# Hardware discovery, so capabilities are discovered and not configured.
from marp_inference_worker.system import device as device_module
from marp_inference_worker.system.resource_monitor import get_system_resources

# The reduction registry, reported in capabilities.
from marp_inference_worker.reduction import keyframes as keyframe_reduction


# How often a lease is held by heartbeat, in seconds.
# Cancel, pause and abandon arrive in the response, so this interval is also the
# worst-case delay before the worker acts on one (R5).
_HEARTBEAT_INTERVAL_S = 10.0


# How long to wait between polls when the long-poll returned nothing.
# The long-poll is the primary mechanism; this is the fallback for a coordinator
# that answers immediately, and keeps an idle worker from spinning (R4).
_POLL_FALLBACK_INTERVAL_S = 5.0


# How long to ask the coordinator to hold an idle poll open, in seconds.
# The coordinator caps this at its own ceiling, so asking for more is harmless.
# Only used when nothing is running -- see _poll_once().
_LONG_POLL_WAIT_S = 60.0


# How long to wait after a coordinator error before trying again.
# Longer than the fallback, because the usual cause is the coordinator being
# down or the home connection being out, and hammering it helps nobody.
_ERROR_BACKOFF_S = 30.0


# How long a stop request is given before the child is killed.
# A cooperative stop lands within one should_stop() check, so this is generous;
# a child still running after it is wedged, not slow.
_STOP_GRACE_S = 60.0


# _failure_reason()
# Builds the one-line explanation the coordinator records against an attempt.
# Inputs: the outcome name and the child's terminal payload.
# Output: a short string, or None for a success.
# Use this at the terminal report. `failure_reason` is a single text field and
# it is the only part of a failure the coordinator keeps on the attempt row, so
# it has to carry the reason rather than a shrug -- the full payload goes to the
# event stream alongside it.
def _failure_reason(outcome: str, payload: dict[str, Any]) -> str | None:

    # A success has nothing to explain.
    if outcome == "succeeded":
        return None

    # Prefer what the engine actually said.
    parts = [str(payload[key]) for key in ("reason", "message") if payload.get(key)]

    # A cancellation usually has no message: it stopped because it was asked to.
    if not parts:
        parts = [f"the worker reported {outcome}"]

    return " | ".join(parts)[:2000]


# JobRunner
# The worker's outbound loop.
# One instance per worker process. It is single-threaded on purpose: the only
# concurrency is one reader thread per running job, inside JobProcess.
class JobRunner:

    # __init__()
    # Builds a runner against one coordinator, with its state on local disk.
    # Inputs: the coordinator client, the state directory, the number of job
    # slots, and the shared worker state object the API reads.
    # Output: initialized JobRunner.
    # Use this once at worker start. `slot_count` defaults to the number of
    # CUDA devices found, so a host's capacity is discovered (R2, A4).
    def __init__(
        self,
        client: CoordinatorClient,
        state_dir: Path,
        worker_state: Any,
        slot_count: int | None = None,
    ) -> None:

        # The only channel to MARP.
        self._client = client

        # Where identity, workspaces and the in-flight record live.
        self._state_dir = state_dir
        self._workspace_root = state_dir / "jobs"
        self._inflight_path = state_dir / "in-flight.json"

        # The object the FastAPI app reads for /status and writes for pause.
        self._state = worker_state

        # One job per GPU slot. A machine with no GPU still gets one slot, so
        # it can run a CPU job that explicitly asked for CPU.
        discovered = device_module.cuda_device_count()
        self._slot_count = slot_count if slot_count is not None else max(1, discovered)

        # Slot index -> the job running on it. A slot absent from this mapping
        # is free.
        self._jobs_by_slot: dict[int, JobProcess] = {}

        # Set by stop() so the loop can be shut down from outside.
        self._should_exit = False

        # This worker's durable identity.
        self.identity: WorkerIdentity | None = None

    # capabilities()
    # Describes this machine, for enrolment and for /status.
    # Inputs: none.
    # Output: JSON-safe mapping.
    # Use this rather than any configured description. Hardware, GPU count,
    # VRAM, driver and disk are all read from the machine (R2).
    def capabilities(self) -> dict[str, Any]:

        # resource_monitor already reads CPU, memory, disk, torch and NVML.
        resources = get_system_resources()

        return {
            "slots": self._slot_count,
            "cuda_device_count": device_module.cuda_device_count(),
            "cuda_devices": device_module.describe_cuda_devices(),
            "engines": engine_registry.describe_engines(),
            "reductions": keyframe_reduction.available_reductions(),
            # The full hardware snapshot, so MARP sees the real machine.
            "resources": resources,
        }

    # ensure_enrolled()
    # Makes sure this worker has a coordinator-assigned id.
    # Inputs: none.
    # Output: none.
    # Use this at start, and again when a call is answered "unknown worker".
    # Re-enrolling with the same local_id is how a worker whose record the
    # coordinator has lost gets back in without being reconfigured (R3).
    def ensure_enrolled(self, force: bool = False) -> None:

        # Load or mint the durable identity first.
        identity_path = self._state_dir / "worker-identity.json"
        if self.identity is None:
            self.identity = load_or_create_identity(identity_path)

        # Nothing to do when already enrolled and not being forced.
        if self.identity.worker_id and not force:
            return

        # Enrol with the discovered hardware.
        #
        # Hostname plus a slice of the durable local id: readable in the pool
        # view, stable across restarts, and unique between two machines that
        # happen to share a hostname.
        display_name = f"{socket.gethostname()}-{self.identity.local_id[:8]}"
        record = self._client.enrol(
            self.identity.local_id,
            self.capabilities(),
            name=display_name,
            slot_count=self._slot_count,
            worker_version=worker_version(),
        )

        # Persist the assigned id, so a restart does not enrol again.
        self.identity.worker_id = str(record["worker_id"])
        save_identity(identity_path, self.identity)
        self._state.set_identity(self.identity.local_id, self.identity.worker_id)

    # report_lost_job()
    # Tells the coordinator about a job that was running when the worker died.
    # Inputs: none.
    # Output: none.
    # Use this once, at start, before polling for new work.
    #
    # This is what makes a restart survivable (R15). The in-flight record is
    # written before a job starts and removed when it finishes, so a record
    # still present at start means the worker died mid-job. It is reported as
    # `lost` -- not silently forgotten -- so MARP can re-queue it. Silent loss
    # was the alternative and it is worse: the job sits leased until its lease
    # expires and nobody knows why nothing happened.
    def report_lost_job(self) -> None:

        # No record means the last shutdown was clean.
        if not self._inflight_path.is_file():
            return

        # A malformed record is still evidence something was running, but there
        # is nothing to report it against, so remove it and say so.
        try:
            record = json.loads(self._inflight_path.read_text(encoding="utf-8"))
        except Exception:
            self._inflight_path.unlink(missing_ok=True)
            return

        # Report each attempt the record knew about.
        for attempt in record.get("attempts", []):
            try:
                # Reported as `failed`, with the loss named in the reason.
                #
                # `lost` is the worker's own word for it and the coordinator has
                # no such outcome -- its vocabulary is succeeded, failed and
                # cancelled -- so sending `lost` was answered 400 and the job
                # was left leased to a machine that no longer existed until the
                # lease timed out. `failed` is what puts it back in the queue
                # while attempts remain, which is what R15 needs; that it reads
                # as a failure rather than a loss is a presentation question and
                # is called out in the report.
                self._client.report_result(
                    attempt_id=str(attempt["attempt_id"]),
                    worker_id=str(attempt["worker_id"]),
                    lease_epoch=int(attempt["lease_epoch"]),
                    outcome="failed",
                    failure_reason=(
                        "Lost: the worker restarted while this job was running. "
                        f"Last known progress: {attempt.get('progress')}"
                    ),
                )
            except CoordinatorError:
                # A coordinator that will not take the report must not stop the
                # worker from coming back up; the lease will expire instead.
                pass

        # Clear it either way, so the same job is not reported lost forever.
        self._inflight_path.unlink(missing_ok=True)

    # _write_inflight()
    # Records which attempts are currently running.
    # Inputs: none.
    # Output: none.
    # Use this whenever a job starts or finishes. Written before the child is
    # launched, so a crash in between still leaves the record.
    def _write_inflight(self) -> None:

        # An empty set of jobs means no record at all, which is what a clean
        # idle worker should leave behind.
        if not self._jobs_by_slot:
            self._inflight_path.unlink(missing_ok=True)
            return

        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._inflight_path.write_text(
            json.dumps(
                {
                    "attempts": [
                        {
                            "attempt_id": job.attempt_id,
                            "worker_id": job.worker_id,
                            "lease_epoch": job.lease_epoch,
                            "slot_index": job.slot_index,
                            "progress": job.current_progress(),
                        }
                        for job in self._jobs_by_slot.values()
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # free_slots()
    # How many job slots are currently free.
    # Inputs: none.
    # Output: count of free slots.
    def free_slots(self) -> int:

        # A paused worker reports no free slots, which is how pause stops it
        # being offered work without needing a separate coordinator concept.
        if self._state.is_paused():
            return 0
        return self._slot_count - len(self._jobs_by_slot)

    # _next_free_slot()
    # Returns the lowest free slot index.
    # Inputs: none.
    # Output: slot index, or None when all slots are busy.
    # Use this to pin a job. Lowest-first means slot 0 maps to cuda:0, which
    # makes a running job easy to find in nvidia-smi.
    def _next_free_slot(self) -> int | None:

        for slot_index in range(self._slot_count):
            if slot_index not in self._jobs_by_slot:
                return slot_index
        return None

    # _free_slot_indexes()
    # Which slot indexes are free right now.
    # Inputs: none.
    # Output: list of free slot indexes, lowest first.
    # Use this for the poll. The coordinator records the first of them on the
    # attempt, so a pool view can say which GPU a job is on -- a bare count
    # cannot, which is why the poll names the slots rather than counting them.
    def _free_slot_indexes(self) -> list[int]:

        # A paused worker has nothing free, however many slots it has.
        if self._state.is_paused():
            return []
        return [index for index in range(self._slot_count) if index not in self._jobs_by_slot]

    # stop()
    # Asks the loop to finish after the current iteration.
    # Inputs: none.
    # Output: none.
    # Use this from a signal handler or the API's shutdown path.
    def stop(self) -> None:

        self._should_exit = True

    # run_forever()
    # The worker's main loop.
    # Inputs: none.
    # Output: none.
    # Use this as the worker's job-taking process. It enrols, reports any lost
    # job, then alternates between servicing running jobs and polling for work.
    def run_forever(self) -> None:

        # Identity and enrolment first; nothing else can happen without them.
        # Enrol on every start, not only the first, so a machine that gained a
        # GPU, changed driver or installed an engine re-reports its hardware.
        # Enrolment is idempotent on the coordinator side (keyed on name), so
        # this refreshes the pool row rather than creating a second one. Found
        # live: this machine's pool entry still read "no GPU" after CUDA was
        # working, because capabilities had only ever been sent once.
        self.ensure_enrolled(force=True)

        # Report a job that did not survive the last restart, before taking new
        # work, so MARP can re-queue it promptly.
        self.report_lost_job()

        # The loop. Each pass services what is running and then, if there is
        # room, asks for one more job.
        while not self._should_exit:
            try:
                self._service_running_jobs()

                # Only ask for work when there is somewhere to put it.
                if self.free_slots() > 0:
                    took_work = self._poll_once()

                    # Nothing offered: wait before asking again. This is the
                    # interval-polling fallback behind the long poll (R4).
                    if not took_work:
                        time.sleep(_POLL_FALLBACK_INTERVAL_S)
                else:
                    # Full or paused: wait a heartbeat interval and service again.
                    time.sleep(_HEARTBEAT_INTERVAL_S)

            except CoordinatorError as error:
                # The coordinator is unreachable or unhappy. Back off and keep
                # the running jobs going; they do not depend on it moment to
                # moment, only on the lease not being reassigned.
                self._state.note_error(str(error))
                time.sleep(_ERROR_BACKOFF_S)

            except Exception as error:
                # Anything else, recorded and survived rather than allowed to
                # end the loop.
                #
                # This thread *is* the worker, and an exception out of it leaves
                # a process that serves /status, says it is idle, and will never
                # take work again -- which is exactly what happened when a
                # transport error escaped the clause above. A worker that keeps
                # failing loudly can be diagnosed; one that quietly stops
                # cannot.
                self._state.note_error(f"{type(error).__name__}: {error}")
                time.sleep(_ERROR_BACKOFF_S)

    # _poll_once()
    # Long-polls for one job and starts it if one was offered.
    # Inputs: none.
    # Output: True when a job was taken.
    # Use this from the loop. A 204 from the coordinator is not an error, it is
    # the long poll timing out with nothing queued.
    def _poll_once(self) -> bool:

        # Identity is set by ensure_enrolled() before the loop starts.
        assert self.identity is not None and self.identity.worker_id is not None

        # Hold the request open only while nothing is running. A long poll with
        # a job in flight delays the next heartbeat by however long it waits,
        # which both breaks R5's "within one heartbeat" and can outlast the
        # lease itself -- the coordinator would take the job away from a worker
        # that was busy doing it.
        wait_seconds = 0 if self._jobs_by_slot else int(_LONG_POLL_WAIT_S)

        # Say what this worker can take right now.
        offered = self._client.poll(
            worker_id=self.identity.worker_id,
            free_slot_indexes=self._free_slot_indexes(),
            capabilities=self.capabilities(),
            wait_seconds=wait_seconds,
        )

        # Nothing queued.
        if not offered:
            return False

        # The lease carries no worker id -- it is the answer to this worker's
        # own poll, so the only worker it could belong to is this one. Filled in
        # here so the envelope that every later call quotes is complete.
        offered = dict(offered)
        offered.setdefault("worker_id", self.identity.worker_id)

        # Validate the offer before acting on it. A malformed spec is refused
        # here rather than after a child process has been launched for it.
        try:
            envelope = AttemptEnvelope.model_validate(offered)
        except Exception as error:
            # Report it against whatever identity fields did arrive, so the
            # coordinator learns its offer was rejected rather than timing out.
            attempt_id = str(offered.get("attempt_id", ""))
            if attempt_id:
                self._client.report_result(
                    attempt_id=attempt_id,
                    worker_id=str(offered.get("worker_id", self.identity.worker_id)),
                    lease_epoch=int(offered.get("lease_epoch", 0)),
                    outcome="failed",
                    failure_reason=f"invalid_spec: {error}",
                )
            return False

        # Start it.
        self._start_job(envelope)
        return True

    # _start_job()
    # Prepares and launches one job.
    # Inputs: the validated attempt envelope.
    # Output: none.
    # Use this from _poll_once(). Everything that can be checked before the
    # child process exists is checked here, so a job that cannot run says why
    # immediately rather than starting and dying (R14).
    def _start_job(self, envelope: AttemptEnvelope) -> None:

        # There is a free slot: free_slots() was checked before polling.
        slot_index = self._next_free_slot()
        if slot_index is None:
            return

        try:
            # Fetch and hash-verify the model in the parent. It can fail for
            # reasons the coordinator should hear about as a refusal rather than
            # as a crashed child.
            prepared_params = self._prepare_job_params(envelope, slot_index)

        except Exception as error:
            # Refused before any work started, with the reason attached.
            self._client.report_result(
                attempt_id=envelope.attempt_id,
                worker_id=envelope.worker_id,
                lease_epoch=envelope.lease_epoch,
                outcome="failed",
                failure_reason=f"preparation_failed: {type(error).__name__}: {error}",
            )
            return

        # Build the spec the child receives: the coordinator's spec with the
        # local model details merged into params. The engine reads these and
        # still learns nothing about MARP or the coordinator.
        spec = envelope.spec.model_dump(mode="json")
        spec["params"] = prepared_params

        # Create and launch the child.
        job = JobProcess(
            envelope={
                "attempt_id": envelope.attempt_id,
                "worker_id": envelope.worker_id,
                "lease_epoch": envelope.lease_epoch,
                "spec": spec,
            },
            slot_index=slot_index,
            workspace_root=self._workspace_root,
        )
        self._jobs_by_slot[slot_index] = job

        # Record it as in-flight before launching, so a crash between the two
        # still leaves evidence for report_lost_job() (R15).
        self._write_inflight()

        job.start()
        self._state.set_jobs(self._describe_jobs())

    # _prepare_job_params()
    # Resolves everything the engine needs that only the worker can supply.
    # Inputs: the attempt envelope and the slot index.
    # Output: the params mapping the child receives.
    #
    # One thing happens here rather than in the engine, deliberately: the model
    # is fetched and its sha256 verified, so no engine can be handed unverified
    # weights (R13). The video is not resolved anywhere in the worker -- the
    # coordinator does that and the spec arrives carrying an openable url (A8).
    def _prepare_job_params(self, envelope: AttemptEnvelope, slot_index: int) -> dict[str, Any]:

        # Start from what the coordinator sent.
        params = dict(envelope.spec.params)

        # Tell the engine which slot it is pinned to, so device resolution can
        # map it to a GPU (R6).
        params["slot_index"] = slot_index

        # Fetch the model and verify it. A job spec's model always carries a
        # sha256, so this always verifies -- unlike the frame routes, where the
        # hash is optional.
        model = envelope.spec.model
        if not model.sha256:
            raise ValueError("job spec's model carried no sha256")

        cache_state = model_cache.ensure_artifact_cached(
            ModelSpec(
                model_id=model.name,
                engine=envelope.spec.engine,
                model_arch="unknown",
                task="detect",
                artifact=ArtifactSpec(
                    url=model.url or model.name,
                    format="pt",
                    sha256=model.sha256,
                ),
            )
        )
        params["model_path"] = cache_state["artifact_path"]
        params["model_sha256"] = cache_state["sha256"]

        return params

    # _service_running_jobs()
    # Heartbeats every running job and acts on the answers.
    # Inputs: none.
    # Output: none.
    # Use this once per loop pass. This is where cancel, pause and abandon are
    # received and acted on, and where a finished job is reported (R5).
    def _service_running_jobs(self) -> None:

        # Iterate over a copy: finishing a job mutates the mapping.
        for slot_index, job in list(self._jobs_by_slot.items()):

            # A job whose child has exited is reported and its slot freed.
            if not job.is_running():
                self._finish_job(slot_index, job)
                continue

            # Send progress up and read the instruction that comes back. The
            # event batch rides along with it, so logging is not per-line HTTP.
            try:
                self._client.post_events(
                    attempt_id=job.attempt_id,
                    worker_id=job.worker_id,
                    lease_epoch=job.lease_epoch,
                    events=job.take_events(),
                )
                response = self._client.heartbeat(
                    attempt_id=job.attempt_id,
                    worker_id=job.worker_id,
                    lease_epoch=job.lease_epoch,
                    progress=job.current_progress(),
                )
            except CoordinatorError as error:
                # A failed heartbeat is not a reason to kill a running job: the
                # network may be back before the lease expires, and killing it
                # throws away work that is still valid.
                self._state.note_error(f"heartbeat failed for {job.attempt_id}: {error}")
                continue

            # Read the instruction. An action this worker does not recognize is
            # treated as `continue`, so an old worker against a newer
            # coordinator keeps working rather than stopping on every job.
            action = str(response.get("action", "continue"))

            # Pause stops the worker taking new work and leaves the running job
            # alone. Abandoning a running job is the separate `abandon` action.
            if action == "pause":
                self._state.set_paused(True, reason="paused by coordinator")
                continue

            # Cancel and abandon both stop this job.
            if is_stop_action(action) and not job.stop_requested:
                self._state.note_error(f"{action} received for {job.attempt_id}")
                job.request_stop()

            # A job that was asked to stop and has not is killed once the grace
            # period is up. It then reports nothing about itself, so _finish_job
            # decides the outcome on its behalf.
            if job.stop_requested and job.current_progress()["elapsed_s"] > _STOP_GRACE_S:
                if not job.wait(timeout_s=0.1):
                    job.kill()

        # Refresh what /status reports.
        self._state.set_jobs(self._describe_jobs())

        # And refresh the in-flight record, so it carries how far each job had
        # actually got. Written once before launch it always said zero, so a job
        # lost to a restart was reported lost with no progress at all -- which
        # is most of what R15's report is for.
        if self._jobs_by_slot:
            self._write_inflight()

    # _finish_job()
    # Reports one finished job's outcome and frees its slot.
    # Inputs: the slot index and the job.
    # Output: none.
    # Use this when the child has exited. The child's own terminal event is
    # preferred over its exit code, because an exit code cannot say why a job
    # was refused -- and a child that died without sending one is reported as a
    # crash with whatever it left on stderr.
    def _finish_job(self, slot_index: int, job: JobProcess) -> None:

        # Give the reader thread a moment to finish draining, or a fast job's
        # terminal event can still be in flight.
        job.wait(timeout_s=5.0)

        # Send the last events before the terminal report, so the coordinator
        # has the log that explains the outcome.
        try:
            self._client.post_events(
                attempt_id=job.attempt_id,
                worker_id=job.worker_id,
                lease_epoch=job.lease_epoch,
                events=job.take_events(),
            )
        except CoordinatorError:
            # Losing the tail of a log must not stop the result being reported.
            pass

        # One last heartbeat, carrying the progress the job finished on.
        #
        # Progress is overwritten in place on the attempt, and the last
        # heartbeat before this point was sent up to a heartbeat interval ago --
        # so a job shorter than that interval finished with its progress still
        # reading 0 of null, which is what the first live round trip showed for
        # a 300-frame job that had demonstrably processed all 300. `uploading`
        # is also the honest state here: the work is done and the bytes are
        # about to move.
        try:
            self._client.heartbeat(
                attempt_id=job.attempt_id,
                worker_id=job.worker_id,
                lease_epoch=job.lease_epoch,
                progress=job.current_progress(),
                state="uploading",
            )
        except CoordinatorError:
            # The answer does not matter -- this job is over either way.
            pass

        terminal = job.terminal
        if terminal is not None:
            outcome = str(terminal.get("outcome", "failed"))
            payload = dict(terminal.get("payload") or {})
        else:
            # No terminal event: the child was killed, or died in a way that
            # left nothing on stdout -- an OOM kill or a native crash.
            outcome = "cancelled" if job.stop_requested else "failed"
            payload = {
                "reason": "child_exited_without_report",
                "exit_code": job.exit_code(),
                "stderr": job.collect_stderr(),
            }

        # Offer the results file by hash before reporting the outcome. The
        # coordinator answers `already_have` or names somewhere to put it, so
        # the bytes only move when they are actually wanted (R11).
        #
        # The order matters and is not a preference: the coordinator refuses a
        # result naming an artifact it has not been handed, so the hand-off has
        # to complete first. Only what actually landed is named, so a success
        # never points at bytes MARP does not hold.
        delivered: list[dict[str, Any]] = []

        for role, artifact in job.artifacts.items():
            try:
                handover = self._hand_over_artifact(job, artifact)
                payload.setdefault("artifacts", {})[role] = handover

                if handover.get("delivered"):
                    delivered.append({"sha256": str(artifact["sha256"]), "role": role})

            except CoordinatorError as error:
                # Say the artifact exists and could not be delivered, rather
                # than reporting a success with no result behind it.
                payload.setdefault("artifacts", {})[role] = {
                    "sha256": artifact.get("sha256"),
                    "delivered": False,
                    "error": str(error),
                }

        # Send the terminal payload as one last log event.
        #
        # The result route records an outcome, a failure reason and artifact
        # hashes, and nothing else -- so the engine's own summary, and a crash's
        # traceback, have nowhere to go on that call. Events are where durable
        # detail belongs (R12), and this is the only place a failure can be
        # diagnosed from afterwards.
        self._report_terminal_detail(job, outcome, payload)

        # Report the outcome. The route is idempotent, so a retry is safe.
        try:
            self._client.report_result(
                attempt_id=job.attempt_id,
                worker_id=job.worker_id,
                lease_epoch=job.lease_epoch,
                outcome=outcome,
                artifacts=delivered,
                failure_reason=_failure_reason(outcome, payload),
            )
        except CoordinatorError as error:
            self._state.note_error(f"could not report result for {job.attempt_id}: {error}")

        # Free the slot and clear the in-flight record.
        self._jobs_by_slot.pop(slot_index, None)
        self._write_inflight()
        self._state.set_jobs(self._describe_jobs())

    # _hand_over_artifact()
    # Offers one artifact by hash and uploads it only if asked.
    # Inputs: the job and the artifact event.
    # Output: mapping describing what happened.
    # Use this from _finish_job(). Two steps, not one, so a results file the
    # coordinator already has is never sent twice.
    def _hand_over_artifact(self, job: JobProcess, artifact: dict[str, Any]) -> dict[str, Any]:

        sha256 = str(artifact["sha256"])

        # Offer it.
        answer = self._client.check_artifact(
            sha256=sha256,
            size_bytes=int(artifact.get("size_bytes", 0)),
        )

        # Already stored: nothing to send.
        if answer.get("already_have"):
            return {"sha256": sha256, "delivered": True, "upload": "not_needed"}

        # Otherwise the coordinator named somewhere to put it, as a plain path.
        upload_url = answer.get("upload_url")
        if not upload_url:
            return {
                "sha256": sha256,
                "delivered": False,
                "error": "coordinator neither had the artifact nor named an upload target",
            }

        self._client.upload_artifact(
            str(upload_url),
            Path(str(artifact["path"])),
            attempt_id=job.attempt_id,
        )
        return {"sha256": sha256, "delivered": True, "upload": "sent"}

    # _report_terminal_detail()
    # Sends the engine's terminal payload as one final log event.
    # Inputs: the job, the outcome name and the terminal payload.
    # Output: none.
    # Use this immediately before the terminal report, so the detail is already
    # recorded whichever way the result call goes.
    def _report_terminal_detail(
        self,
        job: JobProcess,
        outcome: str,
        payload: dict[str, Any],
    ) -> None:

        # A sequence number past everything the child sent. The child has
        # exited by now, so nothing else will claim it -- and a colliding seq
        # would be dropped silently by the coordinator's replay guard.
        event = {
            "seq": job.next_parent_seq(),
            "kind": "log",
            # Stamped like the child's own events, so the coordinator does not
            # have to fall back to its own clock for the one event that says how
            # the attempt ended.
            "at": time.time(),
            "level": "info",
            "message": f"attempt finished as {outcome}",
            "terminal": payload,
        }

        # Anything the child printed to stdout that was not an event goes here
        # too: it has no sequence number of its own and would otherwise be lost.
        stray = job.take_stray_output()
        if stray:
            event["stray_stdout"] = stray

        try:
            self._client.post_events(
                attempt_id=job.attempt_id,
                worker_id=job.worker_id,
                lease_epoch=job.lease_epoch,
                events=[event],
            )
        except CoordinatorError:
            # Detail is worth having and not worth failing the report over.
            pass

    # _describe_jobs()
    # Describes the running jobs for /status.
    # Inputs: none.
    # Output: list of JSON-safe job mappings.
    # Use this whenever the running set changes, so /status reports what is
    # actually true (R12).
    def _describe_jobs(self) -> list[dict[str, Any]]:

        return [
            {
                "attempt_id": job.attempt_id,
                "slot_index": job.slot_index,
                "engine": job.spec.get("engine"),
                "range": job.spec.get("range"),
                "video_source": (job.spec.get("video") or {}).get("source_name"),
                "progress": job.current_progress(),
                "stop_requested": job.stop_requested,
            }
            for job in self._jobs_by_slot.values()
        ]
