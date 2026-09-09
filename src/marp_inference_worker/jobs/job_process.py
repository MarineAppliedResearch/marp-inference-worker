# job_process.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# One running job, from the parent's side, for the MARP Inference Worker.
# This file owns a single child process: launching it pinned to a slot, reading
# its events, asking it to stop, and collecting its outcome. The runner above it
# owns the coordinator conversation and the slots; this owns the process.
#
# Keeping the two apart matters because they fail differently. A child that dies
# is a job problem; a coordinator that stops answering is a worker problem, and
# the second must not be mistaken for the first.

# json parses the child's event lines and writes its envelope.
import json

# subprocess launches and supervises the child.
import subprocess

# sys names the interpreter the child is launched with.
import sys

# threading drains the child's stdout without blocking the runner's loop.
import threading

# time stamps the process' start, for the lost-job report after a restart.
import time

# Path types the workspace and envelope paths.
from pathlib import Path

# Any types the events and the spec.
from typing import Any


# JobProcess
# One job's child process and everything the parent knows about it.
# Not thread-safe by design beyond its event queue: exactly one runner thread
# drives it, and one reader thread fills it.
class JobProcess:

    # __init__()
    # Prepares a job's workspace and records what it is, without launching.
    # Inputs: the attempt envelope mapping, the slot index, and the root
    # directory job workspaces live under.
    # Output: initialized JobProcess.
    # Use start() to actually launch it.
    def __init__(
        self,
        envelope: dict[str, Any],
        slot_index: int,
        workspace_root: Path,
    ) -> None:

        # The lease this job runs under. Every coordinator call carries all three.
        self.attempt_id = str(envelope["attempt_id"])
        self.worker_id = str(envelope["worker_id"])
        self.lease_epoch = int(envelope["lease_epoch"])

        # The work itself, and the GPU slot it is pinned to (R6).
        self.spec = dict(envelope["spec"])
        self.slot_index = slot_index

        # Each attempt gets its own workspace, so two jobs cannot collide over
        # a results file, a run directory or an Ultralytics settings file (R16).
        self.workspace = workspace_root / self.attempt_id

        # The parent creates this file to ask the job to stop. A file rather
        # than a signal: identical on Windows and POSIX, and it cannot interrupt
        # a decode or a CUDA call part way through.
        self.stop_file = self.workspace / "STOP"

        # Filled in by start().
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None

        # The latest progress the child reported, sent up on each heartbeat.
        self.progress: dict[str, Any] = {"done": 0, "total": None, "unit": "frames"}

        # Events not yet forwarded to the coordinator, drained in batches.
        self._pending_events: list[dict[str, Any]] = []

        # Guards the two collections the reader thread writes to.
        self._lock = threading.Lock()

        # The artifacts the child published, keyed by role.
        self.artifacts: dict[str, dict[str, Any]] = {}

        # The child's terminal event, once it has sent one.
        self.terminal: dict[str, Any] | None = None

        # When the job started, so a heartbeat can report how long it has run.
        self.started_at = 0.0

        # Set once the parent has asked it to stop, so it is only asked once.
        self.stop_requested = False

    # start()
    # Writes the job's envelope and launches the child process.
    # Inputs: none.
    # Output: none.
    # Use this once. The child is launched with this interpreter, so it runs in
    # the same environment the worker was started in.
    def start(self) -> None:

        # Build the workspace before the child needs it.
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Hand the job over in a file. Not on the command line: a spec's params
        # can exceed the Windows command-line limit, and anything on a command
        # line is visible in a process listing.
        envelope_path = self.workspace / "envelope.json"
        envelope_path.write_text(
            json.dumps(
                {
                    "attempt_id": self.attempt_id,
                    "worker_id": self.worker_id,
                    "lease_epoch": self.lease_epoch,
                    "workspace": str(self.workspace),
                    "stop_file": str(self.stop_file),
                    "spec": self.spec,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Launch the child. stdout is the event channel; stderr is captured
        # separately so a library writing a warning cannot corrupt an event line.
        self._process = subprocess.Popen(
            [sys.executable, "-m", "marp_inference_worker.jobs.child_main", str(envelope_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.started_at = time.time()

        # Drain stdout on its own thread. Without this the child blocks once the
        # pipe buffer fills, which for a job reporting progress means it stalls
        # somewhere between one and a few thousand frames in -- a hang that looks
        # like slow inference.
        self._reader = threading.Thread(target=self._read_events, daemon=True)
        self._reader.start()

    # _read_events()
    # Reads the child's event lines until its stdout closes.
    # Inputs: none.
    # Output: none.
    # Runs on its own thread. Never raises out: a malformed line is recorded as
    # a log event rather than killing the reader and silently stopping progress.
    def _read_events(self) -> None:

        # The process is always started before the reader thread.
        assert self._process is not None and self._process.stdout is not None

        # One JSON object per line, as the child's context writes them.
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue

            # A line that is not JSON is something in the child that printed to
            # stdout directly. Keep it, as a log, rather than discarding it.
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                with self._lock:
                    self._pending_events.append(
                        {"kind": "log", "level": "warning", "message": line}
                    )
                continue

            self._handle_event(event)

    # _handle_event()
    # Folds one child event into the parent's state.
    # Inputs: the decoded event mapping.
    # Output: none.
    # Use this from the reader thread only.
    def _handle_event(self, event: dict[str, Any]) -> None:

        kind = event.get("kind")

        with self._lock:

            # Progress is kept as a latest-value, not queued: the coordinator
            # wants where the job is now, and queueing 108,000 progress events
            # for an hour of video would be pointless traffic.
            if kind == "progress":
                self.progress = {
                    "done": event.get("done", 0),
                    "total": event.get("total"),
                    "unit": event.get("unit", "frames"),
                }
                return

            # An artifact is recorded under its role, so the runner can find
            # the results file without knowing what the engine called it.
            if kind == "artifact":
                self.artifacts[str(event.get("role", "unnamed"))] = event
                return

            # The terminal event is the child's verdict on itself.
            if kind == "terminal":
                self.terminal = event
                return

            # Logs and metrics are batched up to the coordinator.
            self._pending_events.append(event)

    # take_events()
    # Removes and returns the events not yet sent to the coordinator.
    # Inputs: none.
    # Output: list of event mappings.
    # Use this from the runner before each heartbeat, so a batch goes with it.
    def take_events(self) -> list[dict[str, Any]]:

        # Swap the list out under the lock; the reader keeps filling a new one.
        with self._lock:
            events = self._pending_events
            self._pending_events = []
        return events

    # current_progress()
    # Returns the latest progress snapshot.
    # Inputs: none.
    # Output: progress mapping, with how long the job has been running added.
    # Use this as the heartbeat's progress payload.
    def current_progress(self) -> dict[str, Any]:

        # Copy under the lock; the reader thread replaces this mapping.
        with self._lock:
            progress = dict(self.progress)

        # Elapsed time lets the coordinator spot a job that is running but not
        # advancing, which progress alone cannot show.
        progress["slot_index"] = self.slot_index
        progress["elapsed_s"] = round(time.time() - self.started_at, 3)
        return progress

    # request_stop()
    # Asks the job to stop, cooperatively.
    # Inputs: none.
    # Output: none.
    # Use this when a heartbeat returns cancel or abandon. The child notices
    # within one should_stop() check, unwinds, writes what it has and reports
    # itself cancelled -- so partial work is reported rather than lost (R5).
    def request_stop(self) -> None:

        # Creating the file is the whole signal, and it is idempotent.
        self.stop_file.touch(exist_ok=True)
        self.stop_requested = True

    # is_running()
    # Whether the child process is still alive.
    # Inputs: none.
    # Output: True while the process has not exited.
    def is_running(self) -> bool:

        # A JobProcess that was never started is not running.
        if self._process is None:
            return False
        return self._process.poll() is None

    # exit_code()
    # The child's exit code, or None while it is still running.
    # Inputs: none.
    # Output: exit code or None.
    def exit_code(self) -> int | None:

        if self._process is None:
            return None
        return self._process.poll()

    # kill()
    # Terminates the child immediately.
    # Inputs: how long to wait for a terminate before killing outright.
    # Output: none.
    # Use this only after request_stop() has been given time and ignored. A job
    # killed this way reports nothing about itself, so the runner has to decide
    # its outcome on the child's behalf.
    def kill(self, grace_s: float = 10.0) -> None:

        # Nothing to kill if it never started or already exited.
        if self._process is None or self._process.poll() is not None:
            return

        # Ask first, then insist. terminate() lets the interpreter run its
        # finally blocks, which is what releases the video capture and the GPU.
        self._process.terminate()
        try:
            self._process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            self._process.kill()

    # collect_stderr()
    # Returns whatever the child wrote to stderr.
    # Inputs: none.
    # Output: the captured stderr text, truncated.
    # Use this when a child failed without sending a terminal event -- a
    # segfault or an OOM kill leaves nothing on stdout but often something here.
    def collect_stderr(self) -> str:

        if self._process is None or self._process.stderr is None:
            return ""

        # Read what is there. The process has exited by the time this is called,
        # so this does not block.
        try:
            text = self._process.stderr.read() or ""
        except Exception:
            return ""

        # Truncated, because a torch traceback with a CUDA dump is very long and
        # the coordinator only needs enough to identify the failure.
        return text[-4000:]

    # wait()
    # Waits for the child to exit and for its events to be fully drained.
    # Inputs: how long to wait.
    # Output: True when the child exited within the timeout.
    # Use this before reading the terminal event, or a fast job's terminal event
    # can still be in flight when the runner looks for it.
    def wait(self, timeout_s: float) -> bool:

        if self._process is None:
            return True

        # Wait for the process first.
        try:
            self._process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False

        # Then for the reader, so every event it read has been folded in.
        if self._reader is not None:
            self._reader.join(timeout=5.0)
        return True
