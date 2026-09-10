# context.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Job context implementation for the MARP Inference Worker.
# This is the object handed to an engine as `ctx`. It lives in the job's child
# process and turns the engine's six calls into newline-delimited JSON on stdout,
# which the parent reads and folds into heartbeats and event batches.
# The engine cannot see the coordinator, the token or the attempt from here, and
# that is the point: this file is the whole boundary.

# hashlib computes the sha256 an artifact is handed over by.
import hashlib

# json encodes one event per line.
import json

# sys gives the stdout stream the events are written to.
import sys

# time stamps events so the coordinator can order them independently of arrival.
import time

# Path types the checkpoint directory and published artifacts.
from pathlib import Path

# Any and Mapping type the free-form params and metrics.
from typing import Any, Mapping


# How long a should_stop() answer is trusted before the stop file is checked again.
# Engines call should_stop() once per frame, so an uncached check would be a
# filesystem stat per frame; a quarter second is far inside one heartbeat.
_STOP_CACHE_SECONDS = 0.25


# How many bytes are read at a time when hashing an artifact.
# Results files can be large, so hashing streams rather than loading the file.
_HASH_CHUNK_BYTES = 1024 * 1024


# hash_file()
# Computes the sha256 of a file without loading it into memory.
# Inputs: path to the file.
# Output: lower-case hex digest.
# Use this wherever an artifact or model has to be identified or verified.
def hash_file(path: Path) -> str:

    # Stream the file so a multi-gigabyte artifact costs one chunk of memory.
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# ChildJobContext
# The JobContext an engine sees inside a job's child process.
# Everything it emits goes to one stream, so the parent is the only thing that
# ever talks to MARP and the engine stays transport-free.
class ChildJobContext:

    # __init__()
    # Builds a context around one job's workspace and one output stream.
    # Inputs: params mapping, workspace directory, stop-file path, resume path,
    # and the stream events are written to.
    # Output: initialized ChildJobContext.
    # Use this from the child entry point; tests substitute their own stream.
    def __init__(
        self,
        params: Mapping[str, Any],
        checkpoint_dir: Path,
        stop_file: Path,
        resume_from: Path | None = None,
        stream: Any = None,
    ) -> None:

        # The engine's own settings, straight from the job spec.
        self._params = dict(params)

        # Private writable directory for this attempt's outputs.
        self._checkpoint_dir = checkpoint_dir

        # The parent creates this file to ask the job to stop.
        self._stop_file = stop_file

        # Milestone 1 always passes None; training is what will use it.
        self._resume_from = resume_from

        # Default to stdout, which the parent is reading line by line.
        self._stream = stream if stream is not None else sys.stdout

        # Monotonically increasing event key, so the coordinator can deduplicate
        # a batch that gets resent after a network failure.
        self._next_seq = 0

        # Cached stop answer plus when it was taken, to keep should_stop() cheap.
        self._stop_cached = False
        self._stop_checked_at = 0.0

        # The workspace has to exist before an engine writes into it.
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # params
    # The job's free-form engine parameters.
    @property
    def params(self) -> Mapping[str, Any]:

        # Returned as-is; engines ignore keys they do not recognize.
        return self._params

    # checkpoint_dir
    # This attempt's private writable directory.
    @property
    def checkpoint_dir(self) -> Path:

        # Created in __init__, so an engine can write immediately.
        return self._checkpoint_dir

    # resume_from
    # A checkpoint to continue from, or None.
    @property
    def resume_from(self) -> Path | None:

        # Always None in Milestone 1: a job runs on one worker start to finish.
        return self._resume_from

    # _emit()
    # Writes one event to the stream and flushes it.
    # Inputs: event kind and its fields.
    # Output: the seq assigned to that event.
    # Use this for every outbound engine call. The flush matters: without it the
    # parent sees nothing until the child's buffer fills, which for a slow job
    # means progress appears to stall.
    def _emit(self, kind: str, fields: dict[str, Any]) -> int:

        # Assign the sequence number before writing so gaps are detectable.
        seq = self._next_seq
        self._next_seq += 1

        # One event per line, so the parent can read without framing logic.
        event = {"seq": seq, "kind": kind, "at": time.time(), **fields}
        self._stream.write(json.dumps(event) + "\n")
        self._stream.flush()
        return seq

    # log()
    # Records one line of engine narrative.
    # Inputs: message text and severity level.
    # Output: none.
    def log(self, message: str, level: str = "info") -> None:

        # Kept as an event rather than a print, so it reaches the coordinator.
        self._emit("log", {"level": level, "message": message})

    # report_progress()
    # Reports how far through the work the engine is.
    # Inputs: units done, total units, and the unit name.
    # Output: none.
    def report_progress(self, done: int, total: int | None, unit: str) -> None:

        # The parent keeps only the latest of these and sends it on heartbeat.
        self._emit("progress", {"done": done, "total": total, "unit": unit})

    # report_metrics()
    # Reports engine metrics for one step of one phase.
    # Inputs: step number, phase name, and a free-form metric mapping.
    # Output: none.
    def report_metrics(self, step: int, phase: str, metrics: Mapping[str, Any]) -> None:

        # The mapping is passed through unschema'd on purpose: Ultralytics'
        # metric names vary by task and version, so fixing them here would be
        # wrong within a release.
        self._emit("metrics", {"step": step, "phase": phase, "metrics": dict(metrics)})

    # publish_artifact()
    # Hands a finished file to the parent and returns its hash.
    # Inputs: local path and a role name such as "results".
    # Output: the sha256 of the file.
    # Use this for every result. The hash is computed here, in the process that
    # wrote the file, so it describes exactly the bytes on disk.
    def publish_artifact(self, path: Path, kind: str) -> str:

        # A missing file is an engine bug and must not be reported as a result.
        if not path.is_file():
            raise FileNotFoundError(f"published artifact does not exist: {path}")

        # Hash and measure, then tell the parent where it is. The role is sent
        # as `role`, not `kind`: _emit() merges these fields over its own
        # envelope, so a field called `kind` would overwrite the event type and
        # the parent could no longer tell an artifact event from a log one.
        sha256 = hash_file(path)
        self._emit(
            "artifact",
            {
                "role": kind,
                "path": str(path),
                "sha256": sha256,
                "size_bytes": path.stat().st_size,
            },
        )
        return sha256

    # emit_terminal()
    # Writes the job's final event.
    # Inputs: outcome name and the outcome payload.
    # Output: none.
    # Use this once, from the child entry point, on every exit path. It shares
    # the context's stream and sequence numbering so the terminal event is
    # ordered against the progress that came before it.
    def emit_terminal(self, outcome: str, payload: dict[str, Any]) -> None:

        # The parent reads this rather than inferring an outcome from the exit
        # code, which cannot carry the reason a job was refused.
        self._emit("terminal", {"outcome": outcome, "payload": payload})

    # should_stop()
    # True once the parent has asked this job to stop.
    # Inputs: none.
    # Output: True when the engine should unwind.
    # Use this inside the engine's work loop. A stop file is used rather than a
    # signal because it is the same on Windows and POSIX and cannot interrupt a
    # decode or a CUDA call part way through.
    def should_stop(self) -> bool:

        # Once true it stays true; no need to keep checking.
        if self._stop_cached:
            return True

        # Rate-limit the filesystem check so a per-frame call stays free.
        now = time.monotonic()
        if now - self._stop_checked_at < _STOP_CACHE_SECONDS:
            return False
        self._stop_checked_at = now

        # The parent creates the file; its existence is the whole signal.
        self._stop_cached = self._stop_file.exists()
        return self._stop_cached
