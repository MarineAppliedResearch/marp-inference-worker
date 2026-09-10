# child_main.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Job child process entry point for the MARP Inference Worker.
# One job runs in one child process pinned to one GPU slot (R6). This file is
# that process: it reads the job it was given, sets the runtime environment for
# it, builds the context, preflights and runs the engine, then reports the
# outcome on stdout as a final event.
#
# A separate process rather than a thread, for three reasons that all cost time
# to learn the hard way: a CUDA out-of-memory or a driver fault takes the process
# down and not the worker; Ultralytics sets global state and re-execs itself for
# multi-GPU; and killing a job is only actually possible if it owns a process.
#
# This process talks to nobody. Every event goes to stdout and the parent decides
# what reaches MARP.

# json reads the envelope and writes the terminal event.
import json

# os sets the per-job Ultralytics environment variables.
import os

# sys reads the arguments and writes the final event.
import sys

# traceback captures a failure in a form worth reporting.
import traceback

# Path types the envelope and workspace paths.
from pathlib import Path

# Any types the loaded envelope.
from typing import Any


# main()
# Runs one job and reports its outcome.
# Inputs: argv[1] is the path to the job envelope JSON written by the parent.
# Output: process exit code -- 0 for success or a clean stop, 1 for a failure.
# Use this only through the parent's subprocess launch; it is not a CLI.
def main(argv: list[str]) -> int:

    # The parent always passes exactly one argument.
    if len(argv) < 2:
        sys.stderr.write("usage: child_main.py <envelope.json>\n")
        return 2

    # Read the job the parent wrote out. A file rather than an argument because
    # a spec with a long params mapping would exceed the command-line limit on
    # Windows, and because a token must never appear in a process listing.
    envelope_path = Path(argv[1])
    envelope: dict[str, Any] = json.loads(envelope_path.read_text(encoding="utf-8"))

    # The workspace this job owns, and the file the parent creates to stop it.
    workspace = Path(envelope["workspace"])
    stop_file = Path(envelope["stop_file"])
    spec = envelope["spec"]

    # Set Ultralytics' runtime environment before anything imports it. This is
    # the worker's decision and not the engine's (R16), and it has to happen
    # before the first ultralytics import because the library reads these at
    # import time and caches them.
    _apply_ultralytics_environment(workspace)

    # Imported after the environment is set, for the reason above.
    from marp_inference_worker.engines import engine_registry
    from marp_inference_worker.engines.base_engine import JobUnrunnable
    from marp_inference_worker.jobs.context import ChildJobContext

    # Build the context the engine sees. It is the whole boundary: the engine
    # gets these six calls and no way to reach the coordinator.
    ctx = ChildJobContext(
        params=spec.get("params") or {},
        checkpoint_dir=workspace / "work",
        stop_file=stop_file,
        resume_from=None,
        stream=sys.stdout,
    )

    try:
        # Resolve the engine by name. A frame-only engine is refused here.
        engine = engine_registry.get_job_engine(str(spec["engine"]))

        # Preflight before any work. A job this machine cannot run fails now,
        # with a reason, instead of starting and dying (R14).
        engine.preflight(spec)

        # Run it. The engine returns a small summary; bulk results were handed
        # over through ctx.publish_artifact().
        summary = engine.run(ctx, spec)

        # A stop that the engine noticed and unwound from is a cancellation, not
        # a success. The engine says which by setting stopped_early.
        outcome = "cancelled" if summary.get("stopped_early") else "succeeded"
        _emit_terminal(ctx, outcome, {"summary": summary})
        return 0

    except JobUnrunnable as error:
        # A refusal is a clean, informative failure and is reported as one.
        _emit_terminal(
            ctx,
            "failed",
            {"reason": "unrunnable", "message": str(error)},
        )
        return 1

    except Exception as error:
        # Anything else is reported with its traceback, verbatim. A job that
        # died has to say how; a summarized failure cannot be diagnosed.
        _emit_terminal(
            ctx,
            "failed",
            {
                "reason": "exception",
                "message": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        return 1


# _apply_ultralytics_environment()
# Sets the Ultralytics runtime environment for this job.
# Inputs: the job's workspace directory.
# Output: none.
# Use this before importing ultralytics, and only from the child process.
#
# All three matter. YOLO_OFFLINE stops Ultralytics reaching the network mid-job
# for a font, a version check or a dataset. YOLO_AUTOINSTALL stops it pip
# installing into the running environment, which would change what the next job
# runs against. YOLO_CONFIG_DIR points its settings and cache inside this job's
# own workspace, so two jobs on one machine cannot fight over one settings file.
def _apply_ultralytics_environment(workspace: Path) -> None:

    # A per-job config directory, created before Ultralytics looks for it.
    config_dir = workspace / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)

    # No network, no self-installation, and a private config directory.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)


# _emit_terminal()
# Writes the job's final event.
# Inputs: the context, the outcome name, and the outcome payload.
# Output: none.
# Use this on every exit path. The parent reads this event to learn the outcome
# rather than inferring it from the exit code, because an exit code cannot carry
# the reason a job was refused.
def _emit_terminal(ctx: Any, outcome: str, payload: dict[str, Any]) -> None:

    # Delegated to the context, which owns the stream and the sequence numbers.
    ctx.emit_terminal(outcome, payload)


# Entry point guard, so the module can be imported by tests without running.
if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
