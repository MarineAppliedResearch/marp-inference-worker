# worker_main.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Worker service entry point for the MARP Inference Worker.
# This is what a GPU machine actually runs: it reads its one service token from
# the environment, starts the outbound job loop on a background thread, and
# serves the operator's own API on loopback.
#
# The two halves are deliberately unequal. The job loop is the worker; the API is
# a window onto it. The API binds to 127.0.0.1 and nothing outside the machine
# can reach it, which is what makes a worker behind a home router work at all (R1).
#
# Configuration is one token and a coordinator address, and nothing else. GPU
# count, VRAM, driver, disk and engine list are discovered (R2).

# os reads the two environment variables this worker is configured with.
import os

# threading runs the job loop beside the API server.
import threading

# Path types the state directory.
from pathlib import Path


# Environment variable holding the worker's service token.
# The only secret a worker is given. Never logged, never written to disk, and
# never passed on a command line where a process listing would show it.
_TOKEN_ENV = "MARP_WORKER_TOKEN"


# Environment variable holding the coordinator's base address.
# An address, not a host and port written into code: a worker in the office and
# a worker on somebody's home connection reach MARP at different addresses.
_COORDINATOR_ENV = "MARP_COORDINATOR_URL"


# Environment variable overriding where worker state is kept.
# Identity, job workspaces and the in-flight record. Defaults under the service
# directory, which .gitignore covers.
_STATE_DIR_ENV = "MARP_WORKER_STATE_DIR"


# build_runner()
# Builds the job runner from the environment.
# Inputs: none.
# Output: the runner, ready to run.
# Use this from start_worker(). Raises when the token or address is missing,
# because a worker with neither cannot do anything and should say so at start
# rather than failing silently on its first poll.
def build_runner():

    # Imported here so importing this module does not pull the runner's
    # dependency tree into a test that only wanted the API.
    from marp_inference_worker.jobs.coordinator_client import CoordinatorClient
    from marp_inference_worker.jobs.runner import JobRunner
    from marp_inference_worker.jobs.worker_state import WORKER_STATE

    # One token configures the worker.
    service_token = os.environ.get(_TOKEN_ENV)
    if not service_token:
        raise RuntimeError(f"{_TOKEN_ENV} is not set; a worker is configured by one token")

    # And one address to reach MARP at.
    coordinator_url = os.environ.get(_COORDINATOR_ENV)
    if not coordinator_url:
        raise RuntimeError(f"{_COORDINATOR_ENV} is not set")

    # State lives on local disk and has to survive a restart.
    state_dir = Path(os.environ.get(_STATE_DIR_ENV) or (Path("data") / "worker"))

    # Build the client and the runner.
    client = CoordinatorClient(base_url=coordinator_url, service_token=service_token)
    runner = JobRunner(client=client, state_dir=state_dir, worker_state=WORKER_STATE)

    # Publish the discovered capabilities so /status has them before the first
    # poll, and so an operator can see what the machine reported.
    capabilities = runner.capabilities()
    WORKER_STATE.set_capabilities(capabilities, slot_count=int(capabilities["slots"]))

    return runner


# start_worker()
# Starts the job loop on a background thread.
# Inputs: none.
# Output: the runner, so a caller can stop it.
# Use this from the service entry point. A thread rather than a second process:
# the loop and the API have to share one WorkerState, and a job's real isolation
# is its own child process, which the runner already gives it.
def start_worker():

    runner = build_runner()

    # Daemon, so a stopped API server does not leave the loop running. The
    # runner's own jobs are separate processes and are stopped through it.
    thread = threading.Thread(target=runner.run_forever, name="marp-job-runner", daemon=True)
    thread.start()

    return runner


# main()
# Runs the worker: the job loop plus the operator's loopback API.
# Inputs: none.
# Output: none; runs until interrupted.
# Use this as the process a GPU machine starts.
def main() -> None:

    # uvicorn serves the operator API.
    import uvicorn

    # Start taking work first, so a worker is useful even if the API fails to
    # bind -- the API is a window, not the service.
    runner = start_worker()

    try:
        # Loopback only. Not a default that can be overridden by an environment
        # variable: a worker that can be reached from off-machine is a different
        # design, and the point of this one is that push is unrepresentable (R1).
        uvicorn.run(
            "marp_inference_worker.main:app",
            host="127.0.0.1",
            port=int(os.environ.get("MARP_WORKER_API_PORT", "8010")),
            log_level="info",
        )
    finally:
        # Ask the loop to finish so a Ctrl-C does not leave it polling.
        runner.stop()


# Entry point guard, so the module can be imported without starting anything.
if __name__ == "__main__":
    main()
