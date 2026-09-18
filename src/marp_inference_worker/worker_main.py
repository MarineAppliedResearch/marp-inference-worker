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
import argparse
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
def build_runner(screen_mode: str = "window", slot_count: int | None = None):

    # Imported here so importing this module does not pull the runner's
    # dependency tree into a test that only wanted the API.
    from marp_inference_worker.jobs.coordinator_client import CoordinatorClient
    from marp_inference_worker.jobs.runner import JobRunner
    from marp_inference_worker.jobs.worker_state import WORKER_STATE
    from marp_inference_worker.installation.operator_control import read_action

    # State and its protected credential survive versions and restarts.
    state_dir = Path(os.environ.get(_STATE_DIR_ENV) or (Path("data") / "worker"))

    # One token configures the worker.
    service_token = os.environ.get(_TOKEN_ENV)
    if not service_token:
        from marp_inference_worker.installation.activation import credential_path
        from marp_inference_worker.installation.credential_store import load

        service_token = load(credential_path(state_dir))
    if not service_token:
        raise RuntimeError("this worker is not activated")

    # And one address to reach MARP at.
    coordinator_url = os.environ.get(_COORDINATOR_ENV)
    if not coordinator_url:
        raise RuntimeError(f"{_COORDINATOR_ENV} is not set")

    # Build the client and the runner.
    client = CoordinatorClient(base_url=coordinator_url, service_token=service_token)
    runner = JobRunner(
        client=client,
        state_dir=state_dir,
        worker_state=WORKER_STATE,
        slot_count=slot_count,
        screen_mode=screen_mode,
    )

    # Publish the discovered capabilities so /status has them before the first
    # poll, and so an operator can see what the machine reported.
    capabilities = runner.capabilities()
    WORKER_STATE.set_capabilities(capabilities, slot_count=int(capabilities["slots"]))
    WORKER_STATE.set_operator_action(read_action())
    WORKER_STATE.set_screen_mode(screen_mode)

    return runner


# start_worker()
# Starts the job loop on a background thread.
# Inputs: none.
# Output: the runner, so a caller can stop it.
# Use this from the service entry point. A thread rather than a second process:
# the loop and the API have to share one WorkerState, and a job's real isolation
# is its own child process, which the runner already gives it.
def start_worker(screen_mode: str = "window", slot_count: int | None = None):

    runner = build_runner(screen_mode, slot_count)

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

    parser = argparse.ArgumentParser(description="Run the MARP inference worker")
    parser.add_argument(
        "--screen",
        choices=("off", "window", "fullscreen"),
        default="window",
        help="permit watched jobs to open a local display",
    )
    # How many jobs to run at once. Omitted, the worker measures the machine
    # -- see derive_slot_count(). Given, it is believed without argument: a
    # person who has watched their own machine knows something the derivation
    # does not, and a small card wants fewer slots than its memory suggests.
    parser.add_argument(
        "--slots",
        type=int,
        default=None,
        help="how many jobs to run at once; omit to decide from this machine",
    )
    parser.add_argument("--activate-code-file", help=argparse.SUPPRESS)
    parser.add_argument("--coordinator-url", help=argparse.SUPPRESS)
    parser.add_argument("--state-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.activate_code_file:
        from marp_inference_worker.installation.activation import activate

        code_path = Path(args.activate_code_file)
        try:
            coordinator = args.coordinator_url or os.environ.get(_COORDINATOR_ENV)
            if not coordinator:
                raise RuntimeError("coordinator URL is required for activation")
            state_dir = Path(args.state_dir or os.environ.get(_STATE_DIR_ENV) or (Path("data") / "worker"))
            activate(coordinator, code_path.read_text(encoding="utf-8").strip(), state_dir)
        finally:
            code_path.unlink(missing_ok=True)
        return

    # uvicorn serves the operator API.
    import uvicorn

    # Start taking work first, so a worker is useful even if the API fails to
    # bind -- the API is a window, not the service.
    runner = start_worker(args.screen, args.slots)

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
