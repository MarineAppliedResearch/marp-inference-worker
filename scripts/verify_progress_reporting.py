# verify_progress_reporting.py
#
# Cross-repository verification for issue #10. It submits a diagnostic job to a
# running MARP_API, then drives the real worker client, runner and child process
# until the API records the terminal attempt.

import hashlib
import os
import sys
import time
from pathlib import Path

import httpx

from marp_inference_worker.jobs.coordinator_client import CoordinatorClient
from marp_inference_worker.jobs.runner import JobRunner
from marp_inference_worker.jobs.worker_state import WorkerState


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    base_url = os.environ.get("MARP_COORDINATOR_URL", "").rstrip("/")
    token = os.environ.get("MARP_WORKER_TOKEN", "")
    require(bool(base_url), "MARP_COORDINATOR_URL is required")
    require(bool(token), "MARP_WORKER_TOKEN is required")

    local_root = Path(__file__).resolve().parents[1] / ".marp" / "local" / "progress-e2e"
    local_root.mkdir(parents=True, exist_ok=True)
    model_path = local_root / "model.bin"
    model_bytes = b"MARP progress reporting end-to-end model sentinel\n"
    model_path.write_bytes(model_bytes)
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()

    headers = {"Authorization": f"Bearer {token}"}
    http = httpx.Client(base_url=base_url, headers=headers, timeout=30.0)
    coordinator = CoordinatorClient(base_url, token)
    runner = JobRunner(
        client=coordinator,
        state_dir=local_root / f"worker-{time.time_ns()}",
        worker_state=WorkerState(),
        slot_count=1,
    )

    try:
        submitted = http.post(
            "/api/v2/gpu/jobs",
            json={
                "kind": "diagnostic",
                "spec": {
                    "engine": "mock",
                    "model": {
                        "name": "progress-e2e-model",
                        "sha256": model_sha256,
                        "url": str(model_path),
                    },
                    "video": {
                        "url": "file:///not-opened-by-the-mock-engine.mp4",
                        "source_name": "progress-e2e.mp4",
                    },
                    "range": {"start_frame": 200, "end_frame": 240},
                    "params": {"frame_delay_s": 0.25},
                    "reduction": {"name": "v3_dirpad", "version": "1"},
                },
            },
        )
        submitted.raise_for_status()
        job_id = submitted.json()["jobs"][0]["id"]

        runner.ensure_enrolled(force=True)
        require(runner._poll_once(), "the real worker client did not lease the submitted job")
        runner._service_running_jobs()

        live = http.get(f"/api/v2/gpu/jobs/{job_id}")
        live.raise_for_status()
        live_attempt = live.json()["attempts"][0]
        require(live_attempt["progress_done"] == 0, "first API snapshot was not before frame work")
        require(live_attempt["progress_total"] == 40, "API did not receive the half-open range total")
        require(bool(live_attempt["progress_phase"]), "API did not receive a starting phase")
        require(live_attempt["progress_elapsed_s"] >= 0, "API did not retain elapsed seconds")

        pool = http.get("/api/v2/gpu/workers")
        pool.raise_for_status()
        pool_attempts = [
            attempt
            for worker in pool.json()
            for attempt in worker["attempts"]
            if attempt["job"]["job_id"] == job_id
        ]
        require(len(pool_attempts) == 1, "worker pool did not expose the live attempt")
        require(
            pool_attempts[0]["progress"]["phase"] == live_attempt["progress_phase"],
            "job detail and worker pool disagreed on phase",
        )

        deadline = time.monotonic() + 30.0
        while runner._jobs_by_slot and time.monotonic() < deadline:
            runner._service_running_jobs()
            time.sleep(0.05)
        require(not runner._jobs_by_slot, "job did not finish before the deadline")

        terminal = http.get(f"/api/v2/gpu/jobs/{job_id}")
        terminal.raise_for_status()
        final_attempt = terminal.json()["attempts"][0]
        require(final_attempt["state"] == "succeeded", "attempt did not succeed")
        require(final_attempt["progress_done"] == 40, "terminal frame count was not retained")
        require(final_attempt["progress_total"] == 40, "terminal total was not retained")
        require(final_attempt["progress_phase"] == "publishing", "terminal phase was not retained")
        require(final_attempt["progress_elapsed_s"] > 0, "terminal elapsed seconds were not retained")

        print(
            "PASS: real worker client -> MARP_API -> PostgreSQL retained "
            "live and terminal progress for job " + str(job_id)
        )
        return 0
    finally:
        for job in runner._jobs_by_slot.values():
            job.request_stop()
        coordinator.close()
        http.close()


if __name__ == "__main__":
    sys.exit(main())
