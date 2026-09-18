# test_status.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Tests for the MARP Inference Worker status and control endpoints.
# These verify that /status reports what is actually true rather than a fixed
# response, and that pause/resume works from the machine's own operator.
#
# The previous version of this file asserted the hard-coded literal the old
# route returned -- worker_id "dev-worker", status "idle", version "0.1.0". It
# passed on an idle worker, a busy one and one that had never enrolled, which is
# exactly the defect it should have caught. It is replaced rather than adjusted.

from fastapi.testclient import TestClient

from marp_inference_worker.main import app

# The shared state the route reads, so a test can put the worker in a state.
from marp_inference_worker.jobs.worker_state import WORKER_STATE


# test_status_reports_unenrolled_worker_honestly()
# Verifies that a worker with no coordinator identity says so.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves R12: the reported identity is the real one, and a worker that has not
# enrolled is not given a fabricated id.
def test_status_reports_unenrolled_worker_honestly() -> None:

    # Start from a known state; other tests in this file set identity.
    WORKER_STATE.set_identity(local_id="local-abc", worker_id=None)
    WORKER_STATE.set_jobs([])
    WORKER_STATE.set_paused(False)

    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()

    # No coordinator id yet, and the route says so rather than inventing one.
    assert body["worker_id"] is None
    assert body["enrolled"] is False
    assert body["local_id"] == "local-abc"

    # Idle with no jobs.
    assert body["status"] == "idle"
    assert body["active_jobs"] == []


# test_status_reflects_a_running_job()
# Verifies that /status changes when the runner reports a job.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves R12: the running-job list is the runner's, not a constant. The old
# route returned active_jobs 0 unconditionally, which this catches.
def test_status_reflects_a_running_job() -> None:

    WORKER_STATE.set_identity(local_id="local-abc", worker_id="worker-7")
    WORKER_STATE.set_capabilities({"slots": 2}, slot_count=2)
    WORKER_STATE.set_paused(False)
    WORKER_STATE.set_jobs(
        [
            {
                "attempt_id": "attempt-1",
                "slot_index": 0,
                "engine": "marp_tracking",
                "progress": {"done": 120, "total": 1800, "unit": "frames"},
            }
        ]
    )

    client = TestClient(app)
    body = client.get("/status").json()

    # The identity, the state and the job all come through.
    assert body["worker_id"] == "worker-7"
    assert body["status"] == "working"
    assert len(body["active_jobs"]) == 1
    assert body["active_jobs"][0]["attempt_id"] == "attempt-1"

    # Slot accounting is derived from the job list, so it cannot disagree.
    # `permitted` is what the worker has decided it can run at real time, and
    # `free` is counted against it rather than against the ceiling. Reporting
    # `total - busy` advertised spare capacity on a machine that had worked out
    # it had none. Unset, it falls back to the ceiling, which is this case.
    assert body["slots"] == {"total": 2, "permitted": 2, "busy": 1, "free": 1}

    # Reset so the next test starts clean.
    WORKER_STATE.set_jobs([])


# test_status_lists_the_engines_actually_registered()
# Verifies that the engine list is read from the registry.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves R12: the engine list is discovered. Both job engines must appear, and
# each must say whether it needs a GPU, so a coordinator can route to a machine
# that has none.
def test_status_lists_the_engines_actually_registered() -> None:

    client = TestClient(app)
    body = client.get("/status").json()

    engines_by_name = {entry["engine"]: entry for entry in body["engines"]}

    # Both job engines are registered and describe themselves.
    assert "mock" in engines_by_name
    assert "marp_tracking" in engines_by_name

    # The tracking engine needs a GPU; the mock engine does not.
    assert engines_by_name["marp_tracking"]["requires_gpu"] is True
    assert engines_by_name["mock"]["requires_gpu"] is False

    # The tracking engine reports which keyframe reductions it can apply, so a
    # coordinator will not send it a job naming a rule it does not have.
    reductions = engines_by_name["marp_tracking"]["reductions"]
    assert {"name": "v3_dirpad", "version": "1"} in reductions


# test_status_version_is_not_a_literal()
# Verifies that the reported version comes from the installed package.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves R12 for the version field specifically. The old route returned the
# string "0.1.0" regardless of what pyproject.toml said.
def test_status_version_is_not_a_literal() -> None:

    from importlib.metadata import version as package_version

    client = TestClient(app)
    body = client.get("/status").json()

    # Whatever the installed version is, that is what must be reported.
    assert body["version"] == package_version("marp-inference-worker")


# test_pause_and_resume_change_what_status_reports()
# Verifies the operator's pause control.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves R1's operator half: the loopback API can stop this worker taking new
# work, and the reason is recorded.
def test_pause_and_resume_change_what_status_reports(monkeypatch, tmp_path) -> None:

    monkeypatch.setenv("MARP_WORKER_STATE_DIR", str(tmp_path))
    client = TestClient(app)

    # Pause with a reason.
    paused = client.post("/status/pause", json={"paused": True, "reason": "operator needs the GPU"})
    assert paused.status_code == 200
    assert paused.json()["paused"] is True
    assert paused.json()["status"] == "paused"
    assert paused.json()["pause_reason"] == "operator needs the GPU"

    # The change is visible through /status, not only in the response.
    assert client.get("/status").json()["paused"] is True

    # Resume clears both the flag and the reason.
    resumed = client.post("/status/pause", json={"paused": False})
    assert resumed.json()["paused"] is False
    assert resumed.json()["pause_reason"] is None
    assert resumed.json()["status"] == "idle"


def test_compute_controls_persist_and_resume(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MARP_WORKER_STATE_DIR", str(tmp_path))
    client = TestClient(app)

    finishing = client.post("/status/control", json={"action": "finish"})
    assert finishing.status_code == 200
    assert finishing.json()["operator_action"] == "finish"
    assert finishing.json()["paused"] is True

    persisted = (tmp_path / "operator-control.json").read_text(encoding="utf-8")
    assert '"action": "finish"' in persisted

    stopping = client.post("/status/control", json={"action": "stop"})
    assert stopping.json()["operator_action"] == "stop"

    resumed = client.post("/status/control", json={"action": "resume"})
    assert resumed.json()["operator_action"] == "running"
    assert resumed.json()["paused"] is False


def test_screen_shortcut_changes_future_job_display_policy() -> None:
    client = TestClient(app)
    response = client.post("/status/screen", json={"mode": "fullscreen"})

    assert response.status_code == 200
    assert response.json()["screen_mode"] == "fullscreen"
    assert client.get("/status").json()["screen_mode"] == "fullscreen"

    client.post("/status/screen", json={"mode": "off"})


def test_watch_page_origin_can_call_loopback_controls() -> None:
    client = TestClient(app)
    response = client.options(
        "/status/control",
        headers={
            "Origin": "http://127.0.0.1:54321",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:54321"


# test_track_total_survives_a_job_ending()
# Verifies the machine-wide track count is not reset by a job rolling over.
# Inputs: none.
# Output: pytest pass/fail result.
#
# The window shows "N of TOTAL tracks", where TOTAL is every distinct animal
# this worker has tracked since it started. A running job carries its own count, so a
# total read only from running jobs would fall back to nothing every time a
# piece finished — and a counter that goes backwards is worse than no counter.
def test_track_total_survives_a_job_ending() -> None:
    from marp_inference_worker.jobs.worker_state import WorkerState

    state = WorkerState()
    assert state.describe()["tracks_total"] == 0

    state.set_jobs([
        {"progress": {"tracks": 40}},
        {"progress": {"tracks": 12}},
    ])
    assert state.describe()["tracks_total"] == 52

    # The first job ends: its count is banked, its row goes, and the total holds.
    state.retire_job_tracks(40)
    state.set_jobs([{"progress": {"tracks": 12}}])
    assert state.describe()["tracks_total"] == 52

    # And a new job adds to it rather than replacing it.
    state.retire_job_tracks(12)
    state.set_jobs([{"progress": {"tracks": 3}}])
    assert state.describe()["tracks_total"] == 55
