# File name: test_system_resources.py
# Date created: 2026-07-08
# Author: Isaac Travers
# Purpose: Tests the worker system resource API endpoint.
# System role: Protects the coordinator-facing resource route contract.
# Code in this file should test resource endpoint shape without requiring specific hardware.

# FastAPI TestClient provides in-process HTTP testing for API routes.
from fastapi.testclient import TestClient

# The app object exposes the configured MARP inference worker API.
from marp_inference_worker.main import app


# Shared API client used by this test module.
client = TestClient(app)


# Verify that the system resources route exists and returns the expected top-level shape.
# Inputs: none.
# Outputs: passing test when the route returns resource sections.
# This protects the future coordinator-facing resource endpoint contract.
def test_system_resources_route_returns_expected_sections():

    # Request the worker resource snapshot.
    response = client.get("/system/resources")

    # The resource endpoint should return successfully.
    assert response.status_code == 200

    # Parse the response body as JSON.
    data = response.json()

    # Verify stable top-level resource sections.
    assert "host" in data
    assert "cpu" in data
    assert "memory" in data
    assert "disk" in data
    assert "python" in data
    assert "torch" in data
    assert "nvidia" in data
    assert "job_pressure" in data

    # Verify a few coordinator-useful nested fields.
    assert "platform" in data["host"]
    assert "logical_cpu_count" in data["cpu"]
    assert "percent_used" in data["memory"]
    assert "free_bytes" in data["disk"]
    assert "torch_available" in data["torch"]
    assert "nvml_available" in data["nvidia"]
    assert "worker_accepting_jobs" in data["job_pressure"]

# psutil is monkeypatched to prove the sampled CPU read does not block.
import psutil

# The module itself, for the sampler that feeds the per-minute chart.
from marp_inference_worker.system import resource_monitor


# test_the_snapshot_says_how_long_this_worker_has_been_up()
# Verifies the one thing on the Workers page MARP cannot answer itself.
# Inputs: none.
# Output: pytest pass/fail result.
#
# MARP knows when a worker last spoke, not when it started -- so a machine that
# restarted between two polls looked identical to one up for a week.
def test_the_snapshot_says_how_long_this_worker_has_been_up() -> None:

    process = resource_monitor.get_system_resources()["process"]

    assert isinstance(process["uptime_s"], (int, float))
    assert process["uptime_s"] >= 0


# test_the_sample_carries_cpu_memory_and_disk_under_the_resources_names()
# Verifies the chart gets the three blocks, spelled once.
# Inputs: none.
# Output: pytest pass/fail result.
#
# `capabilities` is a snapshot and cannot show pressure over time, so these
# ride with the GPU telemetry. Under the SAME names `resources` uses, because
# two spellings of one fact is how a reader ends up not knowing which is
# current.
def test_the_sample_carries_cpu_memory_and_disk_under_the_resources_names() -> None:

    sample = resource_monitor.get_gpu_telemetry() or {}
    snapshot = resource_monitor.get_system_resources()

    for block in ("cpu", "memory", "disk"):
        assert block in sample, f"{block} missing from the per-minute sample"
        assert block in snapshot, f"{block} missing from the snapshot"


# test_the_sampled_cpu_read_does_not_block_the_heartbeat()
# Verifies the hot-path read is the non-blocking form.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
#
# **`cpu_percent(interval=0.1)` sleeps for a tenth of a second.** That is fine
# in the snapshot route and wrong on the loop that heartbeats every job on the
# machine -- four slots would spend four tenths of a second a cycle inside a
# timer. `interval=None` reports the percentage since the previous call, which
# is what a ten-second sampler actually wants anyway.
def test_the_sampled_cpu_read_does_not_block_the_heartbeat(monkeypatch) -> None:

    seen = {}

    def recording_cpu_percent(interval=None, percpu=False):
        seen["interval"] = interval
        return 12.5

    monkeypatch.setattr(psutil, "cpu_percent", recording_cpu_percent)

    block = resource_monitor._get_sampled_cpu()

    assert seen["interval"] is None, "the heartbeat's CPU read would block"
    assert block["cpu_percent"] == 12.5


# test_a_machine_with_no_gpu_still_reports_its_pressure()
# Verifies the system blocks are not hostage to having a GPU.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
#
# A CPU-only volunteer still has a CPU worth charting. The GPU-absent case used
# to return None outright.
def test_a_machine_with_no_gpu_still_reports_its_pressure(monkeypatch) -> None:

    monkeypatch.setattr(resource_monitor, "_get_nvidia_resources", lambda: {"devices": []})

    sample = resource_monitor.get_gpu_telemetry() or {}

    assert "devices" not in sample
    assert "cpu" in sample


# test_one_unreadable_block_does_not_cost_the_others()
# Verifies a failing sensor degrades the sample rather than emptying it.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
def test_one_unreadable_block_does_not_cost_the_others(monkeypatch) -> None:

    def exploding(*_args, **_kwargs):
        raise RuntimeError("sensor gone")

    monkeypatch.setattr(resource_monitor, "_get_memory_resources", exploding)

    sample = resource_monitor.get_gpu_telemetry() or {}

    assert "memory" not in sample
    assert "cpu" in sample and "disk" in sample


# test_the_first_cpu_sample_is_null_not_zero()
# Verifies "not measured yet" does not draw as "idle".
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
#
# **psutil returns 0.0 when it has no previous call to compare against.** On a
# chart a zero means the machine was idle, and "no window yet" is not that: a
# line breaks on null and draws a gap, but draws a zero as a floor. Every
# worker's chart would have opened with a dip to the bottom that never
# happened.
#
# Only this side can tell them apart -- here it is just "is this the first
# call", and from the series a worker's first sample looks exactly like a
# genuinely idle minute after a restart.
def test_the_first_cpu_sample_is_null_not_zero(monkeypatch) -> None:

    monkeypatch.setattr(resource_monitor, "_CPU_SAMPLED_BEFORE", False)
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None, percpu=False: 0.0)

    first = resource_monitor._get_sampled_cpu()
    assert first["cpu_percent"] is None

    # The counts are real on the first sample and are still reported.
    assert first["physical_cpu_count"] is not None


# test_a_genuine_zero_is_still_reported()
# The control for the test above.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
#
# Without it, returning None unconditionally would pass -- and an idle machine
# would then be indistinguishable from an unmeasured one, which is the same
# defect pointing the other way.
def test_a_genuine_zero_is_still_reported(monkeypatch) -> None:

    monkeypatch.setattr(resource_monitor, "_CPU_SAMPLED_BEFORE", True)
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None, percpu=False: 0.0)

    assert resource_monitor._get_sampled_cpu()["cpu_percent"] == 0.0


# test_the_first_call_still_primes_the_next_one()
# Verifies withholding the value does not cost the following sample.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
#
# psutil measures against its own previous call, so the first read must still
# HAPPEN even though its value is withheld -- otherwise the second sample would
# be the one with no window, and the gap would just move.
def test_the_first_call_still_primes_the_next_one(monkeypatch) -> None:

    calls = []
    monkeypatch.setattr(resource_monitor, "_CPU_SAMPLED_BEFORE", False)
    monkeypatch.setattr(
        psutil, "cpu_percent",
        lambda interval=None, percpu=False: (calls.append(1), 41.0)[1],
    )

    assert resource_monitor._get_sampled_cpu()["cpu_percent"] is None
    assert len(calls) == 1, "psutil was not primed on the first sample"

    assert resource_monitor._get_sampled_cpu()["cpu_percent"] == 41.0


# test_the_sampled_memory_and_disk_keys_match_the_snapshot()
# Verifies a chart keyed on the snapshot's names finds them in the sample.
# Inputs: none.
# Output: pytest pass/fail result.
def test_the_sampled_memory_and_disk_keys_match_the_snapshot() -> None:

    sample = resource_monitor.get_gpu_telemetry() or {}
    snapshot = resource_monitor.get_system_resources()

    for block in ("memory", "disk"):
        assert set(sample[block]) == set(snapshot[block]), f"{block} keys differ"

    for name in ("used_bytes", "total_bytes", "percent_used"):
        assert name in sample["memory"]
