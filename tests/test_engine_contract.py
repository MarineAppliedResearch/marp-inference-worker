# test_engine_contract.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Tests for the engine contract in the MARP Inference Worker.
# R7 says a job's engine is invoked through one contract and learns nothing
# about MARP, tokens or Jellyfin. These verify both halves: that every job
# engine implements the contract, and that the boundary is actually closed.
#
# The second half is the part worth testing. A contract that is respected today
# and quietly widened next month is not a boundary, and the widening looks
# reasonable at the time -- passing an attempt id "just for logging" is how it
# would start.

# ast inspects engine sources for what they reach for.
import ast

# inspect reads the engine module sources.
import inspect

# Path types the engine module locations.
from pathlib import Path

import pytest

from marp_inference_worker.engines import engine_registry
from marp_inference_worker.engines.base_engine import (
    BaseEngine,
    FrameInferenceCapable,
    JobContext,
    JobUnrunnable,
)


# test_every_job_engine_implements_the_contract()
# Verifies the registry's job engines.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves R7's "one contract" clause: there is one way in, and every engine the
# runner can dispatch to has it.
def test_every_job_engine_implements_the_contract() -> None:

    names = engine_registry.job_engine_names()

    # Both job engines are registered.
    assert set(names) == {"mock", "marp_tracking"}

    for name in names:
        engine = engine_registry.get_job_engine(name)

        # The four contract members, all concrete.
        assert isinstance(engine, BaseEngine)
        assert engine.engine_name == name
        assert callable(engine.run)
        assert callable(engine.preflight)
        assert isinstance(engine.describe(), dict)


# test_frame_only_engine_cannot_be_dispatched_a_job()
# Verifies the typed lookup.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves the registry distinguishes the two kinds of engine. The Ultralytics
# adapter serves the single-frame routes and does not run jobs; asking for it as
# a job engine is refused by name rather than failing later on a missing method.
def test_frame_only_engine_cannot_be_dispatched_a_job() -> None:

    # It is registered, and it does serve frame inference.
    engine = engine_registry.get_frame_engine("ultralytics")
    assert isinstance(engine, FrameInferenceCapable)

    # But it is not a job engine.
    with pytest.raises(ValueError) as raised:
        engine_registry.get_job_engine("ultralytics")
    assert "cannot run jobs" in str(raised.value)


# test_job_engine_is_not_offered_for_frame_inference()
# Verifies the other direction of the typed lookup.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves the capability check replaced the hasattr probes properly. The old code
# tested for the presence of a method name, so a renamed method read as "this
# engine does not support frame inference" rather than as a bug. Now the
# capability is declared by the class.
def test_job_engine_is_not_offered_for_frame_inference() -> None:

    # The tracking engine runs jobs and does not serve the frame routes.
    with pytest.raises(ValueError) as raised:
        engine_registry.get_frame_engine("marp_tracking")
    assert "does not support frame inference" in str(raised.value)


# test_model_manager_no_longer_probes_for_method_names()
# Verifies the hasattr reach-throughs are gone.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves the specific change asked for. The two probes were at
# model_manager.py:124 and :158, and they are the reason a capability check by
# class exists at all -- so this asserts they have not come back.
def test_model_manager_no_longer_probes_for_method_names() -> None:

    from marp_inference_worker.models import model_manager

    source = inspect.getsource(model_manager)

    # No method-name probing of any kind.
    assert "hasattr" not in source

    # And the typed accessor is what is used instead.
    assert "get_frame_engine" in source


# test_context_protocol_is_the_only_thing_engines_are_given()
# Verifies the engine's view of the world.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves R7's "exactly" clause at the protocol, which is what engines are typed
# against. A seventh member here would be a seventh thing an engine could learn.
def test_context_protocol_is_the_only_thing_engines_are_given() -> None:

    members = {name for name in dir(JobContext) if not name.startswith("_")}

    assert members == {
        "params",
        "log",
        "report_progress",
        "report_metrics",
        "checkpoint_dir",
        "resume_from",
        "publish_artifact",
        "should_stop",
    }


# _engine_module_paths()
# Returns the source files of the job engines.
# Inputs: none.
# Output: list of paths.
# Use this for the boundary checks below.
def _engine_module_paths() -> list[Path]:

    from marp_inference_worker.engines import mock_engine, tracking_engine, ultralytics_engine

    return [
        Path(inspect.getfile(module))
        for module in (mock_engine, tracking_engine, ultralytics_engine)
    ]


# test_no_engine_imports_the_coordinator_or_the_runner()
# Verifies the boundary is closed at the import level.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R7's "learns nothing about MARP" clause structurally. An engine that
# cannot import the coordinator client, the runner or the Jellyfin client cannot
# accidentally acquire a second channel to the outside -- and a structural check
# catches that at the moment somebody adds the import, rather than after a job
# has been written that depends on it.
def test_no_engine_imports_the_coordinator_or_the_runner() -> None:

    forbidden_modules = {
        "marp_inference_worker.jobs.coordinator_client",
        "marp_inference_worker.jobs.runner",
        "marp_inference_worker.jobs.job_process",
        "marp_inference_worker.jobs.worker_state",
        "marp_inference_worker.media.jellyfin_client",
        "marp_inference_worker.media.video_source_resolver",
    }

    for path in _engine_module_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):

            # `import x.y.z`
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden_modules, f"{path.name} imports {alias.name}"

            # `from x.y import z`
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in forbidden_modules, f"{path.name} imports {node.module}"

                # And nothing named jellyfin, by any route.
                assert "jellyfin" not in node.module.lower(), f"{path.name} imports {node.module}"


# test_no_engine_reads_a_token_or_a_coordinator_address()
# Verifies engines cannot read the worker's configuration.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves the token half of R7. The worker is configured by one service token; an
# engine that read os.environ could pick it up and use it, which would make the
# engine part of the trust boundary instead of outside it.
def test_no_engine_reads_a_token_or_a_coordinator_address() -> None:

    for path in _engine_module_paths():
        source = path.read_text(encoding="utf-8")

        # No environment reads at all in an engine. The per-job Ultralytics
        # variables are set by the child entry point, which is the worker's
        # side of the boundary, not the engine's (R16).
        assert "os.environ" not in source, path.name
        assert "os.getenv" not in source, path.name

        # And no token or coordinator vocabulary.
        for word in ("MARP_WORKER_TOKEN", "MARP_COORDINATOR_URL", "service_token", "Bearer"):
            assert word not in source, f"{path.name} mentions {word}"


# test_ultralytics_environment_is_set_by_the_worker_not_the_engine()
# Verifies where the Ultralytics controls live.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R16's "controlled by the worker, not the engine" clause. All three
# variables are set in the child entry point, before the first ultralytics
# import -- which is the only place it can work, because the library reads them
# at import time and caches them.
def test_ultralytics_environment_is_set_by_the_worker_not_the_engine() -> None:

    from marp_inference_worker.jobs import child_main

    source = inspect.getsource(child_main)

    # All three, per job.
    assert "YOLO_OFFLINE" in source
    assert "YOLO_AUTOINSTALL" in source
    assert "YOLO_CONFIG_DIR" in source

    # The environment is applied before the engine registry is imported, so
    # nothing has pulled ultralytics in yet.
    apply_at = source.index("_apply_ultralytics_environment(workspace)")
    import_at = source.index("from marp_inference_worker.engines import engine_registry")
    assert apply_at < import_at


# test_ultralytics_version_is_pinned_exactly()
# Verifies the pin.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R16's last clause. Ultralytics ships a release most days and changes
# metric names, argument names and defaults within a minor version, so a range
# means two workers given the same job spec can disagree about what it means.
def test_ultralytics_version_is_pinned_exactly() -> None:

    import re

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")

    # An exact == pin on 8.4.x, not a range and not a floor.
    match = re.search(r'"ultralytics==8\.4\.(\d+)"', pyproject)
    assert match, "ultralytics is not pinned exactly to an 8.4.x version"

    # And what is installed is what is pinned, so the pin is not aspirational.
    from marp_inference_worker.engines.ultralytics_engine import UltralyticsEngine

    installed = UltralyticsEngine.ultralytics_version()
    if installed is not None:
        assert installed == f"8.4.{match.group(1)}"


# test_tracking_engine_refuses_an_unknown_reduction()
# Verifies preflight's reduction check.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R14 and R10b together. A job naming a reduction this worker does not
# have is refused before any work starts -- rather than falling back to a rule
# that did not produce the output, which would misattribute the observations.
def test_tracking_engine_refuses_an_unknown_reduction() -> None:

    engine = engine_registry.get_job_engine("marp_tracking")

    spec = {
        "engine": "marp_tracking",
        "range": {"start_frame": 0, "end_frame": 100},
        "params": {"device": "cpu"},
        "reduction": {"name": "v9_imaginary", "version": "1"},
    }

    with pytest.raises(JobUnrunnable) as raised:
        engine.preflight(spec)

    assert "v9_imaginary" in str(raised.value)


# test_tracking_engine_refuses_a_gpu_job_on_a_machine_without_one()
# Verifies preflight's device check.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R14's central case at the engine. This test is meaningful on the
# machine it was written on precisely because that machine has no GPU: the
# refusal is the real behaviour, not a simulated one.
def test_tracking_engine_refuses_a_gpu_job_on_a_machine_without_one() -> None:

    from marp_inference_worker.system.device import cuda_device_count

    engine = engine_registry.get_job_engine("marp_tracking")

    spec = {
        "engine": "marp_tracking",
        "range": {"start_frame": 0, "end_frame": 100},
        # No device named, which means "auto".
        "params": {},
        "reduction": {"name": "v3_dirpad", "version": "1"},
    }

    if cuda_device_count() == 0:
        with pytest.raises(JobUnrunnable) as raised:
            engine.preflight(spec)
        assert "no usable CUDA device" in str(raised.value)
    else:
        # On a GPU machine this must NOT refuse, or the check is too strict.
        engine.preflight(spec)


# test_tracking_engine_describes_its_real_dependencies()
# Verifies the tracking engine's describe().
# Inputs: none.
# Output: pytest pass/fail result.
# Proves R12 at the engine level: what /status reports about this engine is read
# from what is installed rather than written in.
def test_tracking_engine_describes_its_real_dependencies() -> None:

    from marp_inference_worker.engines.ultralytics_engine import UltralyticsEngine

    described = engine_registry.get_job_engine("marp_tracking").describe()

    assert described["engine"] == "marp_tracking"
    assert described["pipeline"] == "detect -> track -> reduce"
    assert described["tracker"] == "bytetrack"
    assert described["requires_gpu"] is True

    # The version is whatever is actually installed, including None.
    assert described["ultralytics_version"] == UltralyticsEngine.ultralytics_version()
