# test_module_imports.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Import-order tests for the MARP Inference Worker.
#
# These exist because of a defect this work introduced and then found: a circular
# import between the engines and the models package. models/__init__.py
# re-exports model_manager, model_manager imports engine_registry, and
# engine_registry imports the engine modules -- so an engine module that imported
# ModelSpec at runtime completed the loop.
#
# It was invisible under most conditions. Importing marp_inference_worker.main
# first pulled the package in through a path that happened to work, so the whole
# suite passed; only a test that imported an engine module before anything else
# failed. That is the worst kind of breakage, because whether it appears depends
# on test collection order.
#
# Each module is therefore imported in a fresh interpreter, first, with nothing
# else loaded. One subprocess per module is slower than one import loop, and it
# is the only way to actually test import order rather than testing whatever
# order pytest happened to produce.

# subprocess gives each import a clean interpreter.
import subprocess

# sys names the interpreter to launch.
import sys

import pytest


# Every module in the package that a caller might import first.
# Listed explicitly rather than discovered, so adding a module is a deliberate
# decision about whether it must be independently importable.
_MODULES = [
    "marp_inference_worker.main",
    "marp_inference_worker.worker_main",
    "marp_inference_worker.api.app",
    "marp_inference_worker.api.status_routes",
    "marp_inference_worker.engines.base_engine",
    "marp_inference_worker.engines.engine_registry",
    "marp_inference_worker.engines.mock_engine",
    "marp_inference_worker.engines.tracking_engine",
    "marp_inference_worker.engines.ultralytics_engine",
    "marp_inference_worker.jobs.child_main",
    "marp_inference_worker.jobs.context",
    "marp_inference_worker.jobs.coordinator_client",
    "marp_inference_worker.jobs.identity",
    "marp_inference_worker.jobs.job_process",
    "marp_inference_worker.jobs.job_spec",
    "marp_inference_worker.jobs.runner",
    "marp_inference_worker.jobs.worker_state",
    "marp_inference_worker.media.frame_range_reader",
    "marp_inference_worker.models.model_cache",
    "marp_inference_worker.models.model_manager",
    "marp_inference_worker.models.model_spec",
    "marp_inference_worker.reduction.keyframes",
    "marp_inference_worker.system.device",
    "marp_inference_worker.system.resource_monitor",
    "marp_inference_worker.tracking.byte_tracker_adapter",
    "marp_inference_worker.tracking.observations",
    "marp_inference_worker.tracking.track_accumulator",
]


# test_module_imports_first_in_a_clean_interpreter(module_name)
# Verifies one module can be the first thing imported.
# Inputs: the module name, parametrized.
# Output: pytest pass/fail result.
#
# Proves there is no import cycle reachable from any entry point. A cycle here
# is not a style problem: it makes whether the worker starts depend on which
# module the caller happened to name first.
@pytest.mark.parametrize("module_name", _MODULES)
def test_module_imports_first_in_a_clean_interpreter(module_name: str) -> None:

    completed = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        timeout=180,
        # Explicitly not check=True: the assertion below reports stderr, which
        # a CalledProcessError would replace with a bare exit code.
        check=False,
    )

    # The stderr is included in the failure so a cycle is diagnosable from the
    # test output alone, without having to reproduce it by hand.
    assert completed.returncode == 0, (
        f"importing {module_name} first failed:\n{completed.stderr[-3000:]}"
    )
