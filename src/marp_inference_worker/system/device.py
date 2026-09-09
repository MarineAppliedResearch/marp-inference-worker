# device.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Compute device resolution for the MARP Inference Worker.
# A job is pinned to one GPU slot and has to be told which device that is, in a
# form torch and Ultralytics both accept. This file turns a slot index and a
# requested device into a concrete device string, and refuses rather than
# guessing when it cannot.
#
# The rule that matters: "auto" never silently falls back to CPU (R14). A job
# that asked for a GPU and got a CPU does not fail, it just takes a hundred times
# as long and produces results nobody knows were degraded.

# JobUnrunnable is how a device that cannot serve the job is reported.
from marp_inference_worker.engines.base_engine import JobUnrunnable

# Any types the optional torch module.
from typing import Any


# _load_torch()
# Imports torch, or returns None when it is not installed.
# Inputs: none.
# Output: the torch module or None.
# Use this so device questions can be answered on a machine that only serves the
# worker's API and has no ML stack.
def _load_torch() -> Any | None:

    # Absence is a legitimate state, not an error to raise here.
    try:
        import torch

        return torch
    except Exception:
        return None


# cuda_device_count()
# How many CUDA devices this machine has.
# Inputs: none.
# Output: device count, 0 when there is no CUDA or no torch.
# Use this to size the worker's slots, so slot count is discovered rather than
# configured (R2).
def cuda_device_count() -> int:

    torch = _load_torch()
    if torch is None:
        return 0

    # is_available() is the cheap check; device_count() alone can be non-zero on
    # a machine whose driver will not actually initialize.
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        # A broken driver reports as no GPU rather than crashing the worker.
        return 0


# describe_cuda_devices()
# Returns per-device name and memory for the worker's capabilities report.
# Inputs: none.
# Output: list of device mappings; empty when there is no CUDA.
# Use this in enrolment and /status so MARP sees the machine's real hardware.
def describe_cuda_devices() -> list[dict[str, Any]]:

    torch = _load_torch()
    if torch is None or not cuda_device_count():
        return []

    # Read each device's properties. Wrapped because a device can disappear
    # between the count and the query when a driver resets.
    devices: list[dict[str, Any]] = []
    for device_index in range(cuda_device_count()):
        try:
            properties = torch.cuda.get_device_properties(device_index)
            devices.append(
                {
                    "index": device_index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "capability": f"{properties.major}.{properties.minor}",
                }
            )
        except Exception as error:
            # Report the gap rather than omitting the device silently.
            devices.append({"index": device_index, "error": str(error)})
    return devices


# resolve_device()
# Turns a requested device and a slot index into a concrete device string.
# Inputs: what the job asked for ("auto", "cpu", "cuda:1", or None), and the
# slot index this job was pinned to.
# Output: a device string torch and Ultralytics both accept.
# Raises JobUnrunnable when the request cannot be honoured on this machine.
#
# "auto" means "the GPU for my slot", and on a machine with no GPU it is a
# refusal, not a downgrade. A job that wants CPU has to say "cpu" explicitly,
# which makes the slow path a decision somebody made rather than one that
# happened (R14).
def resolve_device(requested: str | None, slot_index: int) -> str:

    # Treat a missing request as "auto", which is what a job spec that says
    # nothing about devices means.
    request = (requested or "auto").strip().lower()

    # An explicit CPU request is honoured without question.
    if request == "cpu":
        return "cpu"

    # An explicit device is checked against what exists, so a job asking for
    # cuda:3 on a two-GPU machine fails now rather than part way through.
    if request.startswith("cuda"):
        device_count = cuda_device_count()
        if device_count == 0:
            raise JobUnrunnable(
                f"job requested device {request!r} but this worker has no usable CUDA device"
            )

        # A bare "cuda" means the slot's device; "cuda:N" names one outright.
        if request == "cuda":
            return f"cuda:{slot_index}"

        try:
            requested_index = int(request.split(":", 1)[1])
        except (IndexError, ValueError):
            raise JobUnrunnable(f"job requested an unparseable device: {requested!r}")

        if requested_index >= device_count:
            raise JobUnrunnable(
                f"job requested {request!r} but this worker has {device_count} CUDA device(s)"
            )
        return f"cuda:{requested_index}"

    # "auto" resolves to this slot's GPU, and refuses if there is not one.
    if request == "auto":
        device_count = cuda_device_count()
        if device_count == 0:
            raise JobUnrunnable(
                "job requested device 'auto' but this worker has no usable CUDA device; "
                "set device to 'cpu' explicitly to run without a GPU"
            )
        if slot_index >= device_count:
            raise JobUnrunnable(
                f"job was pinned to slot {slot_index} but this worker has "
                f"{device_count} CUDA device(s)"
            )
        return f"cuda:{slot_index}"

    # Anything else is a value this worker does not understand.
    raise JobUnrunnable(f"job requested an unsupported device: {requested!r}")
