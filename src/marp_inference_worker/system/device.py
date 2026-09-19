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
# The rule that matters: "auto" falls back to the CPU on a machine with no GPU,
# and says so loudly every time it does.
#
# This reverses R14, which refused instead. R14's reasoning was sound about the
# danger -- "results nobody knows were degraded" -- but its remedy made a machine
# without a graphics card useless to MARP, because the coordinator sends specs
# with no device, which means "auto". Such a worker enrolled, polled, heartbeated
# and declined every job while reporting itself healthy. The volunteer programme's
# goal is any computer, with a GPU or without, so the fallback is now taken and
# the degradation is announced rather than hidden.

# JobUnrunnable is how a device that cannot serve the job is reported.
# loguru carries the CPU-fallback warning, which has to be visible per job
# rather than once at startup.
from loguru import logger

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
# "auto" means "the GPU for my slot", and on a machine with no GPU it means the
# CPU, with a warning on every job. An explicitly requested "cuda" still refuses
# when there is no card: a job that named a GPU outright wanted one, and quietly
# giving it a CPU is the degradation R14 was right to object to.
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

        # A bare "cuda" means this slot's device; "cuda:N" names one outright.
        if request == "cuda":
            return f"cuda:{slot_index % device_count}"

        try:
            requested_index = int(request.split(":", 1)[1])
        except (IndexError, ValueError):
            raise JobUnrunnable(f"job requested an unparseable device: {requested!r}")

        if requested_index >= device_count:
            raise JobUnrunnable(
                f"job requested {request!r} but this worker has {device_count} CUDA device(s)"
            )
        return f"cuda:{requested_index}"

    # "auto" resolves to this slot's GPU, and falls back to the CPU when there is
    # not one.
    if request == "auto":
        device_count = cuda_device_count()
        if device_count == 0:
            # This used to refuse, telling the caller to ask for 'cpu' explicitly.
            # That made a machine without a GPU useless to MARP: the coordinator
            # sends specs with no device, which means 'auto', so such a worker
            # enrolled, polled, heartbeated and then declined every job it was
            # given -- while reporting itself healthy.
            #
            # A volunteer computer without an NVIDIA card is a computer that can
            # still contribute, slowly. "Any computer, any GPU, or no GPU" is the
            # stated goal of the volunteer programme, and refusing here is what
            # stopped it being true. Slow work is worth more than no work.
            #
            # R14's objection was never to the CPU -- it was to the fallback being
            # *quiet*: "results nobody knows were degraded, which is worse than an
            # error". That objection is answered by announcing it rather than by
            # refusing, so this is loud. A machine running every job twenty times
            # slower than expected should say so on every job, not once at startup,
            # because the person reading the log may not have seen the startup.
            logger.warning(
                "no CUDA device on this worker; running this job on the CPU. "
                "This is roughly twenty times slower than a GPU. Install an NVIDIA "
                "driver and restart to use a graphics card."
            )
            return "cpu"

        # Slots share the cards, round robin.
        #
        # A slot used to *be* a device -- the worker ran one job per GPU, so
        # `slot_index` and the device index were the same number and a slot past
        # the last card was a contradiction worth refusing. A slot is now a unit
        # of concurrency: one card can hold several jobs, measured at well under
        # 1 GiB each on a 16 GiB card, and the GPU is not the constraint anyway.
        # So the index wraps instead of refusing, and a two-card machine
        # alternates between them.
        return f"cuda:{slot_index % device_count}"

    # Anything else is a value this worker does not understand.
    raise JobUnrunnable(f"job requested an unsupported device: {requested!r}")
