# File name: resource_monitor.py
# Date created: 2026-07-08
# Author: Isaac Travers
# Purpose: Collects local worker machine resource information.
# System role: Provides system state data for API routes and future coordinator decisions.
# Code in this file should inspect local resources but should not schedule jobs.

# Standard library imports provide OS, Python, host, and disk path information.
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

# psutil provides cross-platform CPU, memory, and disk usage metrics.
import psutil


# Build one coordinator-friendly resource snapshot for this worker.
# Inputs: none.
# Outputs: JSON-serializable dictionary of system resource metrics.
# Use this from API routes or future worker-state services.
def get_system_resources() -> dict[str, Any]:

    # The current working directory is useful as a default disk target for this service.
    current_path = Path.cwd()

    # Resource snapshot returned to API callers.
    resources: dict[str, Any] = {
        "host": _get_host_resources(),
        "cpu": _get_cpu_resources(),
        "memory": _get_memory_resources(),
        "disk": _get_disk_resources(current_path),
        "python": _get_python_resources(),
        "torch": _get_torch_resources(),
        "job_pressure": _get_empty_job_pressure(),
    }

    return resources


# Collect stable host and OS identity fields.
# Inputs: none.
# Outputs: JSON-serializable host metadata.
# This helps a coordinator distinguish workers.
def _get_host_resources() -> dict[str, Any]:

    # Host and platform metadata.
    host_resources: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    return host_resources


# Collect CPU capacity and current load information.
# Inputs: none.
# Outputs: JSON-serializable CPU metrics.
# These fields help estimate whether the worker is currently busy.
def _get_cpu_resources() -> dict[str, Any]:

    # Logical CPU count includes hyperthreaded cores.
    logical_cpu_count = psutil.cpu_count(logical=True)

    # Physical CPU count may be unavailable on some systems.
    physical_cpu_count = psutil.cpu_count(logical=False)

    # Sample per-core CPU pressure once so the route returns live values without double waiting.
    per_cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)

    # Average the per-core values into a whole-worker CPU pressure number.
    cpu_percent = _safe_average(per_cpu_percent)

    # Windows does not always expose Unix-style load averages.
    load_average = _get_load_average()

    # CPU metrics returned to the route.
    cpu_resources: dict[str, Any] = {
        "logical_cpu_count": logical_cpu_count,
        "physical_cpu_count": physical_cpu_count,
        "cpu_percent": cpu_percent,
        "per_cpu_percent": per_cpu_percent,
        "load_average": load_average,
    }

    return cpu_resources


# Collect system RAM capacity and current usage.
# Inputs: none.
# Outputs: JSON-serializable memory metrics.
# These fields help the coordinator avoid overloading a worker.
def _get_memory_resources() -> dict[str, Any]:

    # psutil returns current virtual memory statistics.
    virtual_memory = psutil.virtual_memory()

    # Memory metrics returned to the route.
    memory_resources: dict[str, Any] = {
        "total_bytes": virtual_memory.total,
        "available_bytes": virtual_memory.available,
        "used_bytes": virtual_memory.used,
        "free_bytes": virtual_memory.free,
        "percent_used": virtual_memory.percent,
    }

    return memory_resources


# Collect disk usage for the service working path.
# Inputs: path to inspect.
# Outputs: JSON-serializable disk metrics.
# This is a foundation for later cache, artifact, and job-storage checks.
def _get_disk_resources(path: Path) -> dict[str, Any]:

    # shutil.disk_usage is cross-platform and works on Windows and Linux paths.
    disk_usage = shutil.disk_usage(path)

    # Disk metrics returned to the route.
    disk_resources: dict[str, Any] = {
        "path": str(path),
        "total_bytes": disk_usage.total,
        "used_bytes": disk_usage.used,
        "free_bytes": disk_usage.free,
        "percent_used": _safe_percent(disk_usage.used, disk_usage.total),
    }

    return disk_resources


# Collect Python runtime details for debugging worker compatibility.
# Inputs: none.
# Outputs: JSON-serializable Python metadata.
# This helps diagnose environment differences across worker machines.
def _get_python_resources() -> dict[str, Any]:

    # Python runtime metadata.
    python_resources: dict[str, Any] = {
        "version": sys.version,
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "executable": sys.executable,
    }

    return python_resources


# Collect Torch and CUDA visibility without making Torch a hard import at module load.
# Inputs: none.
# Outputs: JSON-serializable Torch and GPU metadata.
# This helps the coordinator understand ML runtime capacity.
def _get_torch_resources() -> dict[str, Any]:

    # Default response used when Torch is unavailable or import fails.
    torch_resources: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device_count": 0,
        "cuda_current_device_index": None,
        "cuda_devices": [],
        "error": None,
    }

    try:
        # Torch is imported lazily so the API can still boot if Torch is missing or broken.
        import torch

    except Exception as exc:
        # Store the import failure as data instead of failing the whole resource endpoint.
        torch_resources["error"] = str(exc)
        return torch_resources

    # Torch version and CUDA availability are useful worker capability fields.
    cuda_available = torch.cuda.is_available()

    # Store Torch and CUDA runtime metadata.
    torch_resources["torch_available"] = True
    torch_resources["torch_version"] = torch.__version__
    torch_resources["cuda_available"] = cuda_available
    torch_resources["cuda_version"] = torch.version.cuda

    # If CUDA is not visible, return the CPU-only Torch fields.
    if not cuda_available:
        return torch_resources

    # Track the current CUDA device selected by Torch.
    torch_resources["cuda_current_device_index"] = torch.cuda.current_device()

    # Device count can be used by the future coordinator for GPU-aware scheduling.
    cuda_device_count = torch.cuda.device_count()
    torch_resources["cuda_device_count"] = cuda_device_count

    # Collect per-GPU metadata.
    cuda_devices: list[dict[str, Any]] = []

    for device_index in range(cuda_device_count):
        # Device properties include stable static capacity data.
        device_properties = torch.cuda.get_device_properties(device_index)

        # CUDA device capability is useful for comparing GPU feature support across workers.
        device_capability = torch.cuda.get_device_capability(device_index)

        # Dynamic memory fields may fail depending on driver/runtime state.
        memory_info = _get_cuda_memory_info(torch, device_index)

        # Per-device GPU metrics returned to the route.
        cuda_device: dict[str, Any] = {
            "device_index": device_index,
            "name": device_properties.name,
            "total_memory_bytes": device_properties.total_memory,
            "multiprocessor_count": device_properties.multi_processor_count,
            "compute_capability": f"{device_capability[0]}.{device_capability[1]}",
            "free_memory_bytes": memory_info.get("free_memory_bytes"),
            "used_memory_bytes": memory_info.get("used_memory_bytes"),
            "memory_percent_used": memory_info.get("memory_percent_used"),
            "memory_error": memory_info.get("memory_error"),
        }

        cuda_devices.append(cuda_device)

    torch_resources["cuda_devices"] = cuda_devices

    return torch_resources


# Collect dynamic CUDA memory information for one GPU.
# Inputs: imported torch module and CUDA device index.
# Outputs: JSON-serializable memory metrics or an error field.
# This is isolated because CUDA memory queries can fail independently.
def _get_cuda_memory_info(torch_module: Any, device_index: int) -> dict[str, Any]:

    try:
        # mem_get_info returns free and total bytes for the requested CUDA device.
        free_memory_bytes, total_memory_bytes = torch_module.cuda.mem_get_info(device_index)

    except Exception as exc:
        # Do not fail the full endpoint if one dynamic GPU query fails.
        return {
            "free_memory_bytes": None,
            "used_memory_bytes": None,
            "memory_percent_used": None,
            "memory_error": str(exc),
        }

    # Calculate used memory from total minus free.
    used_memory_bytes = total_memory_bytes - free_memory_bytes

    # Return GPU memory usage fields.
    memory_info: dict[str, Any] = {
        "free_memory_bytes": free_memory_bytes,
        "used_memory_bytes": used_memory_bytes,
        "memory_percent_used": _safe_percent(used_memory_bytes, total_memory_bytes),
        "memory_error": None,
    }

    return memory_info


# Collect load averages where the operating system supports them.
# Inputs: none.
# Outputs: JSON-serializable load average values.
# Windows workers may return null values here.
def _get_load_average() -> dict[str, float | None]:

    # Some Windows Python builds do not define os.getloadavg at all.
    if not hasattr(os, "getloadavg"):
        return {
            "one_minute": None,
            "five_minutes": None,
            "fifteen_minutes": None,
        }

    try:
        # Unix-like systems expose 1, 5, and 15 minute load averages.
        load_1m, load_5m, load_15m = os.getloadavg()

    except OSError:
        # Return explicit nulls on platforms where load average is unavailable.
        return {
            "one_minute": None,
            "five_minutes": None,
            "fifteen_minutes": None,
        }

    # Return named load average fields.
    load_average: dict[str, float | None] = {
        "one_minute": load_1m,
        "five_minutes": load_5m,
        "fifteen_minutes": load_15m,
    }

    return load_average


# Calculate a percentage while avoiding divide-by-zero errors.
# Inputs: used value and total value.
# Outputs: rounded percentage or None.
# This keeps endpoint output stable for unusual resource values.
def _safe_percent(used_value: int | float, total_value: int | float) -> float | None:

    if total_value == 0:
        return None

    # Round to two decimals so API output stays readable.
    percent_value = round((used_value / total_value) * 100, 2)

    return percent_value

# Calculate an average while avoiding empty-list errors.
# Inputs: list of numeric values.
# Outputs: rounded average or None.
# This keeps resource endpoint output stable when psutil returns no samples.
def _safe_average(values: list[int | float]) -> float | None:

    if len(values) == 0:
        return None

    # Round to two decimals so API output stays readable.
    average_value = round(sum(values) / len(values), 2)

    return average_value


# Return placeholder job pressure fields before job management exists.
# Inputs: none.
# Outputs: JSON-serializable job pressure snapshot.
# This reserves the coordinator-facing shape without implementing jobs yet.
# TODO: replace placeholder once worker job manager exists
def _get_empty_job_pressure() -> dict[str, Any]:

    # These values are intentionally static until the job manager exists.
    job_pressure: dict[str, Any] = {
        "active_jobs": 0,
        "queued_jobs": 0,
        "worker_accepting_jobs": True,
        "note": "Job pressure is a placeholder until the worker job manager exists.",
    }

    return job_pressure