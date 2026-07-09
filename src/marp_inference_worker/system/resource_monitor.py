# File name: resource_monitor.py
# Date created: 2026-07-08
# Author: Isaac Travers
# Purpose: Collects local worker machine resource information.
# System role: Provides system state data for API routes and future coordinator decisions.
# Code in this file should inspect local resources but should not schedule jobs.

# os provides operating-system features such as load average checks where supported.
import os

# platform provides host, operating system, and processor metadata.
import platform

# shutil provides cross-platform disk usage inspection.
import shutil

# sys provides Python runtime version and executable metadata.
import sys

# Path provides cross-platform filesystem path handling.
from pathlib import Path

# Any supports flexible JSON-serializable resource dictionaries.
from typing import Any

# Callable supports typed helper functions that safely call optional NVML methods.
from typing import Callable

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
        "nvidia": _get_nvidia_resources(),
        "job_pressure": _get_empty_job_pressure(),
    }

    return resources

# Collect NVIDIA GPU telemetry through NVML when available.
# Inputs: none.
# Outputs: JSON-serializable NVIDIA driver and GPU telemetry.
# This complements Torch CUDA visibility with live hardware status.
def _get_nvidia_resources() -> dict[str, Any]:

    # Default response used when NVML is unavailable or no NVIDIA driver is present.
    nvidia_resources: dict[str, Any] = {
        "nvml_available": False,
        "driver_version": None,
        "device_count": 0,
        "devices": [],
        "error": None,
    }

    try:
        # pynvml is imported lazily so non-NVIDIA systems can still run the API.
        import pynvml

    except Exception as exc:
        # Store import failure as data instead of failing the endpoint.
        nvidia_resources["error"] = str(exc)
        return nvidia_resources

    try:
        # Initialize NVML before querying driver or device telemetry.
        pynvml.nvmlInit()

    except Exception as exc:
        # NVML may be installed even when the NVIDIA driver is unavailable.
        nvidia_resources["error"] = str(exc)
        return nvidia_resources

    try:
        # Driver version and device count describe the NVIDIA runtime visible to this worker.
        driver_version = _decode_nvml_value(pynvml.nvmlSystemGetDriverVersion())
        device_count = pynvml.nvmlDeviceGetCount()

        # Store top-level NVML capability fields.
        nvidia_resources["nvml_available"] = True
        nvidia_resources["driver_version"] = driver_version
        nvidia_resources["device_count"] = device_count

        # Collect per-device NVIDIA telemetry.
        devices: list[dict[str, Any]] = []

        for device_index in range(device_count):
            # NVML handle is used for all device-specific queries.
            device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

            # Per-device telemetry returned to the API.
            device_resources = _get_nvidia_device_resources(pynvml, device_handle, device_index)

            devices.append(device_resources)

        # Store the completed device telemetry list.
        nvidia_resources["devices"] = devices

    except Exception as exc:
        # Return any partial data collected before the failure.
        nvidia_resources["error"] = str(exc)

    finally:
        # Shut down NVML so the request does not leave the library initialized.
        _safe_nvml_call(pynvml.nvmlShutdown)

    return nvidia_resources


# Collect NVIDIA telemetry for one NVML device handle.
# Inputs: pynvml module, NVML device handle, and device index.
# Outputs: JSON-serializable per-GPU telemetry dictionary.
# This isolates optional hardware queries that may vary by platform.
def _get_nvidia_device_resources(pynvml_module: Any, device_handle: Any, device_index: int) -> dict[str, Any]:

    # Static device identity fields.
    device_name = _decode_nvml_value(_safe_nvml_call(pynvml_module.nvmlDeviceGetName, device_handle))
    device_uuid = _decode_nvml_value(_safe_nvml_call(pynvml_module.nvmlDeviceGetUUID, device_handle))

    # PCI information identifies the physical GPU location.
    pci_info = _safe_nvml_call(pynvml_module.nvmlDeviceGetPciInfo, device_handle)
    pci_bus_id = _decode_nvml_value(getattr(pci_info, "busId", None))

    # Memory telemetry from NVML should roughly match nvidia-smi.
    memory_info = _safe_nvml_call(pynvml_module.nvmlDeviceGetMemoryInfo, device_handle)

    # GPU and memory utilization are core scheduling signals.
    utilization_info = _safe_nvml_call(pynvml_module.nvmlDeviceGetUtilizationRates, device_handle)

    # Temperature, fan, power, and clocks are useful health and load signals.
    temperature_c = _get_nvml_temperature_c(pynvml_module, device_handle)
    fan_speed_percent = _safe_nvml_call(pynvml_module.nvmlDeviceGetFanSpeed, device_handle)
    power_usage_watts = _get_nvml_power_usage_watts(pynvml_module, device_handle)
    power_limit_watts = _get_nvml_power_limit_watts(pynvml_module, device_handle)
    graphics_clock_mhz = _get_nvml_clock_mhz(pynvml_module, device_handle, pynvml_module.NVML_CLOCK_GRAPHICS)
    memory_clock_mhz = _get_nvml_clock_mhz(pynvml_module, device_handle, pynvml_module.NVML_CLOCK_MEM)

    # Per-device NVIDIA resource snapshot.
    device_resources: dict[str, Any] = {
        "device_index": device_index,
        "name": device_name,
        "uuid": device_uuid,
        "pci_bus_id": pci_bus_id,
        "temperature_c": temperature_c,
        "fan_speed_percent": fan_speed_percent,
        "power_usage_watts": power_usage_watts,
        "power_limit_watts": power_limit_watts,
        "graphics_clock_mhz": graphics_clock_mhz,
        "memory_clock_mhz": memory_clock_mhz,
        "gpu_utilization_percent": getattr(utilization_info, "gpu", None),
        "memory_utilization_percent": getattr(utilization_info, "memory", None),
        "memory_total_bytes": getattr(memory_info, "total", None),
        "memory_free_bytes": getattr(memory_info, "free", None),
        "memory_used_bytes": getattr(memory_info, "used", None),
        "memory_percent_used": _safe_percent(
            getattr(memory_info, "used", 0),
            getattr(memory_info, "total", 0),
        ),
    }

    return device_resources


# Safely call one NVML function and return None on unsupported fields.
# Inputs: callable NVML function and optional positional arguments.
# Outputs: NVML result or None.
# This keeps one unsupported telemetry field from failing the whole endpoint.
def _safe_nvml_call(function_to_call: Callable[..., Any], *args: Any) -> Any | None:

    try:
        # Call the NVML function with the supplied arguments.
        result = function_to_call(*args)

    except Exception:
        # Many NVML fields are platform, driver, or device dependent.
        return None

    return result


# Decode NVML byte values into strings when needed.
# Inputs: value returned by NVML.
# Outputs: decoded string, original value, or None.
# This keeps API output JSON-friendly across pynvml versions.
def _decode_nvml_value(value: Any) -> Any:

    if isinstance(value, bytes):
        # NVML may return bytes for names, UUIDs, and driver versions.
        return value.decode("utf-8", errors="replace")

    return value


# Read GPU temperature from NVML when supported.
# Inputs: pynvml module and NVML device handle.
# Outputs: temperature in Celsius or None.
# This isolates the NVML temperature constant from the main device collector.
def _get_nvml_temperature_c(pynvml_module: Any, device_handle: Any) -> int | None:

    # Temperature is reported using the GPU temperature sensor.
    temperature_c = _safe_nvml_call(
        pynvml_module.nvmlDeviceGetTemperature,
        device_handle,
        pynvml_module.NVML_TEMPERATURE_GPU,
    )

    return temperature_c


# Read GPU power usage from NVML when supported.
# Inputs: pynvml module and NVML device handle.
# Outputs: power draw in watts or None.
# NVML reports power usage in milliwatts.
def _get_nvml_power_usage_watts(pynvml_module: Any, device_handle: Any) -> float | None:

    # NVML reports power draw in milliwatts.
    power_usage_milliwatts = _safe_nvml_call(pynvml_module.nvmlDeviceGetPowerUsage, device_handle)

    if power_usage_milliwatts is None:
        return None

    # Convert milliwatts to watts.
    power_usage_watts = round(power_usage_milliwatts / 1000.0, 2)

    return power_usage_watts


# Read GPU power limit from NVML when supported.
# Inputs: pynvml module and NVML device handle.
# Outputs: power limit in watts or None.
# NVML reports power limit in milliwatts.
def _get_nvml_power_limit_watts(pynvml_module: Any, device_handle: Any) -> float | None:

    # NVML reports power limit in milliwatts.
    power_limit_milliwatts = _safe_nvml_call(pynvml_module.nvmlDeviceGetPowerManagementLimit, device_handle)

    if power_limit_milliwatts is None:
        return None

    # Convert milliwatts to watts.
    power_limit_watts = round(power_limit_milliwatts / 1000.0, 2)

    return power_limit_watts


# Read one GPU clock value from NVML when supported.
# Inputs: pynvml module, NVML device handle, and NVML clock type.
# Outputs: clock speed in MHz or None.
# This supports graphics and memory clock telemetry.
def _get_nvml_clock_mhz(pynvml_module: Any, device_handle: Any, clock_type: int) -> int | None:

    # NVML returns the current clock speed in MHz.
    clock_mhz = _safe_nvml_call(pynvml_module.nvmlDeviceGetClockInfo, device_handle, clock_type)

    return clock_mhz


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