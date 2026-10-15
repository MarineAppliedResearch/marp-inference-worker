"""Stable installed-worker launcher with health-gated switch and rollback."""

import json
import argparse
import os
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath

import httpx


def _read_json(path: Path) -> dict:
    # Windows PowerShell 5.1 writes `-Encoding utf8` with a BOM. Setup JSON is
    # read here before the worker can start, so accept both Windows and plain UTF-8.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _entrypoint(version_dir: Path) -> Path:
    manifest = _read_json(version_dir / "manifest.json")
    relative = PurePosixPath(str(manifest.get("entrypoint", "marp-worker.exe")).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("release manifest has an unsafe entrypoint")
    executable = version_dir.joinpath(*relative.parts)
    if not executable.is_file():
        raise FileNotFoundError(f"worker entrypoint is missing: {executable}")
    return executable


def _start(version: str, install_root: Path, state_dir: Path, config: dict) -> subprocess.Popen:
    executable = _entrypoint(install_root / "versions" / version)
    environment = os.environ.copy()
    environment["MARP_WORKER_STATE_DIR"] = str(state_dir)
    environment["MARP_WORKER_INSTALL_ROOT"] = str(install_root)
    environment["MARP_COORDINATOR_URL"] = str(config["coordinator_url"])
    environment["MARP_WORKER_API_PORT"] = str(config.get("api_port", 8010))
    return subprocess.Popen(
        [str(executable), "--screen", str(config.get("screen", "off"))],
        cwd=executable.parent,
        env=environment,
    )


def _healthy(process: subprocess.Popen, port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def _set_running_screen_mode(port: int, mode: str) -> bool:
    """Update the sign-in worker instead of launching a duplicate instance."""
    try:
        response = httpx.post(
            f"http://127.0.0.1:{port}/status/screen",
            json={"mode": mode},
            timeout=2,
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


# _tray_is_running()
# Whether this machine already has a tray icon up.
# Inputs: the state directory.
# Output: True when a live process holds the lock.
#
# The lock is a pid file rather than a flag, because a launcher that was killed
# cannot clean up after itself -- and a stale flag would mean no icon at all,
# which is the failure this whole file exists to avoid. A pid that no longer
# exists, or that belongs to something else now, is treated as no tray.
def _tray_is_running(state_dir: Path) -> bool:

    lock = state_dir / "tray.pid"
    try:
        pid = int(lock.read_text(encoding="utf-8-sig").strip())
    except (OSError, ValueError):
        return False

    try:
        import psutil

        return psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower()
    except Exception:
        # Cannot tell. Say no: a second icon is untidy, no icon is the bug.
        return False


# _start_tray()
# Puts the worker's icon in the system tray, if this machine has one.
# Inputs: the worker's API port and the state directory.
# Output: the tray's thread, or None when one is already running.
#
# Runs on its own daemon thread so the supervision loop below is untouched by
# it. Every failure is swallowed deliberately (#57 R9): a headless machine, a
# Linux box with no status area, or a missing pystray must leave the worker
# running exactly as it did before. An icon is not worth a worker.
#
# Called on both paths out of main(), including the one that finds a worker
# already running and only retargets its screen. That path used to return
# before this, so a volunteer who opened the shortcut while the worker was up
# got no icon at all -- the exact complaint #57 was filed about.
def _start_tray(port: int, state_dir: Path):

    if _tray_is_running(state_dir):
        return None

    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "tray.pid").write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        # Without the lock a second shortcut could add a second icon. Still
        # better than no icon, so carry on.
        pass

    def run() -> None:
        try:
            from marp_inference_worker.installation.tray import WorkerTray

            WorkerTray(port).run()
        except Exception:
            # Nowhere useful to report this: the worker's own log belongs to the
            # child process, and a volunteer cannot act on it either way.
            pass

    thread = threading.Thread(target=run, daemon=True, name="marp-tray")
    thread.start()
    return thread


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the active MARP worker release")
    parser.add_argument("--screen", choices=("off", "window", "fullscreen"))
    parser.add_argument("--control", choices=("pause", "finish", "stop", "resume"))

    # The tray is on by default: a volunteer with no sign the worker is running
    # is the whole of #57. The switch is for a service account or a headless box
    # where there is nothing to put an icon in.
    parser.add_argument("--no-tray", action="store_true")
    args = parser.parse_args()
    local_app_data = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    install_root = Path(os.environ.get("MARP_WORKER_INSTALL_ROOT") or local_app_data / "MARP" / "Worker")
    state_dir = Path(os.environ.get("MARP_WORKER_STATE_DIR") or install_root / "state")
    config = _read_json(install_root / "config.json")
    if args.screen:
        config["screen"] = args.screen
    active_path = install_root / "active.json"
    pending_path = state_dir / "pending-update.json"
    result_path = state_dir / "switch-result.json"
    port = int(config.get("api_port", 8010))

    if args.control:
        response = httpx.post(
            f"http://127.0.0.1:{port}/status/control",
            json={"action": args.control},
            timeout=5,
        )
        response.raise_for_status()
        return 0
    # A worker is already up: retarget its screen, make sure it has an icon,
    # and stay alive only for as long as the tray needs. Returning straight
    # away here is what left a running worker with no icon.
    if args.screen and _set_running_screen_mode(port, args.screen):
        if args.no_tray:
            return 0

        tray = _start_tray(port, state_dir)
        if tray is None:
            return 0

        # Wait on the thread itself, not on the lock: the lock holds this
        # process's own pid, so reading it back would say "running" forever
        # even once the tray had gone.
        tray.join()
        return 0

    if not args.no_tray:
        _start_tray(port, state_dir)

    while True:
        active = _read_json(active_path)
        pending = _read_json(pending_path) if pending_path.is_file() else None
        previous = str(pending.get("previous_release") or active["release"]) if pending else str(active["release"])
        selected = str(pending["target_release"]) if pending else previous

        if pending:
            _write_json(active_path, {"release": selected})
        process = _start(selected, install_root, state_dir, config)

        if pending:
            if _healthy(process, port):
                pending_path.unlink(missing_ok=True)
                _write_json(result_path, {"update_state": "succeeded", "version": selected})
            else:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                _write_json(active_path, {"release": previous})
                pending_path.unlink(missing_ok=True)
                _write_json(result_path, {
                    "update_state": "rolled_back",
                    "version": previous,
                    "message": f"version {selected} did not acknowledge health",
                })
                process = _start(previous, install_root, state_dir, config)

        exit_code = process.wait()
        if exit_code == 75 and pending_path.is_file():
            continue
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
