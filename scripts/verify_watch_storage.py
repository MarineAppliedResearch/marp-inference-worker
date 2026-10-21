# verify_watch_storage.py
# Created: 2026-10-21
# Author: Isaac Travers
#
# Manual verification for marp-inference-worker#65: a worker's disk use is flat.
#
# Two things cannot be settled by pytest and are settled here, against a real
# Chromium on a real desktop:
#   - the window still opens and draws with the flags that stop the background
#     fetching, which is the part a unit test cannot observe at all;
#   - what a profile actually costs, in bytes, after the browser has had long
#     enough to fetch what it would have fetched.
#
# The third part drives the real JobRunner through a real child process, so the
# workspace lifecycle is shown rather than asserted. That part is also covered
# by tests/test_job_runner.py; it is here because seeing the directories appear
# and go is what the issue is about.
#
# Run from the repository root:
#   .venv312\Scripts\python.exe scripts\verify_watch_storage.py
#
# It opens two browser windows, one after the other, for about a minute each.

# json prints the comparison at the end.
import json

# os walks a profile to size it.
import os

# shutil removes the scratch state directory afterwards.
import shutil

# sys puts the repository's own src and tests on the path.
import sys

# tempfile holds the scratch worker state.
import tempfile

# time waits for Chromium to settle before the profile is measured.
import time

# Path types everything on disk.
from pathlib import Path

# SimpleNamespace stands in for a decoded frame, which only needs `.image`.
from types import SimpleNamespace

# numpy makes the frame the window draws.
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from marp_inference_worker.watch import display as display_module  # noqa: E402
from marp_inference_worker.watch.display import WatchDisplay  # noqa: E402


# How long a window is left up before its profile is measured. Chromium fetches
# its component set over the first minute or so, so measuring at ten seconds
# would flatter the result by reading a profile that had not finished growing.
_SETTLE_S = 75.0


# _size_mb()
# How much a directory holds, in megabytes.
# Inputs: the directory.
# Output: megabytes, one decimal place.
def _size_mb(path: Path) -> float:

    total = 0
    for directory, _subdirs, names in os.walk(path):
        for name in names:
            try:
                total += (Path(directory) / name).stat().st_size
            except OSError:
                pass
    return round(total / (1024 * 1024), 1)


# _biggest()
# The largest things in a profile, so a number has an explanation beside it.
# Inputs: the profile directory and how many entries to name.
# Output: a list of (name, megabytes), largest first.
def _biggest(profile: Path, count: int = 5) -> list:

    entries = []
    for entry in profile.iterdir():
        if entry.is_dir():
            entries.append((entry.name, _size_mb(entry)))
        else:
            entries.append((entry.name, round(entry.stat().st_size / 1048576, 1)))
    return sorted(entries, key=lambda pair: pair[1], reverse=True)[:count]


# _show_a_window()
# Opens one real watch window in a slot, draws to it, and closes it.
# Inputs: the workspace for this attempt, the profile root, the slot, a label.
# Output: the profile directory Chromium was given.
#
# Returns the profile rather than the display so the caller can measure it
# after the window has gone; the point of the exercise is what stays on disk.
def _show_a_window(
    workspace: Path, profile_root: Path, slot: int, label: str, baseline: bool = False
) -> Path:

    workspace.mkdir(parents=True, exist_ok=True)

    # `baseline` puts the flags back as they were before #65, so the comparison
    # is made by this script rather than quoted from somewhere. Done by
    # stripping what was added rather than by keeping a second copy of the
    # argument list, which would drift from the real one the first time
    # somebody edited it.
    if baseline:
        real_args = display_module._chromium_args

        def without_the_new_flags(*args, **kwargs):
            built = real_args(*args, **kwargs)
            kept = [
                argument for argument in built
                if not argument.startswith("--disable-background-networking")
                and not argument.startswith("--disable-component-update")
                and not argument.startswith("--disk-cache-size=")
                and not argument.startswith("--hide-crash-restore-bubble")
                and not argument.startswith("--no-default-browser-check")
                and not argument.startswith("--disable-features=")
            ]
            return kept + ["--disable-features=CalculateNativeWinOcclusion"]

        display_module._chromium_args = without_the_new_flags

    display = WatchDisplay(
        screen_mode="window",
        workspace=workspace,
        warn=lambda message: print("   [warn] " + message),
        job_id=label,
        attempt_id=label,
        model_name="rockfish5",
        species_names=["Lingcod", "Vermilion Rockfish"],
        slot_index=slot,
        slot_count=1,
        profile_root=profile_root,
    )

    if not display.start():
        raise SystemExit("the watch window would not start for " + label)

    profile = Path(display._profile_dir)
    print("   profile      " + str(profile))

    # Draw something, so this proves the window works rather than merely that a
    # process launched. A window that opens and never paints is the failure the
    # new flags could plausibly cause.
    frame = SimpleNamespace(image=np.full((360, 640, 3), 40, dtype=np.uint8))
    print("   drew a frame " + str(display.present(frame, 1, [], range_start=0, range_end=100)))

    print("   settling for " + str(int(_SETTLE_S)) + "s while Chromium fetches what it will...")
    deadline = time.monotonic() + _SETTLE_S
    while time.monotonic() < deadline:
        display.present(frame, int(time.monotonic()), [], range_start=0, range_end=100)
        time.sleep(1.0)

    display.close()
    time.sleep(3.0)

    if baseline:
        display_module._chromium_args = real_args

    return profile


# _workspace_lifecycle()
# Drives the real runner through a mock job and reports what survives.
# Inputs: the scratch directory.
# Output: none; prints what it saw.
#
# Uses the test suite's fake coordinator rather than a second copy of one: it
# already records every call the worker makes and is the thing the runner tests
# drive, so a divergence here would be a divergence from what is tested.
def _workspace_lifecycle(scratch: Path) -> None:

    # By path, because `tests/` is a plain directory with no `__init__.py`.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "marp_test_job_runner", _ROOT / "tests" / "test_job_runner.py"
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    FakeCoordinator, _job_for, _runner = (
        harness.FakeCoordinator, harness._job_for, harness._runner
    )

    state_dir = scratch / "state"
    jobs_dir = state_dir / "jobs"

    # Three workspaces from runs that are long over, as a machine upgrading
    # into this change would have.
    for attempt in ("attempt-old-1", "attempt-old-2", "attempt-old-3"):
        leftover = jobs_dir / attempt / "chromium-profile"
        leftover.mkdir(parents=True, exist_ok=True)
        (leftover / "Cookies").write_bytes(b"x" * 4096)

    print("   before start  " + str(sorted(p.name for p in jobs_dir.iterdir())))

    coordinator = FakeCoordinator(offers=[_job_for(scratch, "attempt-demo", frames=5)])
    runner, state = _runner(coordinator, scratch)

    # The real start sequence, with the loop told to exit at once.
    runner._check_for_update = lambda force=False: None
    runner._should_exit = True
    runner.run_forever()

    print("   after prune   " + str(sorted(p.name for p in jobs_dir.iterdir())))

    # Now one real job, in a real child process.
    runner._should_exit = False
    runner.ensure_enrolled()
    if runner._poll_once() is not True:
        raise SystemExit("the runner would not take the demo job")

    workspace = Path(runner._jobs_by_slot[0].workspace)
    print("   job running   " + str(sorted(p.name for p in jobs_dir.iterdir())))

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not coordinator.results:
        runner._service_running_jobs()
        time.sleep(0.05)

    print("   reported      " + str(coordinator.results[0]["outcome"]))
    print("   after result  " + str(sorted(p.name for p in jobs_dir.iterdir())))
    print("   workspace survived: " + str(workspace.exists()))
    print("   last notice   " + str(state.describe()["last_notice"]))


# How long two concurrent windows are left up. Shorter than the settle above:
# this part is about whether both exist and stay independent, and the size
# question is already answered by parts 0 to 2.
_CONCURRENT_S = 30.0


# _two_slots_at_once()
# Opens two watch windows at the same time and checks they stay separate.
# Inputs: the profile root to put both profiles under.
# Output: none; prints what it saw, and raises if the rule is broken.
#
# This is the case the per-slot decision exists for, and the one a single-slot
# run cannot see. Chromium will not start a second process against a
# `--user-data-dir` another one holds -- it signals the holder and exits -- so
# two windows sharing a profile would be one process owning both, and
# `_kill_by_profile()` would take down a running job's window when the other
# job ended. Two profiles is what stops that, and this is where it is checked.
def _two_slots_at_once(profile_root: Path, workspace_root: Path) -> dict:

    displays = []
    for slot in (0, 1):
        workspace = workspace_root / ("attempt-slot-" + str(slot))
        workspace.mkdir(parents=True, exist_ok=True)
        display = WatchDisplay(
            screen_mode="window",
            workspace=workspace,
            warn=lambda message: print("   [warn] " + message),
            job_id="slot-" + str(slot),
            attempt_id="attempt-slot-" + str(slot),
            model_name="rockfish5",
            species_names=["Lingcod", "Vermilion Rockfish"],
            slot_index=slot,
            slot_count=2,
            profile_root=profile_root,
        )
        if not display.start():
            raise SystemExit("the watch window would not start for slot " + str(slot))
        displays.append(display)

    first, second = displays

    # Two profiles, and two processes. A single process owning both windows is
    # exactly the failure this is looking for.
    print("   slot 0 profile  " + first._profile_dir.name)
    print("   slot 1 profile  " + second._profile_dir.name)
    print("   different       " + str(first._profile_dir != second._profile_dir))
    print("   two processes   " + str(first._process.pid != second._process.pid)
          + "  (" + str(first._process.pid) + ", " + str(second._process.pid) + ")")

    # Both drawing, side by side, for long enough to look at.
    frames = [
        SimpleNamespace(image=np.full((360, 640, 3), shade, dtype=np.uint8))
        for shade in (40, 90)
    ]
    print("   both drawing for " + str(int(_CONCURRENT_S)) + "s -- two windows should be on screen")
    deadline = time.monotonic() + _CONCURRENT_S
    while time.monotonic() < deadline:
        for display, frame in zip(displays, frames):
            display.present(frame, int(time.monotonic()), [], range_start=0, range_end=100)
        time.sleep(1.0)

    print("   slot 0 window alive " + str(first._profile_in_use()))
    print("   slot 1 window alive " + str(second._profile_in_use()))

    # The part that matters: one job ending must not close the other's window.
    # Under one shared profile it would, because the kill matches on the
    # profile path and both windows would be behind the same one.
    print("   closing slot 0 only...")
    first.close()
    time.sleep(5.0)

    still_up = second._profile_in_use()
    print("   slot 0 window alive " + str(first._profile_in_use()))
    print("   slot 1 window alive " + str(still_up))
    print("   slot 1 still draws  " + str(
        second.present(frames[1], 999, [], range_start=0, range_end=100)))

    second.close()
    time.sleep(3.0)

    if not still_up:
        raise SystemExit("closing slot 0 took slot 1's window with it")

    return {
        "two_profiles": first._profile_dir != second._profile_dir,
        "two_processes": first._process.pid != second._process.pid,
        "closing_one_left_the_other_up": still_up,
    }


def main() -> int:

    scratch = Path(tempfile.mkdtemp(prefix="marp-65-"))
    profile_root = scratch / "state" / "chromium-profiles"

    print("scratch: " + str(scratch) + "\n")

    try:
        print("0. BASELINE -- one job with the flags as they were before #65")
        was = _show_a_window(
            scratch / "baseline" / "jobs" / "attempt-0",
            scratch / "baseline" / "chromium-profiles", 0, "baseline", baseline=True,
        )
        baseline_mb = _size_mb(was)
        print("   profile is   " + str(baseline_mb) + " MB")
        for name, size in _biggest(was):
            print("      " + str(size).rjust(7) + " MB  " + name)
        print("")

        print("1. First job in slot 0 -- a profile that has never existed")
        first = _show_a_window(scratch / "state" / "jobs" / "attempt-a", profile_root, 0, "attempt-a")
        first_mb = _size_mb(first)
        print("   profile is   " + str(first_mb) + " MB")
        for name, size in _biggest(first):
            print("      " + str(size).rjust(7) + " MB  " + name)

        print("\n2. Second job in slot 0 -- the same profile, reused")
        second = _show_a_window(scratch / "state" / "jobs" / "attempt-b", profile_root, 0, "attempt-b")
        second_mb = _size_mb(second)
        print("   profile is   " + str(second_mb) + " MB")
        print("   same directory as the first: " + str(first == second))

        print("\n3. Where it all lives")
        for entry in sorted((scratch / "state").iterdir()):
            print("   " + str(_size_mb(entry)).rjust(7) + " MB  state/" + entry.name)
        print("   profiles on this machine: " + str(sorted(p.name for p in profile_root.iterdir())))

        print("\n4. The workspace lifecycle, through the real runner")
        _workspace_lifecycle(scratch)

        print("")
        print("5. Two slots at once -- the case per-slot profiles exist for")
        concurrent = _two_slots_at_once(profile_root, scratch / "state" / "jobs")
        print("   profiles now    " + str(sorted(p.name for p in profile_root.iterdir())))
        for entry in sorted(profile_root.iterdir()):
            print("      " + str(_size_mb(entry)).rjust(7) + " MB  " + entry.name)

        print("\n" + json.dumps({
            "profile_with_the_old_flags_mb": baseline_mb,
            "profile_after_first_job_mb": first_mb,
            "profile_after_second_job_mb": second_mb,
            "profile_is_per_slot": first == second,
            "profile_outside_workspace": profile_root in first.parents,
            **concurrent,
        }, indent=2))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
