# test_volunteer_control.py
# Created: 2026-10-14
# Author: Isaac Travers
#
# Covers what a volunteer can see and do while the worker is running (#57):
# the tray's reading of the worker's state, the mapping from its three words
# onto the actions underneath, and the pause that actually suspends the job in
# flight rather than only refusing new work.
#
# What belongs here: anything about volunteer-facing control. What does not:
# the inference itself, which has its own suites.

# json reads the envelope the parent writes for the child.
import json

# os supplies this process's pid, which stands in for a live tray.
import os

# time measures that a pause really blocks, and lets the clock be moved on.
import time

# threading releases a pause from beside the thread that is waiting on it.
import threading

# Path builds the workspace a context and a job process are given.
from pathlib import Path

from marp_inference_worker.installation import operator_control
from marp_inference_worker.installation import tray as tray_module
from marp_inference_worker.jobs.context import ChildJobContext
from marp_inference_worker.jobs.worker_state import WorkerState


# test_pause_suspends_the_job_and_resuming_lets_it_carry_on()
# Verifies R5 and R6 at the tier that can see them.
# Inputs: tmp_path. Output: pytest pass/fail.
#
# This is the requirement the whole issue turns on, and a store-level check
# cannot see it: whether a running job actually stops burning the GPU is a
# property of the frame loop blocking, not of a flag being set. So the test
# blocks a real context on a real pause file and measures that it waited.
def test_pause_suspends_the_job_and_resuming_lets_it_carry_on(tmp_path: Path) -> None:

    stop_file = tmp_path / "STOP"
    pause_file = tmp_path / "PAUSE"
    context = ChildJobContext(
        params={},
        checkpoint_dir=tmp_path / "work",
        stop_file=stop_file,
        pause_file=pause_file,
    )

    # Not paused: the checkpoint must cost nothing, because it runs on the same
    # cadence as the progress report for the whole job.
    assert context.wait_while_paused() is False

    pause_file.touch()
    released_at: list[float] = []

    def release() -> None:
        time.sleep(0.6)
        released_at.append(time.monotonic())
        pause_file.unlink()

    threading.Thread(target=release, daemon=True).start()
    started = time.monotonic()
    waited = context.wait_while_paused()
    returned = time.monotonic()

    # It blocked, and it blocked until the file went away rather than for a
    # fixed time -- which is what makes resuming feel immediate.
    assert waited is True
    assert returned - started >= 0.5
    assert released_at and returned >= released_at[0]

    # And it did not turn the pause into a stop: the job carries on.
    assert context.should_stop() is False


# test_a_paused_job_can_still_be_stopped()
# Verifies that R7 is reachable: a pause held too long becomes a stop.
# Inputs: tmp_path. Output: pytest pass/fail.
#
# Without this the five-minute release would deadlock -- the runner would ask a
# blocked child to stop and the child would never look.
def test_a_paused_job_can_still_be_stopped(tmp_path: Path) -> None:

    stop_file = tmp_path / "STOP"
    pause_file = tmp_path / "PAUSE"
    pause_file.touch()
    context = ChildJobContext(
        params={},
        checkpoint_dir=tmp_path / "work",
        stop_file=stop_file,
        pause_file=pause_file,
    )

    def stop_it() -> None:
        time.sleep(0.5)
        stop_file.touch()

    threading.Thread(target=stop_it, daemon=True).start()

    # Returns even though the pause file is still there, because a stop beats a
    # pause. The engine's own should_stop() then unwinds the job as usual.
    context.wait_while_paused()
    assert pause_file.exists()
    assert context.should_stop() is True


# test_finish_does_not_suspend_the_running_job()
# Verifies pause and finish stayed different things.
# Inputs: none. Output: pytest pass/fail.
#
# Both set the worker's paused flag, so the obvious implementation suspends the
# job for either -- and `finish` means the opposite: let the running job
# complete. This is the regression that would be invisible until somebody lost
# an hour of inference to a Finish up.
def test_finish_does_not_suspend_the_running_job() -> None:

    state = WorkerState()

    state.set_operator_action("finish")
    assert state.is_paused() is True
    assert state.suspended_for() is None

    state.set_operator_action("stop")
    assert state.suspended_for() is None

    state.set_operator_action("pause")
    assert state.suspended_for() is not None


# test_a_coordinator_pause_leaves_the_running_job_alone()
# Verifies the coordinator's pause kept its old meaning.
# Inputs: none. Output: pytest pass/fail.
# A heartbeat saying pause means stop taking work; abandoning a running job is
# a separate action and must not be acquired by accident here.
def test_a_coordinator_pause_leaves_the_running_job_alone() -> None:

    state = WorkerState()
    state.set_paused(True, reason="paused by coordinator")

    assert state.is_paused() is True
    assert state.suspended_for() is None


# test_repeating_pause_does_not_extend_the_wait()
# Verifies the release clock starts once.
# Inputs: none. Output: pytest pass/fail.
# The tray re-sends the current state when the menu is used, so a clock that
# restarted on every write would mean the five-minute release never fired for
# anybody who kept clicking.
def test_repeating_pause_does_not_extend_the_wait() -> None:

    state = WorkerState()
    state.set_operator_action("pause")
    first = state.suspended_for()

    time.sleep(0.05)
    state.set_operator_action("pause")

    assert state.suspended_for() >= first


# test_resuming_clears_the_release_clock()
# Verifies R6: coming back from a pause is a clean slate.
# Inputs: none. Output: pytest pass/fail.
def test_resuming_clears_the_release_clock() -> None:

    state = WorkerState()
    state.set_operator_action("pause")
    state.set_operator_action("running")

    assert state.is_paused() is False
    assert state.suspended_for() is None


# test_pause_is_a_durable_choice()
# Verifies R8: a worker restarted while paused comes back paused.
# Inputs: tmp_path, monkeypatch. Output: pytest pass/fail.
# The volunteer's choice outliving the process is the whole point -- a pause
# that evaporated on the next update would hand their GPU back without asking.
def test_pause_is_a_durable_choice(tmp_path: Path, monkeypatch) -> None:

    monkeypatch.setenv("MARP_WORKER_STATE_DIR", str(tmp_path))

    operator_control.write_action("pause")
    assert operator_control.read_action() == "pause"


# test_the_tray_reads_the_worker_rather_than_its_own_memory()
# Verifies R2: the state shown is the worker's, whoever changed it.
# Inputs: none. Output: pytest pass/fail.
# The tray is not the only thing that can pause this worker -- the command line
# and the coordinator both can -- so an icon that showed the last thing clicked
# would be wrong without ever being noticed.
def test_the_tray_reads_the_worker_rather_than_its_own_memory() -> None:

    assert tray_module.state_from_status({"operator_action": "pause"}) == "paused"
    assert tray_module.state_from_status({"paused": True}) == "paused"
    assert tray_module.state_from_status(
        {"paused": False, "screen_mode": "window"}
    ) == "foreground"
    assert tray_module.state_from_status(
        {"paused": False, "screen_mode": "off"}
    ) == "background"


# test_the_three_states_map_onto_actions_the_worker_already_has()
# Verifies R3 and R4 without a window manager.
# Inputs: none. Output: pytest pass/fail.
#
# Asserts the calls the tray makes, because that is the contract between it and
# the worker. Driving a real notify-icon is not something a test tier here can
# do, and is listed as uncovered in .marp/verification.md rather than faked.
def test_the_three_states_map_onto_actions_the_worker_already_has() -> None:

    sent: list[tuple[str, dict]] = []

    tray = tray_module.WorkerTray(port=8010)
    tray._post = lambda path, body: (sent.append((path, body)), True)[1]

    assert tray.choose("background") is True
    assert sent == [
        ("/status/control", {"action": "resume"}),
        ("/status/screen", {"mode": "off"}),
    ]

    sent.clear()
    assert tray.choose("foreground") is True
    assert sent == [
        ("/status/control", {"action": "resume"}),
        ("/status/screen", {"mode": "window"}),
    ]

    sent.clear()
    assert tray.choose("paused") is True

    # Pause does not touch the screen mode: a volunteer who was watching should
    # still be watching when they come back.
    assert sent == [("/status/control", {"action": "pause"})]


# test_give_me_my_gpu_now_releases_rather_than_suspends()
# Verifies R10.
# Inputs: none. Output: pytest pass/fail.
#
# The distinction this pins down is the reason the menu has two items: a
# suspend keeps the model in VRAM so resuming is free, which is useless to
# somebody about to start a game. `stop` yields the job, so the child exits and
# the memory goes with it.
def test_give_me_my_gpu_now_releases_rather_than_suspends() -> None:

    sent: list[tuple[str, dict]] = []

    tray = tray_module.WorkerTray(port=8010)
    tray._post = lambda path, body: (sent.append((path, body)), True)[1]

    assert tray.release_gpu() is True
    assert sent == [("/status/control", {"action": "stop"})]


# test_the_tray_never_reaches_off_this_machine()
# Verifies the tray talks to loopback and nothing else.
# Inputs: none. Output: pytest pass/fail.
# It controls this machine's worker. A base URL that could be pointed elsewhere
# would make a volunteer's menu a remote control for somebody else's GPU.
def test_the_tray_never_reaches_off_this_machine() -> None:

    assert tray_module.WorkerTray(port=9999)._base == "http://127.0.0.1:9999"


# test_the_envelope_tells_the_child_where_the_pause_file_is()
# Verifies the parent and child agree on the file, which is the whole signal.
# Inputs: tmp_path. Output: pytest pass/fail.
# They are separate processes, so a name agreed in two places is a name that
# can drift. This asserts it travels in the envelope instead.
def test_the_envelope_tells_the_child_where_the_pause_file_is(tmp_path: Path) -> None:

    from marp_inference_worker.jobs.job_process import JobProcess

    job = JobProcess(
        envelope={
            "attempt_id": "attempt-1",
            "worker_id": "worker-1",
            "lease_epoch": 1,
            "spec": {"engine": "tracking"},
        },
        slot_index=0,
        workspace_root=tmp_path,
    )

    assert job.pause_file == job.workspace / "PAUSE"
    assert job.pause_file.parent == job.stop_file.parent

    # Idempotent in both directions: the runner calls these on every pass.
    job.workspace.mkdir(parents=True, exist_ok=True)
    job.request_pause()
    job.request_pause()
    assert job.pause_file.exists()
    job.release_pause()
    job.release_pause()
    assert not job.pause_file.exists()


# _FakeJob
# Stands in for a running child so the release logic can be exercised without
# starting a process. Records what was asked of it, in order.
class _FakeJob:

    def __init__(self, attempt_id: str = "attempt-1") -> None:

        self.attempt_id = attempt_id
        self.stop_requested = False
        self.calls: list[str] = []

    def request_pause(self) -> None:
        self.calls.append("pause")

    def release_pause(self) -> None:
        self.calls.append("release")

    def request_stop(self) -> None:
        self.calls.append("stop")
        self.stop_requested = True


# _runner_with()
# Builds the smallest thing that can run _apply_pause_to_running_jobs.
# Inputs: the state and the jobs. Output: an object with the method bound.
# JobRunner's constructor wants a coordinator, a credential and a workspace;
# none of them are involved here, so the method is bound to a stand-in rather
# than the whole runner being stood up.
def _runner_with(state, jobs):

    from marp_inference_worker.jobs.runner import JobRunner

    class _Stub:
        pass

    stub = _Stub()
    stub._state = state
    stub._jobs_by_slot = jobs
    stub._apply_pause_to_running_jobs = (
        JobRunner._apply_pause_to_running_jobs.__get__(stub, _Stub)
    )
    return stub


# test_pause_reaches_the_job_in_flight()
# Verifies R5 at the runner: the volunteer's choice is carried down to the job.
# Inputs: none. Output: pytest pass/fail.
# Before #57 pause only changed free_slots(), so the job on the GPU kept the
# GPU -- which is the opposite of why somebody presses it.
def test_pause_reaches_the_job_in_flight() -> None:

    state = WorkerState()
    state.set_operator_action("pause")
    job = _FakeJob()

    _runner_with(state, {0: job})._apply_pause_to_running_jobs()

    assert job.calls == ["pause"]
    assert job.stop_requested is False


# test_a_pause_held_too_long_hands_the_attempt_back()
# Verifies R7, including that the job is let go before it is asked to stop.
# Inputs: monkeypatch. Output: pytest pass/fail.
#
# The ordering is the part worth pinning: a suspended child cannot unwind, so
# stopping one without releasing it first would hang until the kill timeout and
# throw away the partial work the stop exists to preserve.
def test_a_pause_held_too_long_hands_the_attempt_back(monkeypatch) -> None:

    from marp_inference_worker.jobs import runner as runner_module

    state = WorkerState()
    state.set_operator_action("pause")

    # Move the clock on rather than waiting five minutes.
    monkeypatch.setattr(state, "suspended_for", lambda: runner_module._PAUSE_RELEASE_S + 1)
    job = _FakeJob()

    _runner_with(state, {0: job})._apply_pause_to_running_jobs()

    assert job.calls == ["release", "stop"]
    assert job.stop_requested is True


# test_resuming_releases_every_job()
# Verifies R6 at the runner.
# Inputs: none. Output: pytest pass/fail.
# Released unconditionally on every pass, so a resume that arrives while the
# runner is busy elsewhere cannot be missed.
def test_resuming_releases_every_job() -> None:

    state = WorkerState()
    jobs = {0: _FakeJob("a"), 1: _FakeJob("b")}

    _runner_with(state, jobs)._apply_pause_to_running_jobs()

    assert [job.calls for job in jobs.values()] == [["release"], ["release"]]


# test_a_job_already_stopping_is_not_suspended()
# Verifies a stop in progress is left to finish.
# Inputs: none. Output: pytest pass/fail.
# Pausing a job that is unwinding blocks the unwind, which loses the partial
# results the cooperative stop exists to keep.
def test_a_job_already_stopping_is_not_suspended() -> None:

    state = WorkerState()
    state.set_operator_action("pause")
    job = _FakeJob()
    job.stop_requested = True

    _runner_with(state, {0: job})._apply_pause_to_running_jobs()

    assert job.calls == []


# test_the_service_pass_actually_applies_the_pause()
# Verifies the wiring, not the logic.
# Inputs: none. Output: pytest pass/fail.
#
# Written after noticing the tests above kept passing when the call was deleted
# from _service_running_jobs: they exercise the method directly, so they prove
# it works and not that anything runs it. A requirement needs a test at the tier
# that can see it fail, and for wiring that tier is the caller.
#
# The jobs mapping is left empty on purpose, so the loop body below the call
# never runs and this stays a test about one line.
def test_the_service_pass_actually_applies_the_pause() -> None:

    from marp_inference_worker.jobs.runner import JobRunner

    class _Stub:
        pass

    applied: list[bool] = []
    stub = _Stub()
    stub._state = WorkerState()
    stub._jobs_by_slot = {}
    stub._apply_pause_to_running_jobs = lambda: applied.append(True)

    # The tail of the pass refreshes what /status reports; stubbed out because
    # this test is about one line near the top and nothing else.
    stub._describe_jobs = lambda: []
    stub._write_inflight = lambda: None

    JobRunner._service_running_jobs(stub)

    assert applied == [True]


# test_the_menu_builds_and_its_actions_are_wired()
# Verifies the menu pystray is actually handed, not just the logic behind it.
# Inputs: none. Output: pytest pass/fail.
#
# Written after a real install came up with no icon and no error. The actions
# were `lambda _icon, _item, key=key:` -- a default argument to capture the loop
# variable -- and pystray inspects an action's signature and refuses anything
# that is not exactly (icon, item). The menu raised ValueError before it was
# built, and the launcher's deliberate swallow (R9) turned that into silence.
#
# Every test here passed throughout, because none of them built the menu. This
# one does, and it needs no desktop: constructing a Menu is pure Python, and
# only run() wants a display and a message pump.
def test_the_menu_builds_and_its_actions_are_wired() -> None:

    tray = tray_module.WorkerTray(port=8010)
    sent: list[tuple[str, dict]] = []
    tray._post = lambda path, body: (sent.append((path, body)), True)[1]

    items = list(tray.build_menu())
    labels = [str(item.text) for item in items]

    assert labels[:3] == ["Background", "Foreground", "Paused"]
    assert labels[-1] == "Give me my GPU now"

    # MenuItem.__call__ takes the icon and dispatches to the action itself, so
    # invoking the item is invoking it exactly as pystray does -- and an action
    # whose signature pystray would reject never gets this far, because the
    # build above raises.
    for item in items:
        if str(item.text) == "Foreground":
            item(None)
    assert ("/status/screen", {"mode": "window"}) in sent

    sent.clear()
    for item in items:
        if str(item.text) == "Give me my GPU now":
            item(None)
    assert sent == [("/status/control", {"action": "stop"})]


# test_the_current_state_is_ticked()
# Verifies R2's "one of which is always shown as the current one".
# Inputs: none. Output: pytest pass/fail.
# The checked callable takes exactly (item) and is rejected the same way if it
# does not, so this covers the other half of the signature contract.
def test_the_current_state_is_ticked() -> None:

    tray = tray_module.WorkerTray(port=8010)
    tray._state = "paused"

    ticked = [str(item.text) for item in tray.build_menu() if item.checked]

    assert ticked == ["Paused"]


# test_every_action_the_tray_sends_is_accepted_by_the_endpoint()
# Verifies the tray and /status/control agree on the words.
# Inputs: none. Output: pytest pass/fail.
#
# The tray sent `running` to resume -- the name the worker *stores* -- and the
# endpoint only accepts `resume` and turns it into that. So Pause worked,
# un-pause was rejected with a 422, and the tray showed nothing because it
# swallows a failed POST. Found by Isaac pausing a real job on 2026-10-14.
#
# Every test above asserted the calls the tray makes, so they all passed: they
# checked the tray against itself. This checks it against the other side, by
# reading the endpoint's own accepted values rather than a copy of them.
def test_every_action_the_tray_sends_is_accepted_by_the_endpoint() -> None:

    from typing import get_args

    from marp_inference_worker.api.status_routes import ControlRequest

    accepted = set(get_args(ControlRequest.model_fields["action"].annotation))

    sent = {action for _label, action, _screen in tray_module._STATES.values()}

    # release_gpu() sends its own action and is not in _STATES, so name it here
    # rather than leaving the one destructive call uncovered.
    sent.add("stop")

    assert sent <= accepted, (
        f"the tray sends {sorted(sent - accepted)}, which /status/control "
        f"rejects; it accepts {sorted(accepted)}"
    )


# test_the_watch_window_is_kept_fed_while_paused()
# Verifies the window survives a pause of any length.
# Inputs: tmp_path. Output: pytest pass/fail.
#
# The watch page closes itself when its long poll goes unanswered -- it retries
# for thirty seconds and then gives up (#52). A pause can last five minutes, so
# without a keepalive the volunteer's machine ends up running a job with no
# window, which is the one thing #57 exists to prevent. Isaac, 2026-10-14, on
# seeing it happen: "it's very important that this doesn't just run without a
# window".
#
# Asserts the callback is actually invoked while blocked, not merely accepted.
def test_the_watch_window_is_kept_fed_while_paused(tmp_path: Path) -> None:

    stop_file = tmp_path / "STOP"
    pause_file = tmp_path / "PAUSE"
    pause_file.touch()
    context = ChildJobContext(
        params={},
        checkpoint_dir=tmp_path / "work",
        stop_file=stop_file,
        pause_file=pause_file,
    )

    fed: list[int] = []

    def release() -> None:
        # Long enough that the keepalive must fire more than once.
        time.sleep(1.2)
        pause_file.unlink()

    threading.Thread(target=release, daemon=True).start()
    context.wait_while_paused(while_waiting=lambda: fed.append(1))

    assert len(fed) >= 2, f"the window was fed {len(fed)} times during a 1.2s pause"


# test_a_keepalive_that_fails_does_not_strand_the_job()
# Verifies the job stays cancellable when the window has gone.
# Inputs: tmp_path. Output: pytest pass/fail.
# The window is a nicety; a paused job that cannot be stopped is not.
def test_a_keepalive_that_fails_does_not_strand_the_job(tmp_path: Path) -> None:

    stop_file = tmp_path / "STOP"
    pause_file = tmp_path / "PAUSE"
    pause_file.touch()
    context = ChildJobContext(
        params={},
        checkpoint_dir=tmp_path / "work",
        stop_file=stop_file,
        pause_file=pause_file,
    )

    def explode() -> None:
        raise RuntimeError("the watch window has gone")

    threading.Thread(
        target=lambda: (time.sleep(0.5), stop_file.touch()), daemon=True
    ).start()

    context.wait_while_paused(while_waiting=explode)

    assert context.should_stop() is True


# test_a_second_launch_does_not_add_a_second_icon()
# Verifies the tray's single-instance lock.
# Inputs: tmp_path. Output: pytest pass/fail.
# A volunteer opening the shortcut twice should get one icon, not two.
def test_a_second_launch_does_not_add_a_second_icon(tmp_path: Path) -> None:

    from marp_inference_worker.installation import launcher

    assert launcher._tray_is_running(tmp_path) is False

    # The lock names this process, which is alive and is python.
    (tmp_path / "tray.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert launcher._tray_is_running(tmp_path) is True

    assert launcher._start_tray(8010, tmp_path) is None


# test_a_dead_lock_does_not_stop_the_icon_coming_back()
# Verifies a stale pid file is ignored.
# Inputs: tmp_path. Output: pytest pass/fail.
#
# A launcher that was killed cannot clean up after itself, so the lock outlives
# it. Treating a stale lock as "a tray is running" would mean no icon at all,
# which is the failure this whole feature exists to avoid -- strictly worse
# than the duplicate icon the lock is there to prevent.
def test_a_dead_lock_does_not_stop_the_icon_coming_back(tmp_path: Path) -> None:

    from marp_inference_worker.installation import launcher

    # A pid that cannot be running: 0 is not a real process on Windows, and
    # garbage must be treated the same way.
    for content in ("0", "", "not-a-pid", "999999999"):
        (tmp_path / "tray.pid").write_text(content, encoding="utf-8")
        assert launcher._tray_is_running(tmp_path) is False, content
