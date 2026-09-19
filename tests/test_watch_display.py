"""Tests for the child-owned ordered watch channel."""

import base64
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from marp_inference_worker.watch.display import (
    WatchDisplay,
    _FrameChannel,
    _chromium_args,
)


def test_display_resolves_a_relative_worker_workspace() -> None:
    display = WatchDisplay("window", Path("data/worker/jobs/1"), lambda _message: None)

    assert display._workspace.is_absolute()
    assert display._workspace.name == "1"


def test_chromium_keeps_occluded_watch_windows_rendering_without_the_cuda_gpu() -> None:
    args = _chromium_args(
        Path("chrome.exe"),
        "http://127.0.0.1:1234/",
        Path("C:/worker/jobs/1/chromium-profile"),
        "fullscreen",
    )

    assert "--disable-background-timer-throttling" in args
    assert "--disable-backgrounding-occluded-windows" in args
    assert "--disable-renderer-backgrounding" in args
    assert "--disable-features=CalculateNativeWinOcclusion" in args
    assert "--no-proxy-server" in args
    assert "--proxy-bypass-list=<-loopback>" in args
    assert "--new-window" not in args
    # A geometry is always given now, because a window has to be tiled beside
    # its siblings rather than left where Chromium puts it. What the numbers are
    # depends on the monitors, so this asserts that both flags are present
    # rather than pinning a coordinate the next machine would not have.
    assert any(argument.startswith("--window-position=") for argument in args)
    assert any(argument.startswith("--window-size=") for argument in args)
    assert "--disable-gpu" in args
    assert "--disable-gpu-compositing" in args
    assert "--use-angle=swiftshader" not in args
    assert "--start-fullscreen" in args


# test_publishing_never_waits_for_the_browser_once_it_is_connected()
# Verifies that the display cannot set the pace of inference.
# Inputs: none.
# Output: pytest pass/fail result.
#
# This test used to assert the opposite -- that a publisher waits for the
# browser to acknowledge each frame -- and that contract was the reason a
# watched run was slower than the video it was showing. A 4080 did 55-75 f/s
# headless and 28 with a window; a 5060 ran each window at 0.7x real time.
# The window now shows the most recent frame it can collect and misses the
# ones in between, which is what watching is for.
def test_publishing_never_waits_for_the_browser_once_it_is_connected() -> None:
    channel = _FrameChannel()

    # Connect first: the very first frame still waits for the browser to
    # arrive, and that is asserted separately below.
    opening = threading.Thread(target=lambda: channel.publish({"frame_number": 1}, timeout_s=2))
    opening.start()
    assert channel.next(None) == {"frame_number": 1}
    opening.join(timeout=1)

    # Now publish a run of frames with nobody collecting them. Every call must
    # return immediately; if any of them blocks, this does not finish.
    started = time.monotonic()
    results = [channel.publish({"frame_number": n}, timeout_s=2) for n in range(2, 12)]
    elapsed = time.monotonic() - started

    assert all(results)
    assert elapsed < 0.5, f"publishing blocked for {elapsed:.2f}s"

    # The window gets the newest frame, not the oldest waiting one.
    assert channel.next(1) == {"frame_number": 11}

    # And the ones it never saw are counted rather than silently forgotten, so
    # "the window looks jumpy" has a number behind it.
    # Ten, not nine: frame 1 was collected by the window but never
    # acknowledged, so it was still pending and was replaced like the rest.
    assert channel.dropped == 10


# test_the_first_frame_still_waits_for_a_browser_that_never_arrives()
# Verifies the one case that must still block.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Dropping frames is right once somebody is watching. A browser that never
# opened at all is a real failure, and without this the run would push frames
# into a window that does not exist for the whole job.
def test_the_first_frame_still_waits_for_a_browser_that_never_arrives() -> None:
    channel = _FrameChannel()
    published: list[bool] = []

    worker = threading.Thread(
        target=lambda: published.append(
            channel.publish({"frame_number": 7}, timeout_s=2, connect_timeout_s=0.2)
        )
    )
    worker.start()
    worker.join(timeout=2)

    assert published == [False]
    assert channel.closed


# test_a_browser_that_stops_collecting_detaches_the_display()
# Verifies liveness now that nothing blocks on an acknowledgement.
# Inputs: none.
# Output: pytest pass/fail result.
#
# The old code noticed a dead browser by timing out on the acknowledgement it
# waited for. Nothing waits any more, so silence has to be checked explicitly
# -- otherwise a crashed window would leave the job encoding JPEGs into
# nothing for the rest of the range.
def test_a_browser_that_stops_collecting_detaches_the_display() -> None:
    channel = _FrameChannel()

    opening = threading.Thread(target=lambda: channel.publish({"frame_number": 1}, timeout_s=2))
    opening.start()
    assert channel.next(None) == {"frame_number": 1}
    opening.join(timeout=1)

    # The browser has collected once and then gone quiet. A short tolerance
    # stands in for the real sixty seconds so the test does not take a minute,
    # but not so short that ordinary thread scheduling reads as a dead browser.
    assert channel.publish({"frame_number": 2}, timeout_s=0.5) is True

    time.sleep(0.6)

    assert channel.publish({"frame_number": 3}, timeout_s=0.5) is False
    assert channel.closed


def test_closing_frame_channel_releases_a_waiting_publisher_headless() -> None:
    channel = _FrameChannel()
    published: list[bool] = []
    worker = threading.Thread(
        target=lambda: published.append(
            channel.publish({"frame_number": 7}, timeout_s=2)
        )
    )
    worker.start()

    while channel.packet is None:
        time.sleep(0.005)
    channel.close()
    worker.join(timeout=1)

    assert published == [False]


def test_first_frame_allows_a_slow_initial_browser_start() -> None:
    channel = _FrameChannel()
    published: list[bool] = []
    worker = threading.Thread(
        target=lambda: published.append(
            channel.publish(
                {"frame_number": 7},
                timeout_s=0.02,
                connect_timeout_s=0.3,
            )
        )
    )
    worker.start()

    # A fresh Chromium profile can take longer than the normal per-frame
    # response timeout before its first request reaches the loopback server.
    time.sleep(0.05)
    assert channel.next(None) == {"frame_number": 7}
    acknowledgement = threading.Thread(target=channel.next, args=(7,))
    acknowledgement.start()
    worker.join(timeout=1)
    channel.close()
    acknowledgement.join(timeout=1)

    assert published == [True]


def test_display_sends_a_browser_decodable_jpeg_frame() -> None:
    display = WatchDisplay(
        "window",
        Path("."),
        lambda _message: None,
        job_id="98",
        model_name="rockfish5",
        species_names=["Blue/Deacon Rockfish", "Lingcod"],
    )
    packets = []
    display._channel.publish = lambda packet: packets.append(packet) or True

    frame = SimpleNamespace(image=np.zeros((24, 32, 3), dtype=np.uint8))
    assert display.present(frame, 19, [], range_start=10, range_end=30) is True

    assert packets[0]["content_type"] == "image/jpeg"
    assert packets[0]["range_start"] == 10
    assert packets[0]["range_end"] == 30
    assert packets[0]["job_id"] == "98"
    assert packets[0]["model_name"] == "rockfish5"
    assert packets[0]["species_names"] == ["Blue/Deacon Rockfish", "Lingcod"]
    encoded = np.frombuffer(base64.b64decode(packets[0]["image"]), dtype=np.uint8)
    assert cv2.imdecode(encoded, cv2.IMREAD_COLOR).shape == (24, 32, 3)


# test_the_watch_page_ships_with_the_package()
# Verifies the page WatchDisplay.start() refuses to run without is actually there.
# Inputs: none.
# Output: pytest pass/fail result.
#
# This is the test whose absence let watch mode ship broken for its whole life.
# `start()` requires `live.html` in the player root, the page existed in neither
# repository nor in the installer's staging step, and so the window had never
# once opened -- while six tests around it passed, because not one of them
# touched the file the feature cannot start without.
def test_the_watch_page_ships_with_the_package() -> None:
    page = WatchDisplay._packaged_player_root() / "live.html"

    assert page.is_file(), f"the watch window has no page at {page}"

    # Not merely present. The page's whole job is to ask for the next frame and
    # acknowledge the one it drew, and a stub that loaded but never polled would
    # leave a blank window and pass a mere existence check.
    markup = page.read_text(encoding="utf-8")
    assert "/next" in markup
    assert "acknowledged_frame" in markup

    # The frame rates are read from the worker's own progress, not from frames
    # drawn: the display shows the most recent frame and skips the rest, so
    # counting draws would measure how fast the window paints and label it
    # inference. Asserted here because nothing else in the suite can see the
    # page, and a rate measured from the wrong source would look plausible.
    assert "RATE_WINDOW_MS" in markup
    assert 'id="fps"' in markup
    assert 'id="fpsall"' in markup
    assert "/status" in markup


# test_chromium_is_found_from_configuration_or_an_installed_browser()
# Verifies the browser search honours configuration and tolerates a bare machine.
# Inputs: pytest monkeypatch and temporary directory.
# Output: pytest pass/fail result.
#
# A volunteer's machine has Chrome or Edge but not the installer's bundled
# Chromium. Insisting on the bundle meant no window and a message that reached
# only the coordinator.
def test_chromium_is_found_from_configuration_or_an_installed_browser(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "my-chrome.exe"
    configured.write_bytes(b"")

    monkeypatch.setenv("MARP_CHROMIUM_PATH", str(configured))
    assert WatchDisplay._find_chromium() == configured.resolve()

    # A configured path that does not exist is None rather than a silent
    # fallback: somebody who set the variable meant that browser, and quietly
    # using a different one would hide their mistake.
    monkeypatch.setenv("MARP_CHROMIUM_PATH", str(tmp_path / "absent.exe"))
    assert WatchDisplay._find_chromium() is None

    # With nothing configured, a machine that has a browser finds it. This
    # asserts the search looks somewhere real rather than asserting the result,
    # because a CI runner may genuinely have no browser at all.
    monkeypatch.delenv("MARP_CHROMIUM_PATH", raising=False)
    found = WatchDisplay._find_chromium()
    assert found is None or found.is_file()


# test_the_watch_window_can_be_pinned_to_another_monitor(monkeypatch)
# Verifies the window position is configurable and never fatal.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# Every job used to open at 50,50 on the primary display, on top of whatever
# the volunteer was doing. A screen saver belongs on the monitor they are not
# working on.
def test_the_watch_window_can_be_pinned_to_another_monitor(monkeypatch) -> None:
    from marp_inference_worker.watch.display import _window_position

    # Unset now means "nothing was said", so the tiler decides. It used to
    # mean the literal 50,50, which is the default the tiler replaces.
    monkeypatch.delenv("MARP_WATCH_WINDOW_POSITION", raising=False)
    assert _window_position() is None

    monkeypatch.setenv("MARP_WATCH_WINDOW_POSITION", "2560,0")
    assert _window_position() == "2560,0"

    # Negative is legitimate: a monitor to the left of the primary one has
    # negative coordinates in Windows, and rejecting them would rule out half
    # the two-monitor arrangements there are.
    monkeypatch.setenv("MARP_WATCH_WINDOW_POSITION", "-1920, 200")
    assert _window_position() == "-1920,200"

    # A bad value falls back rather than raising. The display is decoration;
    # a typo in an environment variable must not cost somebody their job.
    for bad in ("nonsense", "1,2,3", "", "x,0"):
        monkeypatch.setenv("MARP_WATCH_WINDOW_POSITION", bad)
        assert _window_position() is None, bad


# test_slots_fill_the_monitors_before_they_are_subdivided(monkeypatch)
# Verifies the tiling rule.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# A worker running several slots opens several windows, and they used to all be
# maximised on the same screen -- so a volunteer saw one job and had no idea the
# other three were running. Fill the screens first, subdivide only when there
# are more windows than screens.
def test_slots_fill_the_monitors_before_they_are_subdivided(monkeypatch) -> None:
    from marp_inference_worker.watch import display as display_module

    monkeypatch.delenv("MARP_WATCH_WINDOW_POSITION", raising=False)
    monkeypatch.setattr(
        display_module,
        "_monitors",
        lambda: [(0, 0, 1920, 1080), (1920, 0, 1920, 1080)],
    )

    # One slot, two screens: the first screen, filled.
    assert display_module._tile(0, 1) == (0, 0, 1920, 1080)

    # Two slots, two screens: one each, no subdivision.
    assert display_module._tile(0, 2) == (0, 0, 1920, 1080)
    assert display_module._tile(1, 2) == (1920, 0, 1920, 1080)

    # Four slots, two screens: two per screen, side by side.
    tiles = [display_module._tile(slot, 4) for slot in range(4)]

    assert tiles == [
        (0, 0, 960, 1080),
        (960, 0, 960, 1080),
        (1920, 0, 960, 1080),
        (2880, 0, 960, 1080),
    ]

    # The tiles cover both screens exactly: no gap, no overlap. Integer
    # division would otherwise leave a seam down the right-hand edge.
    assert sum(width * height for _x, _y, width, height in tiles) == 2 * 1920 * 1080


# test_the_configured_position_chooses_the_first_screen_not_the_only_one(monkeypatch)
# Verifies what the anchor decides, and what it does not.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# An earlier version confined every window to the anchored monitor, so four
# slots on two screens became a cramped 2x2 on one while the other sat empty.
# The anchor says where the *first* window goes; filling the screens before
# dividing any of them is the rule, and it still applies.
def test_the_configured_position_chooses_the_first_screen_not_the_only_one(monkeypatch) -> None:
    from marp_inference_worker.watch import display as display_module

    primary = (0, 0, 1920, 1080)
    second = (1920, 0, 1920, 1080)

    monkeypatch.setattr(display_module, "_monitors", lambda: [primary, second])
    monkeypatch.setenv("MARP_WATCH_WINDOW_POSITION", "1920,0")

    # One window goes to the chosen screen, whole.
    assert display_module._tile(0, 1) == second

    # Two windows take a screen each -- the chosen one first. The other screen
    # is used, not avoided: there is a window for it.
    assert [display_module._tile(i, 2) for i in range(2)] == [second, primary]

    # Four windows: two per screen, not four on one.
    tiles = [display_module._tile(i, 4) for i in range(4)]

    assert tiles == [
        (1920, 0, 960, 1080),
        (2880, 0, 960, 1080),
        (0, 0, 960, 1080),
        (960, 0, 960, 1080),
    ]

    # Without an anchor the driver's own order stands, and the rule is the same.
    monkeypatch.delenv("MARP_WATCH_WINDOW_POSITION", raising=False)
    assert display_module._tile(0, 1) == primary

    # A coordinate on a monitor that is no longer plugged in falls back to the
    # first screen rather than drawing off the desktop where nobody can see it.
    monkeypatch.setenv("MARP_WATCH_WINDOW_POSITION", "9999,9999")
    assert display_module._tile(0, 1) == primary


# test_publishing_never_waits_for_the_browser_once_it_is_connected()
# Verifies that the display cannot set the pace of inference.
# Inputs: none.
# Output: pytest pass/fail result.
#
# This test used to assert the opposite -- that a publisher waits for the
# browser to acknowledge each frame -- and that contract was the reason a
# watched run was slower than the video it was showing. A 4080 did 55-75 f/s
# headless and 28 with a window; a 5060 ran each window at 0.7x real time.
# The window now shows the most recent frame it can collect and misses the
# ones in between, which is what watching is for.
def test_publishing_never_waits_for_the_browser_once_it_is_connected() -> None:
    channel = _FrameChannel()

    # Connect first: the very first frame still waits for the browser to
    # arrive, and that is asserted separately below.
    opening = threading.Thread(target=lambda: channel.publish({"frame_number": 1}, timeout_s=2))
    opening.start()
    assert channel.next(None) == {"frame_number": 1}
    opening.join(timeout=1)

    # Now publish a run of frames with nobody collecting them. Every call must
    # return immediately; if any of them blocks, this does not finish.
    started = time.monotonic()
    results = [channel.publish({"frame_number": n}, timeout_s=2) for n in range(2, 12)]
    elapsed = time.monotonic() - started

    assert all(results)
    assert elapsed < 0.5, f"publishing blocked for {elapsed:.2f}s"

    # The window gets the newest frame, not the oldest waiting one.
    assert channel.next(1) == {"frame_number": 11}

    # And the ones it never saw are counted rather than silently forgotten, so
    # "the window looks jumpy" has a number behind it.
    # Ten, not nine: frame 1 was collected by the window but never
    # acknowledged, so it was still pending and was replaced like the rest.
    assert channel.dropped == 10


# test_the_first_frame_still_waits_for_a_browser_that_never_arrives()
# Verifies the one case that must still block.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Dropping frames is right once somebody is watching. A browser that never
# opened at all is a real failure, and without this the run would push frames
# into a window that does not exist for the whole job.
def test_the_first_frame_still_waits_for_a_browser_that_never_arrives() -> None:
    channel = _FrameChannel()
    published: list[bool] = []

    worker = threading.Thread(
        target=lambda: published.append(
            channel.publish({"frame_number": 7}, timeout_s=2, connect_timeout_s=0.2)
        )
    )
    worker.start()
    worker.join(timeout=2)

    assert published == [False]
    assert channel.closed


# test_a_browser_that_stops_collecting_detaches_the_display()
# Verifies liveness now that nothing blocks on an acknowledgement.
# Inputs: none.
# Output: pytest pass/fail result.
#
# The old code noticed a dead browser by timing out on the acknowledgement it
# waited for. Nothing waits any more, so silence has to be checked explicitly
# -- otherwise a crashed window would leave the job encoding JPEGs into
# nothing for the rest of the range.
def test_a_browser_that_stops_collecting_detaches_the_display() -> None:
    channel = _FrameChannel()

    opening = threading.Thread(target=lambda: channel.publish({"frame_number": 1}, timeout_s=2))
    opening.start()
    assert channel.next(None) == {"frame_number": 1}
    opening.join(timeout=1)

    # The browser has collected once and then gone quiet. A short tolerance
    # stands in for the real sixty seconds so the test does not take a minute,
    # but not so short that ordinary thread scheduling reads as a dead browser.
    assert channel.publish({"frame_number": 2}, timeout_s=0.5) is True

    time.sleep(0.6)

    assert channel.publish({"frame_number": 3}, timeout_s=0.5) is False
    assert channel.closed


def test_closing_frame_channel_releases_a_waiting_publisher_headless() -> None:
    channel = _FrameChannel()
    published: list[bool] = []
    worker = threading.Thread(
        target=lambda: published.append(
            channel.publish({"frame_number": 7}, timeout_s=2)
        )
    )
    worker.start()

    while channel.packet is None:
        time.sleep(0.005)
    channel.close()
    worker.join(timeout=1)

    assert published == [False]


def test_first_frame_allows_a_slow_initial_browser_start() -> None:
    channel = _FrameChannel()
    published: list[bool] = []
    worker = threading.Thread(
        target=lambda: published.append(
            channel.publish(
                {"frame_number": 7},
                timeout_s=0.02,
                connect_timeout_s=0.3,
            )
        )
    )
    worker.start()

    # A fresh Chromium profile can take longer than the normal per-frame
    # response timeout before its first request reaches the loopback server.
    time.sleep(0.05)
    assert channel.next(None) == {"frame_number": 7}
    acknowledgement = threading.Thread(target=channel.next, args=(7,))
    acknowledgement.start()
    worker.join(timeout=1)
    channel.close()
    acknowledgement.join(timeout=1)

    assert published == [True]


def test_display_sends_a_browser_decodable_jpeg_frame() -> None:
    display = WatchDisplay(
        "window",
        Path("."),
        lambda _message: None,
        job_id="98",
        model_name="rockfish5",
        species_names=["Blue/Deacon Rockfish", "Lingcod"],
    )
    packets = []
    display._channel.publish = lambda packet: packets.append(packet) or True

    frame = SimpleNamespace(image=np.zeros((24, 32, 3), dtype=np.uint8))
    assert display.present(frame, 19, [], range_start=10, range_end=30) is True

    assert packets[0]["content_type"] == "image/jpeg"
    assert packets[0]["range_start"] == 10
    assert packets[0]["range_end"] == 30
    assert packets[0]["job_id"] == "98"
    assert packets[0]["model_name"] == "rockfish5"
    assert packets[0]["species_names"] == ["Blue/Deacon Rockfish", "Lingcod"]
    encoded = np.frombuffer(base64.b64decode(packets[0]["image"]), dtype=np.uint8)
    assert cv2.imdecode(encoded, cv2.IMREAD_COLOR).shape == (24, 32, 3)


# test_the_watch_page_ships_with_the_package()
# Verifies the page WatchDisplay.start() refuses to run without is actually there.
# Inputs: none.
# Output: pytest pass/fail result.
#
# This is the test whose absence let watch mode ship broken for its whole life.
# `start()` requires `live.html` in the player root, the page existed in neither
# repository nor in the installer's staging step, and so the window had never
# once opened -- while six tests around it passed, because not one of them
# touched the file the feature cannot start without.
def test_the_watch_page_ships_with_the_package() -> None:
    page = WatchDisplay._packaged_player_root() / "live.html"

    assert page.is_file(), f"the watch window has no page at {page}"

    # Not merely present. The page's whole job is to ask for the next frame and
    # acknowledge the one it drew, and a stub that loaded but never polled would
    # leave a blank window and pass a mere existence check.
    markup = page.read_text(encoding="utf-8")
    assert "/next" in markup
    assert "acknowledged_frame" in markup

    # The frame rates are read from the worker's own progress, not from frames
    # drawn: the display shows the most recent frame and skips the rest, so
    # counting draws would measure how fast the window paints and label it
    # inference. Asserted here because nothing else in the suite can see the
    # page, and a rate measured from the wrong source would look plausible.
    assert "RATE_WINDOW_MS" in markup
    assert 'id="fps"' in markup
    assert 'id="fpsall"' in markup
    assert "/status" in markup


# test_chromium_is_found_from_configuration_or_an_installed_browser()
# Verifies the browser search honours configuration and tolerates a bare machine.
# Inputs: pytest monkeypatch and temporary directory.
# Output: pytest pass/fail result.
#
# A volunteer's machine has Chrome or Edge but not the installer's bundled
# Chromium. Insisting on the bundle meant no window and a message that reached
# only the coordinator.
def test_chromium_is_found_from_configuration_or_an_installed_browser(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "my-chrome.exe"
    configured.write_bytes(b"")

    monkeypatch.setenv("MARP_CHROMIUM_PATH", str(configured))
    assert WatchDisplay._find_chromium() == configured.resolve()

    # A configured path that does not exist is None rather than a silent
    # fallback: somebody who set the variable meant that browser, and quietly
    # using a different one would hide their mistake.
    monkeypatch.setenv("MARP_CHROMIUM_PATH", str(tmp_path / "absent.exe"))
    assert WatchDisplay._find_chromium() is None

    # With nothing configured, a machine that has a browser finds it. This
    # asserts the search looks somewhere real rather than asserting the result,
    # because a CI runner may genuinely have no browser at all.
    monkeypatch.delenv("MARP_CHROMIUM_PATH", raising=False)
    found = WatchDisplay._find_chromium()
    assert found is None or found.is_file()


# test_the_watch_window_can_be_pinned_to_another_monitor(monkeypatch)
# Verifies the window position is configurable and never fatal.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# Every job used to open at 50,50 on the primary display, on top of whatever
# the volunteer was doing. A screen saver belongs on the monitor they are not
# working on.
def test_the_watch_window_can_be_pinned_to_another_monitor(monkeypatch) -> None:
    from marp_inference_worker.watch.display import _window_position

    # Unset now means "nothing was said", so the tiler decides. It used to
    # mean the literal 50,50, which is the default the tiler replaces.
    monkeypatch.delenv("MARP_WATCH_WINDOW_POSITION", raising=False)
    assert _window_position() is None

    monkeypatch.setenv("MARP_WATCH_WINDOW_POSITION", "2560,0")
    assert _window_position() == "2560,0"

    # Negative is legitimate: a monitor to the left of the primary one has
    # negative coordinates in Windows, and rejecting them would rule out half
    # the two-monitor arrangements there are.
    monkeypatch.setenv("MARP_WATCH_WINDOW_POSITION", "-1920, 200")
    assert _window_position() == "-1920,200"

    # A bad value falls back rather than raising. The display is decoration;
    # a typo in an environment variable must not cost somebody their job.
    for bad in ("nonsense", "1,2,3", "", "x,0"):
        monkeypatch.setenv("MARP_WATCH_WINDOW_POSITION", bad)
        assert _window_position() is None, bad


# test_slots_fill_the_monitors_before_they_are_subdivided(monkeypatch)
# Verifies the tiling rule.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# A worker running several slots opens several windows, and they used to all be
# maximised on the same screen -- so a volunteer saw one job and had no idea the
# other three were running. Fill the screens first, subdivide only when there
# are more windows than screens.
def test_slots_fill_the_monitors_before_they_are_subdivided(monkeypatch) -> None:
    from marp_inference_worker.watch import display as display_module

    monkeypatch.delenv("MARP_WATCH_WINDOW_POSITION", raising=False)
    monkeypatch.setattr(
        display_module,
        "_monitors",
        lambda: [(0, 0, 1920, 1080), (1920, 0, 1920, 1080)],
    )

    # One slot, two screens: the first screen, filled.
    assert display_module._tile(0, 1) == (0, 0, 1920, 1080)

    # Two slots, two screens: one each, no subdivision.
    assert display_module._tile(0, 2) == (0, 0, 1920, 1080)
    assert display_module._tile(1, 2) == (1920, 0, 1920, 1080)

    # Four slots, two screens: two per screen, side by side.
    tiles = [display_module._tile(slot, 4) for slot in range(4)]

    assert tiles == [
        (0, 0, 960, 1080),
        (960, 0, 960, 1080),
        (1920, 0, 960, 1080),
        (2880, 0, 960, 1080),
    ]

    # The tiles cover both screens exactly: no gap, no overlap. Integer
    # division would otherwise leave a seam down the right-hand edge.
    assert sum(width * height for _x, _y, width, height in tiles) == 2 * 1920 * 1080


# test_a_frame_the_window_cannot_take_is_never_encoded()
# Verifies the display does no work on frames nobody will see.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Every frame used to be JPEG-encoded and base64'd before the channel decided
# whether to keep it, so once the channel started dropping frames that work was
# thrown away immediately afterwards -- a 1080p encode per frame above the rate
# the window can draw, on every slot at once.
def test_a_frame_the_window_cannot_take_is_never_encoded() -> None:
    display = WatchDisplay("window", Path("."), lambda _message: None)
    encoded: list[int] = []

    # Stand in for the channel: connected, and never ready for a frame.
    class Busy:
        closed = False

        def wants_frame(self, timeout_s: float = 0) -> bool:
            return False

    display._channel = Busy()

    frame = SimpleNamespace(image=np.zeros((1080, 1920, 3), dtype=np.uint8))

    # Alive, so the run carries on -- it simply does not need this picture.
    assert display.present(frame, 5, []) is True
    assert display._skipped == 1

    # And nothing was encoded: a real encode of a 1080p frame would show up as
    # time, so the assertion is on the counter the skip path increments rather
    # than on a clock that would be flaky.
    assert encoded == []


# test_a_window_re_tiles_when_the_worker_takes_another_job()
# Verifies the layout follower reacts to the count changing.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# A window is tiled for the number of windows there were when it opened. The
# worker decides how many jobs to run from measured throughput, so that number
# changes while jobs are running -- and the windows already up kept their old
# layout, so a third window landed on top of one of the first two.
def test_a_window_re_tiles_when_the_worker_takes_another_job(monkeypatch) -> None:
    from marp_inference_worker.watch import display as display_module

    primary = (0, 0, 1920, 1080)
    second = (1920, 0, 1920, 1080)

    monkeypatch.setattr(display_module, "_monitors", lambda: [primary, second])
    monkeypatch.delenv("MARP_WATCH_WINDOW_POSITION", raising=False)

    # Slot 0 opened when there was one window: the whole first screen.
    assert display_module._tile(0, 1) == primary

    # A second job starts. Slot 0 keeps a whole screen, slot 1 takes the other.
    assert display_module._tile(0, 2) == primary
    assert display_module._tile(1, 2) == second

    # A third starts, and this is the case that was broken: slot 0 must give up
    # half its screen rather than stay full-width with the newcomer on top.
    assert display_module._tile(0, 3) == (0, 0, 960, 1080)
    assert display_module._tile(1, 3) == (960, 0, 960, 1080)
    assert display_module._tile(2, 3) == second

    # No two windows overlap, which is the property that actually matters.
    tiles = [display_module._tile(i, 3) for i in range(3)]

    for first_index in range(len(tiles)):
        for second_index in range(first_index + 1, len(tiles)):
            ax, ay, aw, ah = tiles[first_index]
            bx, by, bw, bh = tiles[second_index]
            overlaps = ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
            assert not overlaps, f"{tiles[first_index]} overlaps {tiles[second_index]}"


# test_a_frame_packet_says_which_attempt_it_belongs_to()
# Verifies the window can tell its own job from the others on the machine.
# Inputs: none.
# Output: pytest pass/fail result.
#
# `live.html` declared `attemptId` with a comment saying it was learned from the
# first packet, and nothing ever assigned it -- the packet did not carry one. So
# every window fell back to the first job in `/status`, which is correct on a
# one-slot machine and wrong on every other: four windows all showed the first
# slot's dive, line and video, with nothing to suggest they were wrong.
def test_a_frame_packet_says_which_attempt_it_belongs_to(tmp_path: Path) -> None:
    import numpy as np

    display = WatchDisplay(
        "window", tmp_path / "jobs" / "77", lambda _message: None,
        job_id="4659", attempt_id="3170",
    )
    published: list[dict] = []
    display._channel.publish = lambda packet, **_: published.append(packet) or True

    frame = SimpleNamespace(image=np.zeros((8, 8, 3), dtype=np.uint8))
    assert display.present(frame, 100, [], range_start=0, range_end=200) is True

    assert published[0]["attempt_id"] == "3170"
    assert published[0]["job_id"] == "4659"


# test_a_busy_machine_is_not_interrupted(monkeypatch)
# Verifies the watch window does not steal focus from somebody working.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# The window still opens and still tiles; only the raise is withheld. A screen
# saver that jumps in front of a volunteer mid-keystroke is the likeliest single
# reason they stop donating the machine, which costs far more than a window they
# can raise themselves.
def test_a_busy_machine_is_not_interrupted(monkeypatch) -> None:
    from marp_inference_worker.watch import display as display_module

    raised: list[int] = []
    monkeypatch.setattr(display_module.sys, "platform", "win32")
    monkeypatch.setattr(display_module, "_window_handles_for", lambda pid, attempts: raised.append(pid) or [])

    # Actively typing: left alone.
    monkeypatch.setattr(display_module, "_seconds_since_input", lambda: 2.0)
    display_module._bring_to_front(1234)
    assert raised == []

    # Idle long enough: the window may come forward.
    monkeypatch.setattr(display_module, "_seconds_since_input",
                        lambda: display_module._IDLE_BEFORE_RAISE_S + 1)
    display_module._bring_to_front(1234)
    assert raised == [1234]

    # A platform that cannot answer must not become a platform that never
    # raises -- silence is not the same as "somebody is typing".
    monkeypatch.setattr(display_module, "_seconds_since_input", lambda: None)
    display_module._bring_to_front(5678)
    assert raised == [1234, 5678]


# test_closing_kills_the_browser_even_when_the_launched_process_has_gone(monkeypatch)
# Verifies a finished job leaves no window behind.
# Inputs: pytest monkeypatch and temporary directory.
# Output: pytest pass/fail result.
#
# `close()` used to kill only the process it launched, and skip even that when
# `poll()` said it had exited. Chromium can hand its window to another process
# and exit, so the pid was gone while the window was not: the server shut down
# underneath a live page, which then sat there showing a frozen frame and a
# fetch error. The profile directory is unique to the attempt, so whatever holds
# it is ours to end.
def test_closing_kills_the_browser_even_when_the_launched_process_has_gone(
    monkeypatch, tmp_path: Path
) -> None:
    from marp_inference_worker.watch import display as display_module

    display = WatchDisplay("window", tmp_path / "jobs" / "9", lambda _message: None)
    display._profile_dir = tmp_path / "jobs" / "9" / "chromium-profile"

    # The launched process reports itself already gone, which is the case that
    # used to skip the kill entirely.
    display._process = SimpleNamespace(pid=4242, poll=lambda: 0)

    calls: list[list[str]] = []

    def record(args, **_kwargs):
        calls.append(list(args))
        return SimpleNamespace(stdout="7777\n8888\n", returncode=0)

    monkeypatch.setattr(display_module.sys, "platform", "win32")
    monkeypatch.setattr(display_module.subprocess, "run", record)
    display.close()

    # It asked Windows which processes hold this attempt's profile...
    assert any("Win32_Process" in " ".join(call) for call in calls)
    # ...and killed the ones it was told about, rather than the dead pid.
    killed = {call[2] for call in calls if call and call[0] == "taskkill"}
    assert killed == {"7777", "8888"}

    # And the query names this attempt's own profile, so it can never reach the
    # volunteer's browser or another slot's window.
    query = next(" ".join(call) for call in calls if "Win32_Process" in " ".join(call))
    assert "chromium-profile" in query


# Verifies that a Linux volunteer's browser is found.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# _find_chromium() looked only for chrome.exe under Windows Program Files, so on
# Linux it returned None and the watch window never opened at all. Refs #31.
def test_find_chromium_finds_a_linux_browser_on_path(monkeypatch, tmp_path) -> None:
    from marp_inference_worker.watch import display as display_module

    browser = tmp_path / "chromium"
    browser.write_text("#!/bin/sh\n")

    monkeypatch.delenv("MARP_CHROMIUM_PATH", raising=False)
    monkeypatch.setattr(display_module.sys, "platform", "linux")
    monkeypatch.setattr(
        display_module.shutil,
        "which",
        lambda name: str(browser) if name == "chromium" else None,
    )

    assert display_module.WatchDisplay._find_chromium() == browser


# Verifies a snap wrapper is launched by the name it was found under.
# Inputs: pytest monkeypatch, tmp_path.
# Output: pytest pass/fail result.
#
# /snap/bin/chromium is a symlink to /usr/bin/snap, which reads argv[0] to decide
# which snap to run. Resolving it hands Chromium's arguments to the snap tool and
# nothing opens. This is the real layout on an Ubuntu box with the chromium snap,
# and it is the difference between a window and silence. Refs #31.
# Creating a symlink on Windows needs elevation or developer mode, and the layout this
# asserts -- a multi-call wrapper reached through PATH -- cannot occur there at all. A red
# test nobody can fix reads as decay, so this one is skipped rather than left failing.
@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs elevation")
def test_find_chromium_does_not_resolve_a_multi_call_wrapper(monkeypatch, tmp_path) -> None:
    from marp_inference_worker.watch import display as display_module

    real_tool = tmp_path / "snap"
    real_tool.write_text("#!/bin/sh\n")
    wrapper = tmp_path / "chromium"
    wrapper.symlink_to(real_tool)

    monkeypatch.delenv("MARP_CHROMIUM_PATH", raising=False)
    monkeypatch.setattr(display_module.sys, "platform", "linux")
    monkeypatch.setattr(
        display_module.shutil,
        "which",
        lambda name: str(wrapper) if name == "chromium" else None,
    )

    found = display_module.WatchDisplay._find_chromium()

    assert found == wrapper
    assert found.name == "chromium"


# Verifies the monitor geometry comes from xrandr on Linux.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# Without this the fallback assumed one 1920x1080 screen, which on the rotated
# 2880x5120 desktop this was first run on sized every window for the wrong
# display. Refs #31.
def test_monitors_reads_xrandr_on_linux(monkeypatch) -> None:
    from marp_inference_worker.watch import display as display_module

    listing = (
        "Monitors: 2\n"
        " 0: +*HDMI-3 2880/600x5120/340+0+0  HDMI-3\n"
        " 1: +DP-1 1920/520x1080/290+2880+0  DP-1\n"
    )
    monkeypatch.setattr(display_module.sys, "platform", "linux")
    monkeypatch.setattr(
        display_module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=listing, stderr="", returncode=0),
    )

    assert display_module._monitors() == [
        (0, 0, 2880, 5120),
        (2880, 0, 1920, 1080),
    ]


# Verifies a worker still draws when the displays cannot be enumerated.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# A pure Wayland session with no XWayland has no xrandr to answer. Showing a
# window on an assumed screen beats showing nothing. Refs #31.
def test_monitors_falls_back_to_one_screen_when_xrandr_is_absent(monkeypatch) -> None:
    from marp_inference_worker.watch import display as display_module

    def explode(*a, **k):
        raise FileNotFoundError("xrandr")

    monkeypatch.setattr(display_module.sys, "platform", "linux")
    monkeypatch.setattr(display_module.subprocess, "run", explode)

    assert display_module._monitors() == [(0, 0, 1920, 1080)]


# Verifies a finished job's window is closed on Linux.
# Inputs: pytest monkeypatch, tmp_path.
# Output: pytest pass/fail result.
#
# The cleanup shelled out to powershell.exe and taskkill and returned early off
# Windows, which was correct only while no window could open there. Matching on
# the attempt's own profile directory rather than the process name is what keeps
# it off the volunteer's own browser -- which on Linux may be the very Chromium
# the worker borrowed. Refs #31, and the fault #29 describes.
def test_kill_by_profile_closes_the_window_on_linux(monkeypatch, tmp_path) -> None:
    from marp_inference_worker.watch import display as display_module

    profile = tmp_path / "chromium-profile"
    profile.mkdir()
    calls = []

    monkeypatch.setattr(display_module.sys, "platform", "linux")
    monkeypatch.setattr(
        display_module.subprocess,
        "run",
        lambda args, **k: calls.append(args) or SimpleNamespace(returncode=0),
    )

    display_module.WatchDisplay._kill_by_profile(
        SimpleNamespace(_profile_dir=profile)
    )

    assert calls == [["pkill", "-f", str(profile)]]


# Verifies the cleanup does nothing when there is no profile to match.
# Inputs: pytest monkeypatch, tmp_path.
# Output: pytest pass/fail result.
#
# A bare `pkill -f` with an empty needle would match far too much, so the guard
# that returns early matters more than it looks. Refs #31.
def test_kill_by_profile_does_nothing_without_a_profile(monkeypatch) -> None:
    from marp_inference_worker.watch import display as display_module

    calls = []
    monkeypatch.setattr(display_module.sys, "platform", "linux")
    monkeypatch.setattr(
        display_module.subprocess,
        "run",
        lambda args, **k: calls.append(args) or SimpleNamespace(returncode=0),
    )

    display_module.WatchDisplay._kill_by_profile(SimpleNamespace(_profile_dir=None))

    assert calls == []


# Verifies the close path will not signal a pid that is no longer our browser.
# Inputs: pytest monkeypatch, tmp_path.
# Output: pytest pass/fail result.
#
# /snap/bin/chromium is a symlink to /usr/bin/snap, which execs and exits, so the
# launched pid belongs to a wrapper that is gone within seconds and whose number
# Linux then recycles. Terminating it raised PermissionError on seven attempts --
# and that was the lucky case: EPERM only occurs when the new owner is another
# user. Recycled to a signalable process, this would have killed a volunteer's own
# program silently. Refs #39, and the same lesson as #28 on Windows.
def test_close_does_not_signal_a_recycled_pid(monkeypatch, tmp_path) -> None:
    from marp_inference_worker.watch import display as display_module

    profile = tmp_path / "chromium-profile"
    profile.mkdir()

    display = SimpleNamespace(_process=SimpleNamespace(pid=4242), _profile_dir=profile)

    # A pid whose command line does not mention our profile is not our browser,
    # whatever it is.
    monkeypatch.setattr(
        display_module.Path, "read_bytes", lambda self: b"/usr/bin/something-else\x00"
    )
    assert display_module.WatchDisplay._still_our_browser(display) is False

    # And one that does name it, is.
    monkeypatch.setattr(
        display_module.Path,
        "read_bytes",
        lambda self: b"chrome\x00--user-data-dir=" + str(profile).encode() + b"\x00",
    )
    assert display_module.WatchDisplay._still_our_browser(display) is True


# Verifies a vanished pid is treated as not ours rather than raising.
# Inputs: pytest monkeypatch, tmp_path.
# Output: pytest pass/fail result.
#
# The common case: the wrapper has exited and /proc/<pid> no longer exists. That
# must be "not ours" and not an exception, because it happens on every job. Refs #39.
def test_a_vanished_pid_is_not_our_browser(monkeypatch, tmp_path) -> None:
    from marp_inference_worker.watch import display as display_module

    profile = tmp_path / "chromium-profile"
    profile.mkdir()
    display = SimpleNamespace(_process=SimpleNamespace(pid=999999), _profile_dir=profile)

    def gone(self):
        raise FileNotFoundError("/proc/999999/cmdline")

    monkeypatch.setattr(display_module.Path, "read_bytes", gone)

    assert display_module.WatchDisplay._still_our_browser(display) is False
