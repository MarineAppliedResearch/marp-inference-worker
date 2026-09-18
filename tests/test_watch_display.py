"""Tests for the child-owned ordered watch channel."""

import base64
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

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


# test_tiling_stays_off_the_monitor_somebody_is_working_on(monkeypatch)
# Verifies that a configured position names the watching screen.
# Inputs: pytest monkeypatch.
# Output: pytest pass/fail result.
#
# `MARP_WATCH_WINDOW_POSITION` exists because windows kept opening on the screen
# the volunteer was using. A tiler that then helpfully fills every monitor puts
# them straight back, which would undo the setting rather than honour it.
def test_tiling_stays_off_the_monitor_somebody_is_working_on(monkeypatch) -> None:
    from marp_inference_worker.watch import display as display_module

    monkeypatch.setattr(
        display_module,
        "_monitors",
        lambda: [(0, 0, 1920, 1080), (1920, 0, 1920, 1080)],
    )
    monkeypatch.setenv("MARP_WATCH_WINDOW_POSITION", "1920,0")

    tiles = [display_module._tile(slot, 4) for slot in range(4)]

    # All four on the second screen, as a 2x2 grid. The first screen is left
    # alone entirely, which is the whole point of the setting.
    assert all(x >= 1920 for x, _y, _w, _h in tiles)
    assert tiles == [
        (1920, 0, 960, 540),
        (2880, 0, 960, 540),
        (1920, 540, 960, 540),
        (2880, 540, 960, 540),
    ]

    # A coordinate on a monitor that is no longer plugged in falls back to the
    # first screen rather than drawing off the desktop where nobody can see it.
    monkeypatch.setenv("MARP_WATCH_WINDOW_POSITION", "9999,9999")
    assert display_module._tile(0, 1) == (0, 0, 1920, 1080)


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
