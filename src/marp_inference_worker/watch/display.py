"""Child-owned loopback server and local Chromium window for watch mode."""

import base64
import json
import math
import os
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import cv2


_FRAME_ACK_TIMEOUT_S = 60.0


# _window_position()
# Where the watch window opens, as Chromium's `x,y`.
# Inputs: none; reads `MARP_WATCH_WINDOW_POSITION`.
# Output: a coordinate string, or None when nothing was configured.
# Use this rather than a literal. Negative values are legitimate -- a monitor
# to the left of the primary one has negative coordinates in Windows.
def _window_position() -> str | None:

    configured = (os.environ.get("MARP_WATCH_WINDOW_POSITION") or "").strip()

    if configured:
        parts = configured.split(",")

        if len(parts) == 2:
            try:
                return f"{int(parts[0].strip())},{int(parts[1].strip())}"
            except ValueError:
                pass

    return None


# _monitors()
# The desktop's monitors, as (x, y, width, height) in virtual-screen space.
# Inputs: none; asks the window system.
# Output: at least one rectangle, left to right.
# Use this to lay windows out. Coordinates can be negative: a monitor to the
# left of or above the primary one starts at a negative origin, which is why
# nothing here assumes the desktop begins at 0,0.
def _monitors() -> list[tuple[int, int, int, int]]:

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            found: list[tuple[int, int, int, int]] = []

            # The callback signature EnumDisplayMonitors expects.
            callback_type = ctypes.WINFUNCTYPE(
                ctypes.c_int,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.POINTER(wintypes.RECT),
                ctypes.c_double,
            )

            def collect(_monitor, _dc, rect_pointer, _data) -> int:
                rect = rect_pointer.contents
                found.append(
                    (
                        int(rect.left),
                        int(rect.top),
                        int(rect.right - rect.left),
                        int(rect.bottom - rect.top),
                    )
                )
                return 1

            ctypes.windll.user32.EnumDisplayMonitors(
                0, 0, callback_type(collect), 0
            )

            if found:
                # Left to right, then top to bottom, so "the first monitor" is
                # the leftmost one rather than whichever the driver enumerated
                # first -- which is not stable across reboots.
                return sorted(found, key=lambda rect: (rect[0], rect[1]))

        except Exception:
            # Any failure here falls through to the single-screen assumption.
            # A worker must not refuse to draw because it could not enumerate
            # displays.
            pass

    # One screen, conservative size. Wrong on a large display only in that the
    # windows are smaller than they could be, which is the harmless direction.
    return [(0, 0, 1920, 1080)]


# _tile()
# Where one slot's window goes, given how many there are.
# Inputs: this slot's index, and how many slots the worker runs.
# Output: (x, y, width, height).
#
# **Fill the screens before dividing any of them.**
#
#   one window                -> one whole monitor
#   two windows, two monitors -> a whole monitor each
#   n windows, n monitors     -> a whole monitor each
#   more windows than screens -> shared out, then gridded within each screen
#
# A configured `MARP_WATCH_WINDOW_POSITION` chooses which monitor is *first*,
# not which monitor is the only one used. An earlier version confined every
# window to the anchored screen, which turned four slots on two monitors into a
# cramped 2x2 on one while the other sat empty. The anchor answers "where does
# the first window go"; the rule above answers the rest.
def _tile(slot_index: int, slot_count: int) -> tuple[int, int, int, int]:

    monitors = _ordered_monitors()
    count = max(1, slot_count)
    index = max(0, slot_index) % count

    # A screen each, whole, while there are screens to go round.
    if count <= len(monitors):
        return monitors[index]

    # More windows than screens. Share them out as evenly as possible, giving
    # the earlier monitors the extra one when it does not divide -- the first
    # monitor is the one chosen for watching.
    per_monitor = [count // len(monitors)] * len(monitors)

    for spare in range(count % len(monitors)):
        per_monitor[spare] += 1

    # Which monitor this slot lands on, and where in that monitor's own run.
    monitor_index = 0
    local_index = index

    for position, share in enumerate(per_monitor):
        if local_index < share:
            monitor_index = position
            break
        local_index -= share

    return _grid_within(monitors[monitor_index], local_index, per_monitor[monitor_index])


# _ordered_monitors()
# The monitors, with the configured watching screen first.
# Inputs: none; reads `MARP_WATCH_WINDOW_POSITION`.
# Output: the monitor rectangles, at least one.
# Use this so a single window lands on the screen somebody chose rather than on
# whichever one the driver enumerated first. The order is the only thing the
# anchor decides -- every monitor is still used once there are enough windows
# to need them.
def _ordered_monitors() -> list[tuple[int, int, int, int]]:

    monitors = _monitors()
    anchor = _window_position()

    if not anchor:
        return monitors

    x_text, y_text = anchor.split(",")
    chosen = _monitor_containing((int(x_text), int(y_text)), monitors)

    return [chosen] + [rect for rect in monitors if rect != chosen]


# _monitor_containing()
# The monitor a point falls on.
# Inputs: an (x, y) point and the monitor rectangles.
# Output: one rectangle; the first when the point is off every screen.
# Use this to turn a configured coordinate into the screen it meant. A stale
# coordinate from an unplugged monitor lands nowhere, and falling back to the
# first screen is better than drawing off the desktop where nobody can see it.
def _monitor_containing(
    point: tuple[int, int],
    monitors: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:

    x, y = point

    for rect in monitors:
        left, top, width, height = rect

        if left <= x < left + width and top <= y < top + height:
            return rect

    return monitors[0]


# _grid_within()
# One tile of a near-square grid filling a rectangle.
# Inputs: the rectangle, this tile's index, and how many tiles share it.
# Output: (x, y, width, height).
# Use this for the windows sharing one monitor. Columns are taken before rows
# because these frames are wider than they are tall, so a short wide window
# wastes less of one than a tall narrow one.
def _grid_within(
    rect: tuple[int, int, int, int],
    index: int,
    share: int,
) -> tuple[int, int, int, int]:

    left, top, width, height = rect
    share = max(1, share)
    index = max(0, index) % share

    columns = math.ceil(math.sqrt(share))
    rows = math.ceil(share / columns)

    column = index % columns
    row = index // columns

    # Integer division leaves a few pixels at the right and bottom edges. The
    # last column and row absorb them, so the tiling covers the monitor exactly
    # rather than leaving a seam down the side of the screen.
    tile_width = width // columns
    tile_height = height // rows
    this_width = width - tile_width * (columns - 1) if column == columns - 1 else tile_width
    this_height = height - tile_height * (rows - 1) if row == rows - 1 else tile_height

    return (
        left + tile_width * column,
        top + tile_height * row,
        this_width,
        this_height,
    )


def _chromium_args(
    chromium: Path,
    url: str,
    profile: Path,
    mode: str,
    slot_index: int = 0,
    slot_count: int = 1,
) -> list[str]:
    left, top, width, height = _tile(slot_index, slot_count)
    args = [
        str(chromium),
        f"--app={url}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--disable-sync",
        # The display server is loopback-only. Corporate/system proxy settings can
        # otherwise leave the app window stuck on its raw 127.0.0.1 URL.
        "--no-proxy-server",
        "--proxy-bypass-list=<-loopback>",
        # New per-job profiles have no saved bounds, so the window is pinned
        # rather than left to Chromium. `MARP_WATCH_WINDOW_POSITION` as `x,y`
        # moves it: a volunteer with two monitors wants the screen saver on the
        # one they are not working on, and without this every job opens on top
        # of whatever they are doing. Malformed values fall back rather than
        # raise -- a bad coordinate must not cost somebody their job.
        f"--window-position={left},{top}",
        f"--window-size={width},{height}",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-features=CalculateNativeWinOcclusion",
        # Keep Chromium off the inference GPU. This laptop class renders the
        # canvas through Chromium's CPU paint path when GPU compositing is off.
        "--disable-gpu",
        "--disable-gpu-compositing",
        "--mute-audio",
    ]
    # Fullscreen only when asked for it outright.
    #
    # `--start-maximized` used to be here and now fights the tiling: maximising
    # fills whichever monitor the window landed on and throws the computed size
    # away, so two windows meant to sit side by side end up stacked on top of
    # each other. The tile is already the whole monitor when there is a screen
    # per slot, which is what "maximised" was for.
    if mode == "fullscreen":
        args.append("--start-fullscreen")

    return args


# _visible_windows()
# Every visible top-level window a process owns.
# Inputs: the process id.
# Output: window handles, possibly empty.
# Use this rather than taking the first one found. Chromium owns several, and
# the first enumerated is not reliably the app window -- raising that one put
# the work on something invisible while every call reported success.
def _visible_windows(process_id: int) -> list[int]:

    if sys.platform != "win32":
        return []

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def visit(handle, _param):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(owner))

        if owner.value == process_id and user32.IsWindowVisible(handle):
            found.append(handle)

        return True

    try:
        user32.EnumWindows(visit, 0)
    except Exception:
        return []

    return found


# _move_windows()
# Puts a process's windows at a rectangle, without disturbing the user.
# Inputs: the process id and (x, y, width, height).
# Output: none.
#
# `SWP_NOACTIVATE` and no `SetForegroundWindow`: this runs when the layout
# changes underneath a window that is already open, and a screen saver that
# steals focus every time another job starts is a screen saver nobody can work
# beside. The one raise a window gets is when it first appears.
def _move_windows(process_id: int, rect: tuple[int, int, int, int]) -> None:

    if sys.platform != "win32":
        return

    import ctypes

    user32 = ctypes.windll.user32
    left, top, width, height = rect
    SWP_NOACTIVATE = 0x0010
    SWP_NOZORDER = 0x0004

    for handle in _visible_windows(process_id):
        try:
            user32.SetWindowPos(
                handle, 0, left, top, width, height, SWP_NOACTIVATE | SWP_NOZORDER
            )
        except Exception:
            # Best effort. A window that will not move is worth less than the
            # job it is showing.
            pass


# _bring_to_front()
# Raises the watch window above whatever else is on screen.
# Inputs: the Chromium process id.
# Output: none.
# Use this after launch, on Windows only.
#
# Chromium does not always come up in front: launched from a background service
# it can open behind the active window, and a screen saver nobody can see is
# the same as no screen saver. `--start-maximized` sizes it; only this puts it
# in front. Best effort throughout -- a window that will not raise is worth
# less than the job, so nothing here may raise.
# How long the machine must have been untouched before a watch window is
# allowed to take the foreground, in seconds.
#
# A screen saver that jumps in front of somebody mid-sentence is a screen saver
# they uninstall. The window still opens and still tiles; it simply opens behind
# what they are doing and waits its turn.
_IDLE_BEFORE_RAISE_S = 45.0


# _seconds_since_input()
# How long since the volunteer last touched this machine.
# Inputs: none.
# Output: seconds, or None when the platform cannot say.
# Use this before taking the foreground. `GetLastInputInfo` counts keyboard and
# mouse across the whole session, which is the question being asked -- not
# whether this process has focus.
def _seconds_since_input() -> float | None:

    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    class _LastInput(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    info = _LastInput()
    info.cbSize = ctypes.sizeof(_LastInput)

    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return None

    # Both are millisecond tick counts that wrap after 49 days; the subtraction
    # is masked to 32 bits so a wrap reads as a small number rather than a
    # negative one, which would look like input from the future.
    elapsed_ms = (ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF
    return elapsed_ms / 1000.0


# _window_handles_for()
# Every visible top-level window a process owns.
# Inputs: the process id, and how many quarter-seconds to wait for one.
# Output: the handles, newest search each call.
# Separated from the raise so the decision about *whether* to raise can be
# tested without a real window, which is the half that has been wrong twice.
def _window_handles_for(process_id: int, attempts: int = 40) -> list[int]:

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[int] = []

    # EnumWindows rather than a title match: the page title is ours to change
    # and matching on it would break the moment somebody edited live.html.
    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    # **Every visible window the process owns, not the first one found.**
    # Stopping at the first cost an evening: Chromium owns several top-level
    # windows and the first one enumerated is not reliably the app window, so
    # the raise landed on something invisible and the real window stayed behind
    # whatever the volunteer was doing -- with `SetWindowPos` returning success
    # the whole time. Applying it to all of them is harmless for the others and
    # certain for the one that matters.
    def visit(handle, _param):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(owner))

        if owner.value == process_id and user32.IsWindowVisible(handle):
            found.append(handle)

        return True

    # The window does not exist the instant Popen returns, so wait for it
    # rather than firing once and missing.
    for _ in range(attempts):
        found.clear()
        try:
            user32.EnumWindows(visit, 0)
        except Exception:
            return
        if found:
            break
        time.sleep(0.25)

    return found


def _bring_to_front(process_id: int, attempts: int = 40) -> None:

    if sys.platform != "win32":
        return

    # Do not interrupt somebody who is using their own computer.
    #
    # The window is already open and already in the right place; this only
    # decides whether it is pulled in front. Stealing focus from a volunteer
    # mid-keystroke is the single most likely reason they stop donating the
    # machine, and it is not worth a window they can raise themselves.
    idle = _seconds_since_input()

    if idle is not None and idle < _IDLE_BEFORE_RAISE_S:
        return

    handles = _window_handles_for(process_id, attempts)

    if not handles:
        return

    import ctypes
    user32 = ctypes.windll.user32

    # Constants, named once rather than repeated as magic numbers.
    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2
    SW_SHOWNORMAL = 1
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_NOACTIVATE = 0x0010
    FLAGS = SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE

    # Let Chromium finish building its real window before raising anything.
    #
    # The enumeration above returns as soon as the process owns *a* visible
    # window, which is not reliably the app window -- it owns several and the
    # first one is often not the one with the video in it. A short settle, then
    # one more look, costs half a second at the start of a job and is the
    # difference between raising the window and raising something invisible.
    time.sleep(0.6)

    try:
        found.clear()
        user32.EnumWindows(visit, 0)
    except Exception:
        return

    if not found:
        return

    # **Once, when the window opens. Never again.**
    #
    # A watched job should come to the front when it starts, so the volunteer
    # sees it appear -- and then stay out of the way. An earlier version left
    # the windows permanently topmost and re-asserted it in a loop, which is a
    # screen saver that will not let you work: every few seconds it climbs back
    # over whatever you are doing. One raise at open, then the window takes its
    # chances in the Z-order like anything else.
    #
    # `HWND_TOPMOST` followed immediately by `HWND_NOTOPMOST` is what lifts it
    # above every ordinary window without pinning it there.
    for handle in list(handles):
        try:
            user32.ShowWindow(handle, SW_SHOWNORMAL)
            user32.SetWindowPos(handle, HWND_TOPMOST, 0, 0, 0, 0, FLAGS)
            user32.SetWindowPos(handle, HWND_NOTOPMOST, 0, 0, 0, 0, FLAGS)
            user32.BringWindowToTop(handle)
        except Exception:
            # Best effort. A window that will not raise is worth less than the
            # job it is showing, so nothing here may raise.
            pass

    # Focus, last. `SetForegroundWindow` is refused from a process that does not
    # already own the foreground -- which a background worker never does, and
    # the refusal is a `False` return nobody reads. Attaching this thread's
    # input queue to the foreground window's thread makes Windows treat the call
    # as coming from the active application. Detached again immediately: left
    # attached, the two threads share input state for the life of the process.
    try:
        foreground = user32.GetForegroundWindow()
        our_thread = ctypes.windll.kernel32.GetCurrentThreadId()
        their_thread = user32.GetWindowThreadProcessId(foreground, None)

        if their_thread and their_thread != our_thread:
            user32.AttachThreadInput(our_thread, their_thread, True)
            try:
                user32.SetForegroundWindow(handles[0])
            finally:
                user32.AttachThreadInput(our_thread, their_thread, False)
        else:
            user32.SetForegroundWindow(handles[0])
    except Exception:
        pass


# close_orphaned_windows()
# Ends watch windows left behind by a previous run of this worker.
# Inputs: the directory job workspaces live under.
# Output: how many processes were asked to stop.
# Use this once at start, before taking work.
#
# `WatchDisplay.close()` only runs in the process that owns the display, so a
# worker that was killed -- by a crash, a reaper, or somebody closing the
# terminal -- leaves its windows alive with nothing left to close them. The new
# process knows nothing about them, so they sit there forever showing the last
# frame they drew, labels and all.
#
# That is not merely untidy. A frozen window keeps whatever was on its canvas,
# so a window from a job that ran a coral model goes on showing coral names
# while the machine runs something else entirely -- and somebody looking at the
# screen sees a label that is wrong for the work being done, with no way to tell
# it is a ghost.
#
# Matched on the workspace root, so it can only ever reach this worker's own
# windows: the volunteer's browser and any other worker's windows use different
# profile paths.
def close_orphaned_windows(workspace_root: Path) -> int:

    if sys.platform != "win32":
        return 0

    needle = str(Path(workspace_root).resolve()).replace("\\", "\\\\").replace("'", "''")
    query = (
        "SELECT ProcessId FROM Win32_Process WHERE Name = 'chrome.exe' "
        f"AND CommandLine LIKE '%{needle}%'"
    )

    try:
        found = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             f"Get-CimInstance -Query \"{query}\" | ForEach-Object {{ $_.ProcessId }}"],
            capture_output=True, text=True, timeout=20, check=False,
        )
    except Exception:
        return 0

    stopped = 0

    for line in (found.stdout or "").split():
        if not line.strip().isdigit():
            continue
        subprocess.run(
            ["taskkill", "/PID", line.strip(), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        stopped += 1

    return stopped


class _FrameChannel:
    """One-slot ordered channel whose consumer explicitly acknowledges a draw."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.packet: dict[str, Any] | None = None
        self.acknowledged: int | None = None
        self.closed = False
        self.browser_connected = False

        # Frames the window never saw, because a newer one arrived first. Not a
        # fault: it is the display keeping up with inference rather than the
        # other way round, and the count is worth having so "the window looks
        # jumpy" can be answered with a number.
        self.dropped = 0

        # When a frame was last offered, and when the window last came to
        # collect one. Their difference is how long the browser has been
        # silent, which is the only remaining way to notice it has gone.
        self.last_offered_at = 0.0
        self.last_collected_at = 0.0

    def next(self, acknowledged: int | None) -> dict[str, Any] | None:
        with self.condition:
            self.browser_connected = True
            self.last_collected_at = time.monotonic()

            if acknowledged is not None:
                self.acknowledged = acknowledged
                if self.packet and self.packet["frame_number"] == acknowledged:
                    self.packet = None

            # Notified unconditionally, not only when an acknowledgement came.
            # The browser's *first* request carries no acknowledgement, and it
            # is the one the opening publisher is waiting on -- without this it
            # waits out its connect timeout with the window already open, which
            # put a minute of nothing at the start of every watched job.
            self.condition.notify_all()

            self.condition.wait_for(lambda: self.packet is not None or self.closed)
            return self.packet

    # publish()
    # Offers a frame to the window without waiting for it to be drawn.
    # Inputs: the packet, and how long a silent browser is tolerated.
    # Output: True while the display is alive; False once it has detached.
    #
    # **The display must never set the pace of inference.** This used to wait
    # for the browser to acknowledge every single frame before returning, so a
    # run went exactly as fast as Chromium could draw: 55-75 f/s headless on a
    # 4080 became 28 with a window, and four windowed slots together produced
    # about what one headless slot did. On a smaller card each window ran at
    # 0.7x real time -- a screen saver that made the machine slower than the
    # video it was showing.
    #
    # Now the newest frame simply replaces whatever the window has not yet
    # collected. Inference never blocks; the window shows the most recent frame
    # it can get and misses the ones in between, which is what watching is for.
    # Nothing scientific is lost -- the result comes from the decoded frame, and
    # this is a JPEG of it for a person to look at.
    # wants_frame()
    # Whether the window is ready for another picture.
    # Inputs: how long a silent browser is tolerated.
    # Output: True to encode and publish, False to skip this frame.
    #
    # **Asked before the frame is encoded, which is the whole point.** A JPEG of
    # a 1080p frame plus its base64 is real CPU work, it was being done on every
    # frame, and since the channel started dropping frames most of that work was
    # thrown away immediately afterwards. The window can only draw as fast as it
    # can draw; everything encoded above that rate is heat.
    #
    # Liveness is checked here too. Nothing blocks on an acknowledgement any
    # more, and a window that has stopped collecting would otherwise never be
    # noticed -- the run would skip every frame forever and call it healthy.
    def wants_frame(self, timeout_s: float = _FRAME_ACK_TIMEOUT_S) -> bool:
        with self.condition:
            if self.closed:
                return False

            # The first frame always goes: it is what opens the window, and
            # `publish` waits for the browser on it.
            if not self.browser_connected:
                return True

            # Ready for the next one.
            if self.packet is None:
                return True

            # Still holding one the window has not taken. Skip -- unless it has
            # been silent long enough to be gone.
            if self.last_collected_at:
                if time.monotonic() - self.last_collected_at > timeout_s:
                    self.close()

            return False

    def publish(
        self,
        packet: dict[str, Any],
        timeout_s: float = _FRAME_ACK_TIMEOUT_S,
        connect_timeout_s: float = 60.0,
    ) -> bool:
        with self.condition:
            if self.closed:
                return False

            # The first frame still waits, and only the first. A browser that
            # never arrives is a real failure worth detaching over, and without
            # this the run would carry on pushing frames into a window that was
            # never opened.
            if not self.browser_connected:
                self.packet = packet
                self.condition.notify_all()

                if not self.condition.wait_for(
                    lambda: self.browser_connected or self.closed,
                    timeout=connect_timeout_s,
                ):
                    self.close()
                    return False

                return not self.closed

            # Connected. Replace whatever is pending rather than waiting for it
            # to be taken: the window wants the latest picture, not a queue of
            # stale ones.
            if self.packet is not None:
                self.dropped += 1

            self.packet = packet
            self.last_offered_at = time.monotonic()
            self.condition.notify_all()

            # Liveness without blocking. A browser that has stopped asking for
            # frames is gone -- closed, crashed, or hung -- and detaching lets
            # the job carry on headless instead of drawing into nothing. The
            # old code noticed this by timing out on the acknowledgement it no
            # longer waits for, so the check has to be made explicitly.
            silent_for = self.last_offered_at - self.last_collected_at

            if self.last_collected_at and silent_for > timeout_s:
                self.close()
                return False

            return True

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class _WatchServer(ThreadingHTTPServer):
    channel: _FrameChannel
    player_root: Path
    render_proof_path: Path
    requests_seen: list[str]


class _Handler(BaseHTTPRequestHandler):
    server: _WatchServer

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        request_path = urlsplit(self.path).path
        self.server.requests_seen.append(f"GET {request_path}")
        relative = "live.html" if request_path == "/" else request_path.lstrip("/")
        candidate = (self.server.player_root / relative).resolve()
        root = self.server.player_root.resolve()
        if root not in candidate.parents and candidate != root:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        # The logo is served from here too, so the window and its taskbar entry
        # carry the MARP mark. Without an image type it went out as JavaScript
        # and the browser refused to draw it.
        content_type = {
            ".html": "text/html",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }.get(candidate.suffix, "text/javascript")
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        request_path = urlsplit(self.path).path
        self.server.requests_seen.append(f"POST {request_path}")
        if request_path == "/render-proof":
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 10 * 1024 * 1024:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            body = self.rfile.read(size)
            if not self.server.render_proof_path.exists():
                self.server.render_proof_path.write_bytes(body)
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if request_path != "/next":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size) or b"{}")
            acknowledged = payload.get("acknowledged_frame")
            acknowledged = int(acknowledged) if acknowledged is not None else None
            packet = self.server.channel.next(acknowledged)
        except (ValueError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST)
            return

        if packet is None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        body = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class WatchDisplay:
    """Own one job's server, Chromium process, and ordered frame channel."""

    def __init__(
        self,
        screen_mode: str,
        workspace: Path,
        warn: Callable[[str], None],
        job_id: str | None = None,
        attempt_id: str | None = None,
        model_name: str | None = None,
        species_names: list[str] | None = None,
        slot_index: int = 0,
        slot_count: int = 1,
    ) -> None:
        self._mode = screen_mode
        # Which of the worker's windows this is, and how many there are. Only
        # the layout uses them: a slot has to know about its siblings to be
        # tiled beside them rather than on top of them.
        self._slot_index = slot_index
        self._slot_count = slot_count
        # The job child may run with its workspace as the current directory.
        # Chromium must receive an absolute profile path or a relative worker
        # state directory is applied twice and the viewer never connects.
        self._workspace = workspace.resolve()
        self._warn = warn
        self._job_id = job_id
        self._attempt_id = attempt_id
        self._profile_dir: Path | None = None
        self._model_name = model_name
        self._species_names = list(species_names or [])
        self._channel = _FrameChannel()
        self._server: _WatchServer | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_path = self._workspace / "chromium.stderr.log"
        self._warned = False
        self._closing = False
        # Frames never encoded, because the window was still holding the last
        # one. Not a fault: it is the measure of how much work the display has
        # stopped doing.
        self._skipped = 0

    @staticmethod
    def _install_root() -> Path:
        return Path(sys.executable).resolve().parent

    # _packaged_player_root()
    # Where the watch page ships.
    # Inputs: none.
    # Output: the directory holding `live.html`.
    # Use this as the default rather than an installer-staged directory. The
    # page is part of this package, so a git checkout and an install both have
    # it; when it lived only in the installer payload it was in neither, and the
    # window had never opened on any machine.
    @staticmethod
    def _packaged_player_root() -> Path:
        return Path(__file__).resolve().parent

    # _find_chromium()
    # Finds a Chromium-family browser to draw in.
    # Inputs: none.
    # Output: the executable, or None when the machine has none.
    # Use this so a volunteer who has Chrome or Edge -- which is every ordinary
    # Windows machine -- gets a window without the installer's bundled copy.
    # Explicit configuration still wins, and the bundle is still preferred over
    # a browser the volunteer uses for their own browsing.
    @staticmethod
    def _find_chromium() -> Path | None:
        configured = os.environ.get("MARP_CHROMIUM_PATH")
        if configured:
            candidate = Path(configured).resolve()
            return candidate if candidate.is_file() else None

        candidates = [WatchDisplay._install_root() / "chromium" / "chrome.exe"]

        if sys.platform == "win32":
            program_files = [
                os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
                os.environ.get("LOCALAPPDATA", ""),
            ]
            for base in filter(None, program_files):
                candidates.append(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe")
                candidates.append(Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe")

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    def start(self) -> bool:
        try:
            player_root = Path(
                os.environ.get("MARP_PLAYER_HOST_DIR")
                or self._packaged_player_root()
            ).resolve()
            if not (player_root / "live.html").is_file():
                raise FileNotFoundError(f"live player is missing from {player_root}")

            chromium = self._find_chromium()
            if chromium is None:
                raise FileNotFoundError(
                    "no Chromium, Chrome or Edge was found to draw the watch window in; "
                    "set MARP_CHROMIUM_PATH to one"
                )

            self._server = _WatchServer(("127.0.0.1", 0), _Handler)
            self._server.channel = self._channel
            self._server.player_root = player_root
            self._server.render_proof_path = self._workspace / "browser-render-proof.jpg"
            self._server.requests_seen = []
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
            port = self._server.server_address[1]
            profile = self._workspace / "chromium-profile"
            # Remembered so close() can find the browser by the profile
            # it holds, which survives Chromium changing process.
            self._profile_dir = profile
            # The worker's own API port travels on the URL. The page needs it
            # for the footer's machine figures, and the display server and the
            # worker API are two servers on two ports -- so the page cannot
            # infer it from where it was loaded. Without this it falls back to
            # the default, which is right for one worker and silently wrong for
            # a second one: an empty footer rather than an error.
            api_port = os.environ.get("MARP_WORKER_API_PORT", "8010")
            args = _chromium_args(
                chromium,
                f"http://127.0.0.1:{port}/?api={api_port}",
                profile,
                self._mode,
                self._slot_index,
                self._slot_count,
            )
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            # Keep Chromium diagnostics with the attempt. A blank app window used
            # to leave no evidence about navigation or page initialization.
            with self._stderr_path.open("wb") as chromium_stderr:
                self._process = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=chromium_stderr,
                    creationflags=creationflags,
                )
            threading.Thread(target=self._watch_process, daemon=True).start()

            # Off-thread: raising the window waits for it to appear, and the
            # job must not wait for that.
            threading.Thread(
                target=_bring_to_front,
                args=(self._process.pid,),
                daemon=True,
            ).start()

            # And a second thread to keep it where it belongs as the layout
            # changes. A window is tiled for the number of windows there were
            # when it opened, so the ones already up keep a stale layout when
            # another job starts -- a third window simply landed on top of one
            # of the first two. Nothing else knows the count has changed, so
            # each window watches for it and moves itself.
            threading.Thread(
                target=self._follow_layout,
                daemon=True,
            ).start()
            return True
        except Exception as error:
            self._detach(f"watch display could not start: {type(error).__name__}: {error}")
            return False

    def _watch_process(self) -> None:
        assert self._process is not None
        exit_code = self._process.wait()
        if not self._closing:
            self._detach(
                f"watch window closed with exit code {exit_code}; inference is continuing headless"
            )

    def _diagnostics(self) -> str:
        requests_seen = list(self._server.requests_seen) if self._server is not None else []
        request_summary = ", ".join(requests_seen[-8:]) or "none"
        stderr_tail = ""
        try:
            stderr_tail = self._stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
        except OSError:
            pass
        return f"requests=[{request_summary}]; chromium_stderr={stderr_tail or 'empty'}"

    def _detach(self, message: str) -> None:
        if not self._warned:
            self._warn(f"{message}; {self._diagnostics()}")
            self._warned = True
        self._channel.close()

    # _follow_layout()
    # Keeps this window in its tile as the number of windows changes.
    # Inputs: none; reads the worker's own API.
    # Output: none; runs until the display closes.
    #
    # The worker decides how many jobs to run from measured throughput, so the
    # count changes while jobs are running. A window tiled for two does not
    # move when a third opens, and the third is laid out for three -- so it
    # lands on top of one of them. The layout has to be re-applied to the
    # windows that are already up, and this is the only thing that knows which
    # window belongs to which slot.
    #
    # Reads `permitted` rather than `total`: the ceiling is what the machine
    # might grow to, and tiling for it leaves most of the screen empty.
    def _follow_layout(self) -> None:

        if sys.platform != "win32" or self._process is None:
            return

        import json as _json
        import urllib.request

        port = os.environ.get("MARP_WORKER_API_PORT", "8010")
        url = f"http://127.0.0.1:{port}/status"
        applied = self._slot_count

        while not self._channel.closed:
            time.sleep(5.0)

            try:
                with urllib.request.urlopen(url, timeout=3) as answer:
                    status = _json.loads(answer.read().decode("utf-8"))

                permitted = int((status.get("slots") or {}).get("permitted") or 0)
            except Exception:
                # The worker API being briefly unreachable is not a reason to
                # stop watching the layout.
                continue

            if permitted < 1 or permitted == applied:
                continue

            # The count moved. Re-tile for what there are now.
            applied = permitted
            _move_windows(self._process.pid, _tile(self._slot_index, permitted))

    def present(
        self,
        frame: Any,
        frame_number: int,
        tracks: list[dict[str, Any]],
        range_start: int | None = None,
        range_end: int | None = None,
    ) -> bool:
        if self._channel.closed:
            return False

        # Skip before encoding, not after.
        #
        # The window is still holding the last picture, so this one would be
        # dropped the moment it was offered -- and encoding it first meant a
        # full JPEG plus base64 of a 1080p frame for nothing, on every frame
        # above the rate the window can draw. Returning True because the
        # display is alive and well; it simply does not need this frame.
        if not self._channel.wants_frame():
            if self._channel.closed:
                self._detach("watch window stopped responding; inference is continuing headless")
                return False

            self._skipped += 1
            return True

        # The display is observational, while the scientific result is produced
        # from the original frame. JPEG keeps 1080p loopback transfers fast.
        encoded, bytes_buffer = cv2.imencode(
            ".jpg",
            frame.image,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        if not encoded:
            self._detach("watch display could not encode a frame; inference is continuing headless")
            return False
        packet = {
            "frame_number": int(frame_number),
            "range_start": int(range_start) if range_start is not None else None,
            "range_end": int(range_end) if range_end is not None else None,
            "job_id": self._job_id,
            "attempt_id": self._attempt_id,
            "model_name": self._model_name,
            "species_names": self._species_names,
            "content_type": "image/jpeg",
            "image": base64.b64encode(bytes_buffer.tobytes()).decode("ascii"),
            "tracks": tracks,
        }
        presented = self._channel.publish(packet)
        if not presented:
            self._detach("watch window stopped responding; inference is continuing headless")
        return presented

    # _kill_by_profile()
    # Ends any browser still holding this job's profile directory.
    # Inputs: none.
    # Output: none.
    # Use this from close(), before the pid-based kill. Best effort throughout:
    # a window that will not die must not stop a job being reported.
    def _kill_by_profile(self) -> None:

        if sys.platform != "win32" or self._profile_dir is None:
            return

        # Matched on the profile path, which is this attempt's own directory, so
        # this can never reach the volunteer's own browser or another slot's
        # window. Quoted for the WMI query, and backslashes doubled because
        # `LIKE` treats one as an escape.
        needle = str(self._profile_dir).replace("\\", "\\\\").replace("'", "''")
        query = (
            "SELECT ProcessId FROM Win32_Process WHERE Name = 'chrome.exe' "
            f"AND CommandLine LIKE '%{needle}%'"
        )

        try:
            found = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 f"Get-CimInstance -Query \"{query}\" | "
                 "ForEach-Object { $_.ProcessId }"],
                capture_output=True, text=True, timeout=20, check=False,
            )
        except Exception:
            return

        for line in (found.stdout or "").split():
            if not line.strip().isdigit():
                continue
            subprocess.run(
                ["taskkill", "/PID", line.strip(), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )

    def close(self) -> None:
        self._closing = True
        self._channel.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        # Kill whatever is showing this job's profile, whether or not the pid we
        # launched is still alive.
        #
        # Chromium does not always stay in the process it was started as: it can
        # hand the window to another process and exit, and then `poll()` says
        # the launched process is gone while the window is still on screen. The
        # old code read that as "already closed" and killed nothing, so the
        # window outlived its job -- showing a frozen frame and an error from a
        # server that had shut down underneath it. The profile directory is
        # unique to this job, so anything holding it is ours.
        self._kill_by_profile()

        if self._process is not None and self._process.poll() is None:
            if sys.platform == "win32":
                # Installed Chrome owns a renderer tree. Terminating only its
                # first process leaves the visible app window orphaned.
                subprocess.run(
                    ["taskkill", "/PID", str(self._process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
