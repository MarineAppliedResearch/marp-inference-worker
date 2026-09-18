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
# The rule Isaac asked for: fill the monitors first, and only start subdividing
# when there are more windows than screens.
#
#   slots <= monitors : one window per monitor, filling it
#   slots  > monitors : slots shared out over the monitors, then gridded
#                       within each one
#
# A grid rather than a cascade because these are watched, not clicked: a
# volunteer glancing across the room wants to see every running job at once, and
# overlapping windows hide all but the top one.
def _tile(slot_index: int, slot_count: int) -> tuple[int, int, int, int]:

    monitors = _monitors()
    count = max(1, slot_count)
    index = max(0, slot_index) % count

    # A configured position names the watching screen, not just a coordinate.
    #
    # `MARP_WATCH_WINDOW_POSITION` exists because windows kept opening on the
    # monitor somebody was working on. A tiler that then helpfully fills every
    # monitor puts them straight back. So when a position is set, every slot is
    # tiled inside the monitor that position falls on and the others are left
    # alone -- the person has said which screen is for watching.
    anchor = _window_position()

    if anchor:
        x_text, y_text = anchor.split(",")
        anchor_point = (int(x_text), int(y_text))
        chosen = _monitor_containing(anchor_point, monitors)

        return _grid_within(chosen, index, count)

    # Fewer windows than screens: one each, filling the monitor.
    if count <= len(monitors):
        return monitors[index]

    # More windows than screens. Share them out as evenly as possible, giving
    # the earlier monitors the extra one when it does not divide -- the primary
    # is usually the larger and the one being looked at.
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
def _bring_to_front(process_id: int, attempts: int = 40) -> None:

    if sys.platform != "win32":
        return

    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: list[int] = []

    # EnumWindows rather than a title match: the page title is ours to change
    # and matching on it would break the moment somebody edited live.html.
    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(handle, _param):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(owner))
        if owner.value == process_id and user32.IsWindowVisible(handle):
            found.append(handle)
            return False
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

    if not found:
        return

    handle = found[0]
    try:
        # Shown, not maximised.
        #
        # This was SW_MAXIMIZE, from before the windows were tiled, and the two
        # cannot both win: maximising fills whichever monitor the window landed
        # on and throws away the size it was given, so four tiles meant to sit
        # in a 2x2 grid all ended up stacked full-screen on one monitor with
        # only the last one visible. The geometry is already the whole monitor
        # when there is a screen per slot, which is what maximising was for.
        user32.ShowWindow(handle, 1)          # SW_SHOWNORMAL
        user32.SetForegroundWindow(handle)
        user32.BringWindowToTop(handle)
    except Exception:
        # Windows refuses SetForegroundWindow from a process that does not own
        # the foreground. The window is still placed and still there.
        pass


class _FrameChannel:
    """One-slot ordered channel whose consumer explicitly acknowledges a draw."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.packet: dict[str, Any] | None = None
        self.acknowledged: int | None = None
        self.closed = False
        self.browser_connected = False

    def next(self, acknowledged: int | None) -> dict[str, Any] | None:
        with self.condition:
            self.browser_connected = True
            if acknowledged is not None:
                self.acknowledged = acknowledged
                if self.packet and self.packet["frame_number"] == acknowledged:
                    self.packet = None
                self.condition.notify_all()
            self.condition.wait_for(lambda: self.packet is not None or self.closed)
            return self.packet

    def publish(
        self,
        packet: dict[str, Any],
        timeout_s: float = _FRAME_ACK_TIMEOUT_S,
        connect_timeout_s: float = 60.0,
    ) -> bool:
        with self.condition:
            if not self.condition.wait_for(
                lambda: self.packet is None or self.closed,
                timeout=timeout_s,
            ):
                self.close()
                return False
            if self.closed:
                return False
            waiting_for_browser = not self.browser_connected
            self.packet = packet
            self.condition.notify_all()
            if not self.condition.wait_for(
                lambda: self.acknowledged == packet["frame_number"] or self.closed,
                timeout=connect_timeout_s if waiting_for_browser else timeout_s,
            ):
                self.close()
                return False
            return not self.closed

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
        self._model_name = model_name
        self._species_names = list(species_names or [])
        self._channel = _FrameChannel()
        self._server: _WatchServer | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_path = self._workspace / "chromium.stderr.log"
        self._warned = False
        self._closing = False

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

    def close(self) -> None:
        self._closing = True
        self._channel.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
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
