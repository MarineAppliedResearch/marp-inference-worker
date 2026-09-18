"""Child-owned loopback server and local Chromium window for watch mode."""

import base64
import json
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


def _chromium_args(chromium: Path, url: str, profile: Path, mode: str) -> list[str]:
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
        # New per-job profiles have no saved bounds. Pin the initial window to
        # the primary display so disconnected monitors cannot hide it.
        "--window-position=50,50",
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
    # Maximised, not a fixed 1100x700 box. A watched job is something the
    # volunteer is meant to see from across the room, and a small window behind
    # whatever they had open is a screen saver nobody knows is running.
    args.append("--start-fullscreen" if mode == "fullscreen" else "--start-maximized")
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
        user32.ShowWindow(handle, 3)          # SW_MAXIMIZE
        user32.SetForegroundWindow(handle)
        user32.BringWindowToTop(handle)
    except Exception:
        # Windows refuses SetForegroundWindow from a process that does not own
        # the foreground. The window is still maximised and still there.
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
        content_type = "text/html" if candidate.suffix == ".html" else "text/javascript"
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
    ) -> None:
        self._mode = screen_mode
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
            args = _chromium_args(
                chromium,
                f"http://127.0.0.1:{port}/",
                profile,
                self._mode,
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
