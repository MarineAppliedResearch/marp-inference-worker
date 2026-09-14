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
        timeout_s: float = 15.0,
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


class _Handler(BaseHTTPRequestHandler):
    server: _WatchServer

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        request_path = urlsplit(self.path).path
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
        if urlsplit(self.path).path != "/next":
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
    ) -> None:
        self._mode = screen_mode
        self._workspace = workspace
        self._warn = warn
        self._channel = _FrameChannel()
        self._server: _WatchServer | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._warned = False
        self._closing = False

    @staticmethod
    def _install_root() -> Path:
        return Path(sys.executable).resolve().parent

    def start(self) -> bool:
        try:
            player_root = Path(
                os.environ.get("MARP_PLAYER_HOST_DIR")
                or self._install_root() / "player"
            ).resolve()
            chromium = Path(
                os.environ.get("MARP_CHROMIUM_PATH")
                or self._install_root() / "chromium" / "chrome.exe"
            ).resolve()
            if not (player_root / "live.html").is_file():
                raise FileNotFoundError(f"live player is missing from {player_root}")
            if not chromium.is_file():
                raise FileNotFoundError(f"Chromium is missing at {chromium}")

            self._server = _WatchServer(("127.0.0.1", 0), _Handler)
            self._server.channel = self._channel
            self._server.player_root = player_root
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
            port = self._server.server_address[1]
            profile = self._workspace / "chromium-profile"
            args = [
                str(chromium),
                f"--app=http://127.0.0.1:{port}/",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--disable-sync",
                "--mute-audio",
            ]
            if self._mode == "fullscreen":
                args.append("--start-fullscreen")
            else:
                # Let Windows cascade independent job windows instead of
                # stacking maximized surfaces directly on top of each other.
                args.append("--window-size=1100,700")
            self._process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            threading.Thread(target=self._watch_process, daemon=True).start()
            return True
        except Exception as error:
            self._detach(f"watch display could not start: {type(error).__name__}: {error}")
            return False

    def _watch_process(self) -> None:
        assert self._process is not None
        self._process.wait()
        if not self._closing:
            self._detach("watch window closed; inference is continuing headless")

    def _detach(self, message: str) -> None:
        if not self._warned:
            self._warn(message)
            self._warned = True
        self._channel.close()

    def present(self, frame: Any, frame_number: int, tracks: list[dict[str, Any]]) -> bool:
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
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
