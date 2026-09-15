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
    assert "--window-position=50,50" in args
    assert "--disable-gpu" in args
    assert "--disable-gpu-compositing" in args
    assert "--use-angle=swiftshader" not in args
    assert "--start-fullscreen" in args


def test_frame_channel_waits_for_browser_ack_before_accepting_the_next_frame() -> None:
    channel = _FrameChannel()
    published: list[tuple[int, bool]] = []

    def publish(frame_number: int) -> None:
        result = channel.publish({"frame_number": frame_number}, timeout_s=2)
        published.append((frame_number, result))

    first = threading.Thread(target=publish, args=(41,))
    second = threading.Thread(target=publish, args=(42,))
    first.start()
    assert channel.next(None) == {"frame_number": 41}

    second.start()
    time.sleep(0.02)
    assert published == []

    assert channel.next(41) == {"frame_number": 42}
    first.join(timeout=1)
    assert published == [(41, True)]

    channel.close()
    second.join(timeout=1)
    assert published == [(41, True), (42, False)]


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
