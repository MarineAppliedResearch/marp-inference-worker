"""Tests for the child-owned ordered watch channel."""

import threading
import time

from marp_inference_worker.watch.display import _FrameChannel


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
