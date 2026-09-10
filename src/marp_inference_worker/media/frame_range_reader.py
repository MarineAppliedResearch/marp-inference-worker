# frame_range_reader.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Frame-range decoding for the MARP Inference Worker.
# A job covers a frame range of a video, not a whole video, so this file yields
# the frames of exactly that range and nothing else (R9). It is a generator: one
# decoded frame is alive at a time and the video is never accumulated (R8).
# Decoding belongs here; detection, tracking and reduction do not.

# Iterator types the generator this module exists to produce.
from typing import Any, Iterator, NamedTuple


# DecodedFrame
# One frame of video with the numbers a downstream stage needs.
# `index` is the frame's absolute position in the video, not its position within
# the job's range, because MARP's observation rows are numbered against the video.
class DecodedFrame(NamedTuple):

    # Absolute frame index in the source video.
    index: int

    # Presentation time in seconds from the start of the video.
    time_s: float

    # The decoded BGR image array, as OpenCV produced it.
    image: Any


# VideoGeometry
# The fixed facts about a video that every stage needs.
# Read once when the capture opens, because asking OpenCV per frame is both
# slower and, on some streams, wrong.
class VideoGeometry(NamedTuple):

    # Frame width in pixels.
    width: int

    # Frame height in pixels.
    height: int

    # Frames per second, floored to at least 1 -- some streams report 0.
    frame_rate: float

    # Total frames the container claims, or 0 when it will not say.
    total_frames: int


# open_video()
# Opens a video or stream and reads its geometry.
# Inputs: a local path or stream url OpenCV can open.
# Output: the OpenCV capture and its geometry.
# Use this before iterating frames, and release the capture when finished.
def open_video(source: str) -> tuple[Any, VideoGeometry]:

    # Imported lazily so importing this module does not pull in OpenCV, which
    # matters for the API tests that never decode anything.
    import cv2

    # Open the source; OpenCV accepts both local paths and http(s) stream urls.
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    # Read geometry once. A frame rate of 0 has been seen on Jellyfin streams,
    # and dividing by it later produces infinities in every timecode.
    reported_rate = capture.get(cv2.CAP_PROP_FPS)
    geometry = VideoGeometry(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_rate=float(max(1.0, reported_rate)),
        total_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return capture, geometry


# iter_frame_range()
# Yields the frames of one half-open range, one at a time.
# Inputs: an open capture, its geometry, and the start and end frame indices.
# Output: iterator of DecodedFrame.
# Use this as the only way a job reads frames. It seeks once to the range start
# and then reads sequentially, because seeking per frame is far slower and, on a
# long-GOP stream, lands on the wrong frame.
def iter_frame_range(
    capture: Any,
    geometry: VideoGeometry,
    start_frame: int,
    end_frame: int,
) -> Iterator[DecodedFrame]:

    # Imported lazily for the same reason open_video() does.
    import cv2

    # Seeking to 0 is a no-op that some backends still mishandle, so skip it.
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Track position ourselves. CAP_PROP_POS_FRAMES is not reliable on every
    # backend after a seek, and the absolute index is what observations record.
    frame_index = start_frame

    # Read forwards until the range is done or the video ends.
    while frame_index < end_frame:

        # One frame in flight at a time; nothing is retained after the yield.
        read_ok, image = capture.read()

        # A short read means the video ended before end_frame. That is not an
        # error -- a range can be clipped by the true length -- so stop cleanly
        # and let the caller see how many frames actually arrived.
        if not read_ok:
            return

        # Time is derived from the index and the frame rate rather than read
        # from the capture, so it stays consistent across a seek.
        yield DecodedFrame(
            index=frame_index,
            time_s=frame_index / geometry.frame_rate,
            image=image,
        )

        frame_index += 1
