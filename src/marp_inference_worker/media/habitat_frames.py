# habitat_frames.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Which frames a habitat run looks at, and which part of each.
# A habitat model scores the seabed a few times a second rather than every frame,
# on one fixed patch of the picture: the habitat crop, the bottom half of the frame
# and the middle half of its width (MarineAppliedResearch/MARP_API#287).
# Picking frames by time and cutting the crop belong here; scoring them and turning
# scores into stretches do not.

# Iterator types for the filter this module exists to provide.
from collections.abc import Iterator
from typing import Any, NamedTuple

# The frames the existing reader yields, which this filters.
from marp_inference_worker.media.frame_range_reader import DecodedFrame


# HABITAT_SAMPLES_PER_SECOND
# How often a habitat run looks at the video: 3 frames a second, settled on
# MARP_API#287 from the feasibility runs on whole lines.
HABITAT_SAMPLES_PER_SECOND = 3.0


# CropBox
# A part of the frame as fractions of its width and height, left, top, right, bottom.
# Fractions rather than pixels, so one box fits every video resolution.
class CropBox(NamedTuple):

    # Left edge as a fraction of the frame width.
    x0: float

    # Top edge as a fraction of the frame height.
    y0: float

    # Right edge as a fraction of the frame width.
    x1: float

    # Bottom edge as a fraction of the frame height.
    y1: float


# HABITAT_CROP
# The habitat crop: x 0.25 to 0.75, y 0.5 to 1.0, decided by Isaac on 2026-11-10.
# The seabed in front of the ROV, clear of the overlay at the top and the dark edges.
HABITAT_CROP = CropBox(0.25, 0.5, 0.75, 1.0)


# sample_by_time()
# Keeps the frames nearest a steady rate, by playback time, from a stream of frames.
# Inputs: decoded frames in time order, and how many to keep per second.
# Output: iterator of the kept DecodedFrame, in order.
# Use this between the frame reader and the model. By time rather than by count,
# so 3 a second is 3 a second at 25 or 29.97 frames a second alike.
def sample_by_time(frames: Iterator[DecodedFrame], per_second: float = HABITAT_SAMPLES_PER_SECOND) -> Iterator[DecodedFrame]:

    # A rate of zero or less would keep nothing for ever; refuse it at once.
    if per_second <= 0:
        raise ValueError(f"per_second must be positive, not {per_second}")

    # The spacing between kept frames, in seconds.
    step = 1.0 / per_second

    # The time the next kept frame is due; set by the first frame seen.
    due: float | None = None

    # Walk the frames once; only the kept ones go on.
    for frame in frames:

        # The first frame starts the schedule, wherever the range begins.
        if due is None:
            due = frame.time_s

        # Not yet due: this frame is dropped.
        if frame.time_s < due - 1e-6:
            continue

        # Due: keep it, and move the schedule past it. After a gap in the video the
        # schedule restarts at this frame rather than keeping every frame to catch up.
        yield frame
        due += step
        if due <= frame.time_s:
            due = frame.time_s + step


# crop_fraction()
# Cuts one part of a decoded image, given as fractions of the frame.
# Inputs: an image array of shape (height, width, ...) and a CropBox.
# Output: the cropped image array, a view of the original.
# Use this to cut the habitat crop before resizing for the model.
def crop_fraction(image: Any, box: CropBox = HABITAT_CROP) -> Any:

    # Pixel edges from the fractions, as the feasibility training cut them.
    height, width = image.shape[0], image.shape[1]
    left, top = int(width * box.x0), int(height * box.y0)
    right, bottom = int(width * box.x1), int(height * box.y1)
    return image[top:bottom, left:right]


# keyframe_box()
# The crop as a keyframe box: centre-origin, normalised x, y, width and height.
# Inputs: a CropBox.
# Output: a dict with x, y, width and height, as MARP's keyframes hold them.
# Use this for the start and end keyframes a habitat stretch is given, so the
# Mosaic cuts its picture from the same patch the model looked at.
def keyframe_box(box: CropBox = HABITAT_CROP) -> dict[str, float]:

    # MARP keyframes are centre-origin, so the box is its middle and its size.
    return {
        "x": (box.x0 + box.x1) / 2,
        "y": (box.y0 + box.y1) / 2,
        "width": box.x1 - box.x0,
        "height": box.y1 - box.y0,
    }
