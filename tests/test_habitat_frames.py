# test_habitat_frames.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Tests for which frames a habitat run looks at, and which part of each
# (MarineAppliedResearch/MARP_API#287). No video and no model: the sampler is fed
# made-up frames, and the crop is cut from a small array.

# numpy builds the small images the crop is cut from.
import numpy as np

# pytest checks the refusal of a rate that would keep nothing.
import pytest

# The reader's frame type, and the module under test.
from marp_inference_worker.media.frame_range_reader import DecodedFrame
from marp_inference_worker.media.habitat_frames import (
    HABITAT_CROP,
    CropBox,
    crop_fraction,
    keyframe_box,
    sample_by_time,
)


# frames_at()
# Made-up frames at a steady rate, numbered as the reader numbers them.
# Inputs: frames per second and how many seconds.
# Output: a list of DecodedFrame with no image.
def frames_at(rate: float, seconds: float, start_s: float = 0.0) -> list[DecodedFrame]:

    # One frame per tick of the rate, from start_s.
    count = int(round(rate * seconds))
    return [DecodedFrame(index=round((start_s + i / rate) * rate), time_s=start_s + i / rate, image=None) for i in range(count)]


# test_keeps_three_a_second_at_25_fps()
# 25 frames a second, 3 kept a second: about every eighth frame, never two close together.
def test_keeps_three_a_second_at_25_fps():

    # Two seconds of 25 fps video.
    kept = list(sample_by_time(iter(frames_at(25, 2))))

    # Six kept, a third of a second apart to within one frame.
    assert len(kept) == 6
    gaps = [b.time_s - a.time_s for a, b in zip(kept, kept[1:])]
    assert all(abs(g - 1 / 3) <= 1 / 25 for g in gaps)


# test_keeps_three_a_second_at_29_97_fps()
# The same by time at the drop-frame rate, which a count of frames would not give.
def test_keeps_three_a_second_at_29_97_fps():

    # Ten seconds of 29.97 fps video.
    kept = list(sample_by_time(iter(frames_at(29.97, 10))))

    # Thirty kept, and none drifting: the last is due near 9.67 s.
    assert len(kept) == 30
    assert abs(kept[-1].time_s - 29 / 3) <= 1 / 29.97


# test_starts_at_the_first_frame_of_the_range()
# A range that begins mid-video starts its schedule there.
def test_starts_at_the_first_frame_of_the_range():

    # One second of video from 100 s in.
    kept = list(sample_by_time(iter(frames_at(25, 1, start_s=100.0))))

    # The first kept frame is the range's first.
    assert kept[0].time_s == 100.0
    assert len(kept) == 3


# test_a_gap_restarts_the_schedule_rather_than_catching_up()
# After a jump in the video's timestamps, frames are not kept back to back to catch up.
def test_a_gap_restarts_the_schedule_rather_than_catching_up():

    # One second, then a five-second jump, then one more second.
    frames = frames_at(25, 1) + frames_at(25, 1, start_s=6.0)
    kept = list(sample_by_time(iter(frames)))

    # Three before the jump and three after, the first after at 6 s.
    assert [round(f.time_s, 2) for f in kept[3:]][0] == 6.0
    assert len(kept) == 6


# test_refuses_a_rate_that_keeps_nothing()
# A rate of zero would loop on no frames for ever.
def test_refuses_a_rate_that_keeps_nothing():

    # Asked for nothing, the sampler says so before reading a frame.
    with pytest.raises(ValueError):
        list(sample_by_time(iter(frames_at(25, 1)), per_second=0))


# test_crop_is_the_bottom_half_and_middle_half_of_the_width()
# The habitat crop, x 0.25-0.75 and y 0.5-1.0, as pixels of a 1920 x 1080 frame.
def test_crop_is_the_bottom_half_and_middle_half_of_the_width():

    # Each pixel holds its own row and column, so the crop's corners can be read back.
    rows, cols = np.mgrid[0:1080, 0:1920]
    image = np.stack([rows, cols], axis=-1)
    cut = crop_fraction(image)

    # 540 rows from 540, 960 columns from 480.
    assert cut.shape[:2] == (540, 960)
    assert tuple(cut[0, 0]) == (540, 480)
    assert tuple(cut[-1, -1]) == (1079, 1439)
    assert HABITAT_CROP == CropBox(0.25, 0.5, 0.75, 1.0)


# test_keyframe_box_is_the_crop_centre_origin()
# The box a habitat stretch's keyframes carry, so the Mosaic cuts the same patch.
def test_keyframe_box_is_the_crop_centre_origin():

    # Centre-origin and normalised, as MARP keyframes hold boxes.
    assert keyframe_box() == {"x": 0.5, "y": 0.75, "width": 0.5, "height": 0.5}
