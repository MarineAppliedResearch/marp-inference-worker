# test_seek_verification.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Tests for the frame-seek verification analysis.
# The measurement itself needs a real video server and is manual step 3 in
# .marp/verification.md. What is tested here is the part that decides what the
# measurement *means*, because that is the part that got it wrong the first time:
# on a video where 547 of 979 frames are pixel-identical to another frame, a
# single-frame comparison reported offsets of -32, -33, -34 and -129 frames on a
# stream that was seeking perfectly.
#
# These tests use a fake capture, so they run in milliseconds with no stream.
# What they cannot see is whether a real backend seeks correctly -- that is
# structurally a media-tier question and the script exists for it.

# numpy builds frames whose pixel buffer can be hashed, as OpenCV's would be.
import numpy as np

# pytest supplies the parametrization.
import pytest

# The analysis under test.
from marp_inference_worker.media.seek_verification import (
    frame_identity,
    locate_run,
    measure_seek,
    sequential_identities,
    summarize,
)


# FakeCapture
# A capture that decodes a fixed list of frames and seeks with a chosen error.
# `landing_error` is what makes the interesting cases expressible: a real
# long-GOP failure lands short of the target while still *reporting* the target,
# so the fake reports the target either way and moves the read cursor by the
# error.
class FakeCapture:

    # __init__()
    # Builds a capture over a list of frames with a deliberate seek error.
    # Inputs: the frames, and how many frames short the seek lands.
    # Output: initialized FakeCapture.
    def __init__(self, frames: list, landing_error: int = 0) -> None:

        # The frames this capture can produce, in order.
        self._frames = frames

        # How far the read cursor ends up from where it was asked to be.
        self._landing_error = landing_error

        # Where the next read() will come from.
        self._position = 0

        # Whatever the capture last claimed its position was.
        self._reported = 0.0

        # So a test can assert the capture was released.
        self.released = False

    # set()
    # Seeks, landing `landing_error` frames away from the target.
    def set(self, prop: int, value: float) -> bool:

        # The reported position is the target, not where it actually landed --
        # that divergence is exactly the failure being hunted.
        self._reported = float(value)
        self._position = max(0, int(value) + self._landing_error)
        return True

    # get()
    # Returns the position the capture claims to be at.
    def get(self, prop: int) -> float:

        # A real backend answers with what it was asked for.
        return self._reported

    # read()
    # Returns the next frame, or (False, None) past the end.
    def read(self):

        # Past the end is a clean stop, not an error.
        if self._position >= len(self._frames):
            return False, None
        image = self._frames[self._position]
        self._position += 1
        return True, image

    # release()
    # Records that the capture was released.
    def release(self) -> None:

        # Asserted by a test, because a leaked stream against the video server
        # is a real cost and the measurement opens one capture per target.
        self.released = True


# _frame()
# Builds a small frame whose pixels are determined by one number.
# Inputs: the value the frame's pixels encode.
# Output: a numpy array standing in for a decoded frame.
# The value is written in as eight bytes rather than used as a fill level. A
# fill of `value % 256` wraps, which silently made a 400-frame "unique" video
# contain 144 duplicate frames and broke two of the tests below before this was
# spotted -- the same class of mistake the module under test is about.
def _frame(value: int):

    # Small on purpose; only the hash of the buffer matters here.
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image.flat[:8] = np.frombuffer(int(value).to_bytes(8, "little"), dtype=np.uint8)
    return image


# _unique_video()
# Builds a video in which every frame is distinct.
def _unique_video(length: int) -> list:

    # Distinct fill values give distinct identities.
    return [_frame(index) for index in range(length)]


# _periodic_video()
# Builds a video that repeats itself every `period` frames.
# This is the shape of the real CAMPA2021 clip that broke the first measurement.
def _periodic_video(length: int, period: int) -> list:

    # The same handful of pictures over and over, as a static ROV scene gives.
    return [_frame(index % period) for index in range(length)]


# _partly_periodic_video()
# Builds a video that repeats inside a window and is unique outside it.
# This is what the real clip is actually like, and it is the only shape that
# tells a run comparison apart from a single-frame one: a frame inside the window
# occurs at many indices, but a run that reaches past the window's end occurs at
# exactly one. A fully periodic video cannot distinguish the two, which is why
# the first version of these tests passed even with the defect reintroduced.
def _partly_periodic_video(length: int, window: range, period: int) -> list:

    # Offset the periodic fills well clear of the unique ones so the two ranges
    # of values cannot accidentally coincide.
    return [
        _frame(1_000_000 + (index % period)) if index in window else _frame(index)
        for index in range(length)
    ]


# test_frame_identity_is_the_picture_and_not_the_position()
# Two frames with the same pixels have the same identity, and that is the whole
# reason a single-frame comparison cannot locate a seek.
def test_frame_identity_is_the_picture_and_not_the_position():

    # Same picture, two different positions in the video.
    assert frame_identity(_frame(7)) == frame_identity(_frame(7))

    # Different pictures are told apart.
    assert frame_identity(_frame(7)) != frame_identity(_frame(8))


# test_locate_run_reports_every_position_a_run_occurs_at()
# On periodic content a run occurs more than once, and all of them are reported.
# Returning only the first is what invented the phantom offsets.
def test_locate_run_reports_every_position_a_run_occurs_at():

    # A 16-frame period, so a 4-frame run starting at 20 also occurs at 4 and 36.
    identities = [frame_identity(image) for image in _periodic_video(64, 16)]
    run = identities[20:24]

    assert locate_run(identities, run) == [4, 20, 36, 52]


# test_locating_a_run_narrows_what_locating_one_frame_cannot()
# **The reason the comparison is by run.** On video that repeats inside a window,
# the frame at 300 also occurs at 252, 268, 284 and 316, so a single-frame
# comparison cannot say which of them was decoded. A run that reaches past the
# window occurs at exactly one position and does say.
def test_locating_a_run_narrows_what_locating_one_frame_cannot():

    frames = _partly_periodic_video(400, range(250, 320), period=16)
    identities = [frame_identity(image) for image in frames]

    # One frame is hopelessly ambiguous.
    assert len(locate_run(identities, identities[300:301])) > 1

    # The same start, given a run that leaves the periodic window, is not.
    assert locate_run(identities, identities[300:340]) == [300]


# test_locate_run_of_unique_content_is_a_single_position()
# Where the content does not repeat, the answer is unambiguous.
def test_locate_run_of_unique_content_is_a_single_position():

    identities = [frame_identity(image) for image in _unique_video(64)]

    assert locate_run(identities, identities[20:24]) == [20]


# test_exact_seek_on_periodic_video_is_never_called_an_offset()
# **The regression test.** A capture that seeks perfectly, over video that
# repeats every 16 frames, must not be reported as landing somewhere else. The
# verdict may be ambiguous -- that is a property of the video -- but zero must be
# among the offsets and the verdict must not be "offset".
def test_exact_seek_on_periodic_video_is_never_called_an_offset():

    frames = _periodic_video(400, 16)
    identities = sequential_identities(FakeCapture(frames))

    measurement = measure_seek(
        lambda: FakeCapture(frames, landing_error=0), identities, target=300, run_length=40
    )

    assert measurement.verdict != "offset"
    assert 0 in measurement.offsets
    assert 300 in measurement.candidates


# test_exact_seek_is_unambiguous_once_the_run_leaves_the_repeating_stretch()
# The same exact seek, on the video the real clip resembles, is reported
# unambiguously. A single-frame comparison would only manage "exact_ambiguous"
# here, and on a stream that was in fact seeking perfectly that is the difference
# between a measurement that answers the question and one that does not.
def test_exact_seek_is_unambiguous_once_the_run_leaves_the_repeating_stretch():

    frames = _partly_periodic_video(400, range(250, 320), period=16)
    identities = sequential_identities(FakeCapture(frames))

    measurement = measure_seek(
        lambda: FakeCapture(frames, landing_error=0), identities, target=300, run_length=40
    )

    assert measurement.verdict == "exact"
    assert measurement.candidates == [300]


# test_a_short_landing_inside_a_repeating_stretch_is_still_caught()
# The dangerous case: the seek lands short *and* the content repeats, so a
# single-frame comparison sees a plausible match and reports nothing wrong. The
# run reaches past the repeating stretch, so the landing is still named.
def test_a_short_landing_inside_a_repeating_stretch_is_still_caught():

    frames = _partly_periodic_video(400, range(250, 320), period=16)
    identities = sequential_identities(FakeCapture(frames))

    # Landing 32 frames short is two whole periods, so the frame at the landing
    # point is pixel-identical to the frame at the target.
    measurement = measure_seek(
        lambda: FakeCapture(frames, landing_error=-32), identities, target=300, run_length=40
    )

    # The run it decoded stays inside the repeating stretch, so it also matches
    # one period earlier. That does not matter: what matters is that the target
    # is *not* among the matches, so the seek is reported as an offset and not
    # waved through -- and that the position it really landed on is named.
    assert measurement.verdict == "offset"
    assert 268 in measurement.candidates
    assert 300 not in measurement.candidates


# test_a_seek_that_lands_short_is_reported_as_an_offset()
# The other direction, and the failure the whole check exists for: a capture
# that reports frame 300 and decodes frame 267 is caught, and the offset is
# named. Unique content, so there is nothing to hide behind.
def test_a_seek_that_lands_short_is_reported_as_an_offset():

    frames = _unique_video(400)
    identities = sequential_identities(FakeCapture(frames))

    measurement = measure_seek(
        lambda: FakeCapture(frames, landing_error=-33), identities, target=300, run_length=40
    )

    # It claimed to be at 300 while decoding 267 -- believing the claim is the
    # mistake this catches.
    assert measurement.reported_position == 300.0
    assert measurement.verdict == "offset"
    assert measurement.offsets == [-33]


# test_measure_seek_releases_every_capture_it_opens()
# One capture is opened per target, so a leak here is a leak per target against
# the video server.
def test_measure_seek_releases_every_capture_it_opens():

    frames = _unique_video(400)
    identities = sequential_identities(FakeCapture(frames))

    # Kept so it can be inspected after the measurement returns.
    opened: list[FakeCapture] = []

    def open_capture():
        capture = FakeCapture(frames)
        opened.append(capture)
        return capture

    measure_seek(open_capture, identities, target=100, run_length=10)

    assert len(opened) == 1
    assert opened[0].released is True


# test_summarize_fails_when_any_seek_landed_elsewhere()
# One bad seek fails the set. An average would hide it, and a single offset
# target is enough to offset every observation in that range.
def test_summarize_fails_when_any_seek_landed_elsewhere():

    frames = _unique_video(400)
    identities = sequential_identities(FakeCapture(frames))

    good = measure_seek(lambda: FakeCapture(frames), identities, target=100, run_length=20)
    bad = measure_seek(
        lambda: FakeCapture(frames, landing_error=-9), identities, target=200, run_length=20
    )

    assert summarize([good])["passed"] is True
    assert summarize([good, bad])["passed"] is False


# test_summarize_will_not_pass_on_ambiguous_results_alone()
# A set in which nothing was unambiguous proves nothing, so it must not pass.
# Without this, a video that repeats everywhere would report a green result
# while saying nothing at all about the decoder.
def test_summarize_will_not_pass_on_ambiguous_results_alone():

    frames = _periodic_video(400, 16)
    identities = sequential_identities(FakeCapture(frames))

    ambiguous = [
        measure_seek(lambda: FakeCapture(frames), identities, target=target, run_length=40)
        for target in (100, 200, 300)
    ]

    assert {m.verdict for m in ambiguous} == {"exact_ambiguous"}
    assert summarize(ambiguous)["passed"] is False


# test_summarize_of_nothing_does_not_pass()
# An empty run is not a green one -- a skipped measurement must not look green.
def test_summarize_of_nothing_does_not_pass():

    assert summarize([])["passed"] is False


# test_a_run_that_is_not_in_the_ground_truth_is_unknown_not_exact()
# Ground truth stopped too early, or the stream is not reproducible. Either way
# it is an unanswered question and must not be reported as a pass.
@pytest.mark.parametrize("target", [10, 50])
def test_a_run_that_is_not_in_the_ground_truth_is_unknown_not_exact(target):

    frames = _unique_video(400)

    # Ground truth over a *different* video, so nothing can match.
    identities = sequential_identities(FakeCapture(_unique_video(400)[::-1]))
    identities = [identity + "x" for identity in identities]

    measurement = measure_seek(
        lambda: FakeCapture(frames), identities, target=target, run_length=20
    )

    assert measurement.verdict == "unknown"
    assert summarize([measurement])["passed"] is False
