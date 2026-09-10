# seek_verification.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Frame-seek verification for the MARP Inference Worker.
# A job that covers frames [300, 600) opens a stream and seeks to 300. If
# CAP_PROP_POS_FRAMES lands short, every observation in every non-zero range is
# silently offset and nothing else in this repository can see it. This file
# measures whether the seek landed where it was asked to, against a sequential
# decode from frame 0 as the ground truth.
#
# The analysis belongs here; the decoding of a particular Jellyfin item does not.
# scripts/verify_seek_accuracy.py is the thin entry point that supplies a stream.
#
# The measurement itself has a trap, and it is the reason the comparison is by
# *run* rather than by single frame. Real ROV video contains stretches of
# pixel-identical frames -- 547 of the 979 frames of one CAMPA2021 clip are
# byte-identical to another frame, in a repeating period. Mapping a frame back to
# "the first index with this hash" therefore reports a duplicate as a seek error;
# the first version of this measurement did exactly that and produced offsets of
# -32, -33, -34 and -129 frames on a stream that was in fact seeking perfectly.

# hashlib gives every decoded frame an identity that is cheap to compare.
import hashlib

# NamedTuple keeps one measurement's fields named rather than positional.
from typing import Any, Callable, Iterable, NamedTuple


# How many frames are decoded after a seek to form the run that gets located.
# One frame is not enough on video with repeated frames; a run of a few dozen is
# unique except where the content is genuinely periodic over that whole window.
_DEFAULT_RUN_LENGTH = 40


# How many hex characters of the digest are kept.
# Full sha1 is 40; a truncation this long still makes an accidental collision
# between two frames of one video vanishingly unlikely, and keeps a ground-truth
# list of a hundred thousand frames small.
_IDENTITY_LENGTH = 16


# frame_identity()
# Gives one decoded frame an identity that can be compared to another frame.
# Inputs: a decoded image array as OpenCV produced it.
# Output: a short hex string.
# Use this for every frame on both sides of the comparison. It hashes the raw
# pixel buffer, so two frames are the same identity exactly when they are the
# same picture -- which is the point, and also the trap this module documents.
def frame_identity(image: Any) -> str:

    # tobytes() is the raw buffer, so nothing about stride or dtype is guessed.
    return hashlib.sha1(image.tobytes()).hexdigest()[:_IDENTITY_LENGTH]


# sequential_identities()
# Decodes a capture from wherever it is and returns one identity per frame.
# Inputs: an open capture, and an optional ceiling on how many frames to read.
# Output: list of identities, in decode order.
# Use this on a freshly opened capture and nothing else -- it is the ground
# truth, so it must never have been seeked. The ceiling exists because a
# 38,000-frame video takes minutes to decode and a measurement only needs to
# reach just past its furthest target.
def sequential_identities(capture: Any, limit: int | None = None) -> list[str]:

    # One identity per frame, in the order the decoder produced them.
    identities: list[str] = []

    # Read until the video ends or the ceiling is reached.
    while limit is None or len(identities) < limit:
        read_ok, image = capture.read()
        if not read_ok:
            break
        identities.append(frame_identity(image))

    return identities


# locate_run()
# Finds every position in the ground truth where a run of frames occurs.
# Inputs: the ground-truth identities and the run of identities to locate.
# Output: sorted list of start indices, empty when the run does not occur.
# Use this rather than looking one frame up. A single frame can occur at many
# indices on video with repeated frames, and treating the first of them as "where
# we landed" invents an offset that is not there.
def locate_run(identities: list[str], run: list[str]) -> list[int]:

    # An empty run matches everywhere, which is not a useful answer.
    if not run:
        return []

    # Every start position at which the whole run appears, in order.
    return [
        start
        for start in range(len(identities) - len(run) + 1)
        if identities[start : start + len(run)] == run
    ]


# SeekMeasurement
# What one seek to one target frame turned out to do.
# Carries the candidates as well as the verdict, because "exact, but the content
# repeats so five positions match" is a materially different result from "exact
# and unambiguous" and both are different from "landed somewhere else".
class SeekMeasurement(NamedTuple):

    # The frame index the seek asked for.
    target: int

    # What the capture said its position was after the seek, before any read.
    # Recorded because a backend can report the position it was asked for while
    # decoding a different frame, which is precisely the failure being hunted.
    reported_position: float

    # How many frames were actually decoded to form the run.
    run_length: int

    # Every ground-truth position at which the decoded run occurs.
    candidates: list[int]

    # verdict
    # One of "exact", "exact_ambiguous", "offset", or "unknown".
    @property
    def verdict(self) -> str:

        # No match at all means the run is not in the ground truth: either the
        # ground truth was too short, or the stream is not reproducible.
        if not self.candidates:
            return "unknown"

        # Exactly the target, and nothing else, is the only unambiguous pass.
        if self.candidates == [self.target]:
            return "exact"

        # The target is among several matches, so the content repeats over the
        # whole run. Consistent with an exact seek, but not proof of one.
        if self.target in self.candidates:
            return "exact_ambiguous"

        # The target is not among the matches, so the seek landed elsewhere.
        return "offset"

    # offsets
    # How far each candidate is from the target, in frames.
    @property
    def offsets(self) -> list[int]:

        # Reported so a constant offset can be told from a varying one.
        return [candidate - self.target for candidate in self.candidates]


# measure_seek()
# Seeks a freshly opened capture to one target and says where it landed.
# Inputs: a callable returning a new open capture, the ground-truth identities,
# the target frame, and how many frames to decode after the seek.
# Output: one SeekMeasurement.
# Use a fresh capture per target. Reusing one would measure a seek from wherever
# the previous target left it, which is a different question from the one a job
# asks -- a job always seeks once, on a capture it just opened.
def measure_seek(
    open_capture: Callable[[], Any],
    identities: list[str],
    target: int,
    run_length: int = _DEFAULT_RUN_LENGTH,
) -> SeekMeasurement:

    # Imported lazily, and only for the property id, so this module can be
    # imported on a machine with no OpenCV.
    import cv2

    # A capture that has only ever been seeked once, as a job's would be.
    capture = open_capture()
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        reported = float(capture.get(cv2.CAP_PROP_POS_FRAMES))

        # Decode forward from wherever the seek actually landed.
        run = sequential_identities(capture, limit=run_length)
    finally:
        # Release whatever happened, so a failed measurement does not leave a
        # stream open against the video server.
        capture.release()

    return SeekMeasurement(
        target=target,
        reported_position=reported,
        run_length=len(run),
        candidates=locate_run(identities, run),
    )


# summarize()
# Reduces a set of measurements to the one line a person needs.
# Inputs: the measurements.
# Output: mapping of verdict counts, the distinct offsets seen, and whether the
# whole set passed.
# Use this to decide, rather than reading the table. `passed` is deliberately
# strict about only one thing: no measurement may be an "offset" or "unknown".
# An "exact_ambiguous" is allowed because periodic content is a property of the
# video and not of the decoder, and at least one unambiguous "exact" is required
# so a set of entirely ambiguous results cannot pass by default.
def summarize(measurements: Iterable[SeekMeasurement]) -> dict[str, Any]:

    # Keep them, since the generator is walked more than once below.
    taken = list(measurements)

    # Count each verdict so a partial failure is visible rather than averaged.
    counts: dict[str, int] = {}
    for measurement in taken:
        counts[measurement.verdict] = counts.get(measurement.verdict, 0) + 1

    # Every offset seen anywhere, so a constant offset can be told apart from
    # one that varies with the target -- a constant one could be corrected for,
    # a varying one could not.
    offsets = sorted({offset for m in taken for offset in m.offsets})

    return {
        "measured": len(taken),
        "verdicts": counts,
        "distinct_offsets": offsets,
        "passed": bool(taken)
        and counts.get("offset", 0) == 0
        and counts.get("unknown", 0) == 0
        and counts.get("exact", 0) > 0,
    }
