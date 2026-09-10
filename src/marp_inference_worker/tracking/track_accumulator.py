# track_accumulator.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Track accumulation for the MARP Inference Worker.
# ByteTrack reports which boxes on this frame belong to which track. This file
# holds the other half: gathering a track's frames until it ends, and deciding
# when it has ended. It is the stateful middle of the detect -> track -> reduce
# pipeline (R10).
#
# The seam rule lives here (R10a). A track ends at the end of its frame range,
# full stop. finish_range() closes every track still open, the worker holding the
# next range starts fresh, and track ids are never compared across ranges. One
# animal crossing a seam becoming two observations is the accepted outcome, not a
# defect to be stitched over.
#
# Keyframe reduction and observation shaping are downstream and are not here.

# Any types the track mappings, which are handed to the reduction unchanged.
from typing import Any, Iterator


# Minimum span, in frames, for a track to be worth reporting.
# From the live script: a track spanning fewer than this many frames is dropped
# as noise rather than reduced and posted. Measured as last frame minus first,
# not as the number of frames seen, because a track with gaps still spans them.
_DEFAULT_MIN_TRACK_SPAN_FRAMES = 30


# EndedTrack
# One finished track, in the shape the keyframe reduction expects.
# The reduction reads `frames` and `class_name` off a mapping, so this carries
# that mapping verbatim rather than a translated copy -- the ported reduction
# code is the reference and must not have to be adapted to a new shape.
class EndedTrack:

    # __init__()
    # Builds a finished track from its id, its raw mapping and why it ended.
    # Inputs: track id, the accumulated mapping, and the reason it closed.
    # Output: initialized EndedTrack.
    def __init__(self, track_id: int, track: dict[str, Any], reason: str) -> None:

        # ByteTrack's id for this track, valid only within this frame range.
        self.track_id = track_id

        # The mapping the reduction consumes: {"class_name", "frames", ...}.
        self.track = track

        # "aged_out" when the tracker lost it, "range_end" when the seam closed
        # it. Recorded because a range_end track is expected to be truncated and
        # a reviewer should be able to tell the two apart.
        self.reason = reason

    # frame_span
    # How many frames this track spanned, first to last.
    @property
    def frame_span(self) -> int:

        # A single-frame track spans 0, which the minimum-span rule then drops.
        frames = self.track["frames"]
        return frames[-1]["frame"] - frames[0]["frame"]


# TrackAccumulator
# Gathers per-frame observations into tracks and reports tracks as they end.
# One instance per job, because its state is meaningful only inside one frame
# range.
class TrackAccumulator:

    # __init__()
    # Builds an empty accumulator.
    # Inputs: how many frames of absence ends a track, and the minimum span a
    # track must have to be reported.
    # Output: initialized TrackAccumulator.
    # Use one per job. `track_buffer` should be the same value the tracker was
    # built with, so the accumulator closes a track at the same moment ByteTrack
    # forgets it.
    def __init__(
        self,
        track_buffer: int,
        min_track_span_frames: int = _DEFAULT_MIN_TRACK_SPAN_FRAMES,
    ) -> None:

        # How many frames a track may go unseen before it is considered ended.
        self._track_buffer = track_buffer

        # Tracks shorter than this span are discarded rather than reported.
        self._min_track_span_frames = min_track_span_frames

        # Tracks still open, keyed by ByteTrack's id for them.
        self._active: dict[int, dict[str, Any]] = {}

        # Counted for the job summary: how many tracks were dropped as too short.
        self.discarded_track_count = 0

    # active_track_count
    # How many tracks are currently open.
    @property
    def active_track_count(self) -> int:

        # Reported in progress metrics so a stalled job is visible.
        return len(self._active)

    # observe()
    # Records one frame of one track.
    # Inputs: track id, class name, absolute frame index, frame time in seconds,
    # the normalized centre-form bbox, and the detection confidence -- None when
    # the track was predicted onto this frame with no detection behind it.
    # Output: none.
    # Use this once per tracked box per frame. The bbox must already be
    # normalized to (xCenter, yCenter, width, height) in 0..1, because that is
    # what the keyframe reduction's thresholds are calibrated against.
    def observe(
        self,
        track_id: int,
        class_name: str,
        frame_index: int,
        frame_time_s: float,
        bbox_normalized: tuple[float, float, float, float],
        confidence: float | None,
    ) -> None:

        # First sighting of this track opens it.
        if track_id not in self._active:
            self._active[track_id] = {
                "class_name": class_name,
                "frames": [],
                "last_seen": frame_index,
            }

        # Append this frame in the shape the reduction reads.
        self._active[track_id]["frames"].append(
            {
                "frame": frame_index,
                "time": frame_time_s,
                "bbox": bbox_normalized,
                "confidence": confidence,
            }
        )

        # Remember when it was last seen, so ageing out can be decided.
        self._active[track_id]["last_seen"] = frame_index

    # take_aged_out()
    # Closes and yields tracks the tracker has not seen for long enough.
    # Inputs: the current absolute frame index.
    # Output: iterator of EndedTrack, already filtered by minimum span.
    # Use this once per frame, after observing that frame's tracks.
    def take_aged_out(self, frame_index: int) -> Iterator[EndedTrack]:

        # Iterate over a copy: closing a track mutates the dictionary.
        for track_id, track in list(self._active.items()):

            # A track ends when it has been unseen for longer than the buffer,
            # which is the same rule the live script used.
            if frame_index - track["last_seen"] > self._track_buffer:
                ended = self._close(track_id, reason="aged_out")
                if ended is not None:
                    yield ended

    # finish_range()
    # Closes and yields every track still open, because the range has ended.
    # Inputs: none.
    # Output: iterator of EndedTrack, already filtered by minimum span.
    # Use this exactly once, when the last frame of the range has been processed.
    # This is the seam: a track does not survive past its range, is not handed
    # forward, and is not stitched to anything the next range finds (R10a).
    def finish_range(self) -> Iterator[EndedTrack]:

        # Close every remaining track, marked so its truncation is attributable.
        for track_id in list(self._active):
            ended = self._close(track_id, reason="range_end")
            if ended is not None:
                yield ended

    # _close()
    # Removes a track and returns it if it is long enough to report.
    # Inputs: track id and the reason it is closing.
    # Output: EndedTrack, or None when the track was too short.
    # Use this from the two public closers so the minimum-span rule is applied
    # in one place.
    def _close(self, track_id: int, reason: str) -> EndedTrack | None:

        # Take it out of the active set regardless of whether it is reported.
        track = self._active.pop(track_id)

        # A track with no frames cannot happen through observe(), but a guard
        # here is cheaper than an IndexError inside the reduction.
        if not track["frames"]:
            self.discarded_track_count += 1
            return None

        # Apply the minimum-span rule before the caller sees it.
        ended = EndedTrack(track_id=track_id, track=track, reason=reason)
        if ended.frame_span < self._min_track_span_frames:
            self.discarded_track_count += 1
            return None

        return ended
