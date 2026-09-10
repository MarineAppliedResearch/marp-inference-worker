# test_track_accumulator.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Tests for track accumulation in the MARP Inference Worker.
# The accumulator decides when a track has ended, and the most important of
# those decisions is the range boundary. R10a settles it: a track ends at the
# end of its range and is never stitched to the next one. These tests are what
# hold that settled answer in place, because "stitch them, it looks nicer" is a
# plausible-sounding change somebody will eventually try to make.

from marp_inference_worker.tracking.track_accumulator import TrackAccumulator


# _observe_span()
# Records one track across a span of frames.
# Inputs: the accumulator, the track id, the first and last frame, and the
# class name.
# Output: none.
# Use this to build a track long enough to pass the minimum-span rule.
def _observe_span(
    accumulator: TrackAccumulator,
    track_id: int,
    first_frame: int,
    last_frame: int,
    class_name: str = "Rockfish",
) -> None:

    for frame_index in range(first_frame, last_frame + 1):
        accumulator.observe(
            track_id=track_id,
            class_name=class_name,
            frame_index=frame_index,
            frame_time_s=frame_index / 30.0,
            bbox_normalized=(0.5, 0.5, 0.05, 0.05),
            confidence=0.9,
        )


# test_track_still_open_at_range_end_is_closed_there()
# Verifies that finish_range() closes an open track.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R10a's first clause. The animal is still in frame when the range ends;
# the track ends anyway, and it is marked so the truncation is attributable
# rather than looking like the animal swam off.
def test_track_still_open_at_range_end_is_closed_there() -> None:

    accumulator = TrackAccumulator(track_buffer=240)

    # A track that is still being seen right up to the last frame of the range.
    _observe_span(accumulator, track_id=1, first_frame=1000, last_frame=1100)

    # Nothing has aged out -- the track was seen on the final frame.
    assert list(accumulator.take_aged_out(1100)) == []
    assert accumulator.active_track_count == 1

    # The range ends. The track ends with it.
    ended = list(accumulator.finish_range())
    assert len(ended) == 1
    assert ended[0].track_id == 1
    assert ended[0].reason == "range_end"

    # And nothing is left open to be carried anywhere.
    assert accumulator.active_track_count == 0


# test_range_end_leaves_nothing_to_hand_forward()
# Verifies that finish_range() empties the accumulator completely.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R10a's "not carried forward" clause. Whatever the worker holding the
# next range starts with, it is not this. Several open tracks all end here.
def test_range_end_leaves_nothing_to_hand_forward() -> None:

    accumulator = TrackAccumulator(track_buffer=240)

    # Three animals in frame as the range ends.
    _observe_span(accumulator, track_id=1, first_frame=0, last_frame=100)
    _observe_span(accumulator, track_id=2, first_frame=20, last_frame=100)
    _observe_span(accumulator, track_id=3, first_frame=50, last_frame=100)
    assert accumulator.active_track_count == 3

    ended = list(accumulator.finish_range())

    # All three closed, all three marked as ended by the seam.
    assert {track.track_id for track in ended} == {1, 2, 3}
    assert {track.reason for track in ended} == {"range_end"}

    # A second call finds nothing: there is no residue to hand on.
    assert list(accumulator.finish_range()) == []
    assert accumulator.active_track_count == 0


# test_two_ranges_produce_two_independent_observations()
# Verifies that one animal crossing a seam becomes two tracks.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R10a's accepted outcome, stated as a test rather than a comment. Two
# accumulators stand in for two workers holding adjacent ranges. The same
# animal, continuously present, yields one track in each -- and the two carry
# the same ByteTrack id purely by coincidence, which is exactly why ids must
# never be compared across ranges.
def test_two_ranges_produce_two_independent_observations() -> None:

    # The worker holding frames 0..100.
    first_range = TrackAccumulator(track_buffer=240)
    _observe_span(first_range, track_id=1, first_frame=0, last_frame=99)
    first_ended = list(first_range.finish_range())

    # The worker holding frames 100..200. A fresh accumulator, because a fresh
    # tracker: nothing is shared between the two.
    second_range = TrackAccumulator(track_buffer=240)
    _observe_span(second_range, track_id=1, first_frame=100, last_frame=199)
    second_ended = list(second_range.finish_range())

    # Two observations for one animal. This is the accepted outcome, not a bug.
    assert len(first_ended) == 1
    assert len(second_ended) == 1

    # Their frame spans do not overlap: ranges are half-open and adjacent.
    assert first_ended[0].track["frames"][-1]["frame"] == 99
    assert second_ended[0].track["frames"][0]["frame"] == 100

    # Both happen to be track id 1, from two independent trackers. Any code that
    # compared these ids across ranges would wrongly conclude they are one track.
    assert first_ended[0].track_id == second_ended[0].track_id == 1


# test_track_ages_out_after_the_track_buffer()
# Verifies that a track the tracker has lost is closed.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves the ordinary end of a track, mid-range: the animal left, the tracker
# stopped reporting it, and after track_buffer frames it is finalized. This is
# the same rule the live script used.
def test_track_ages_out_after_the_track_buffer() -> None:

    accumulator = TrackAccumulator(track_buffer=30)

    # Seen from frame 0 to 50, then never again.
    _observe_span(accumulator, track_id=1, first_frame=0, last_frame=50)

    # Not yet: only 30 frames have passed since it was last seen.
    assert list(accumulator.take_aged_out(80)) == []
    assert accumulator.active_track_count == 1

    # One more frame past the buffer and it ends.
    ended = list(accumulator.take_aged_out(81))
    assert len(ended) == 1
    assert ended[0].reason == "aged_out"
    assert accumulator.active_track_count == 0


# test_short_track_is_discarded_not_reported()
# Verifies the minimum-span rule.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves the noise filter the live script had: a track spanning fewer than 30
# frames is dropped rather than reduced and reported. Without it, a single
# spurious detection becomes an observation a biologist has to reject.
def test_short_track_is_discarded_not_reported() -> None:

    accumulator = TrackAccumulator(track_buffer=240)

    # A ten-frame track: real enough for the tracker, too short to report.
    _observe_span(accumulator, track_id=1, first_frame=0, last_frame=10)

    ended = list(accumulator.finish_range())

    # Nothing reported, and the drop is counted rather than silent.
    assert ended == []
    assert accumulator.discarded_track_count == 1


# test_track_exactly_at_the_minimum_span_is_kept()
# Verifies the boundary of the minimum-span rule.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves the comparison is "shorter than", not "shorter than or equal to". An
# off-by-one here silently drops a category of real observations.
def test_track_exactly_at_the_minimum_span_is_kept() -> None:

    accumulator = TrackAccumulator(track_buffer=240)

    # Span is last minus first, so frames 0..30 span exactly 30.
    _observe_span(accumulator, track_id=1, first_frame=0, last_frame=30)

    ended = list(accumulator.finish_range())
    assert len(ended) == 1
    assert ended[0].frame_span == 30
    assert accumulator.discarded_track_count == 0


# test_accumulated_track_is_in_the_shape_the_reduction_reads()
# Verifies the mapping handed to the keyframe reduction.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves the join between the two stages. The reduction was ported verbatim and
# reads "class_name" and "frames", with each frame carrying "frame", "time" and
# "bbox". If the accumulator's shape drifts, the reduction fails at runtime on a
# real job rather than here.
def test_accumulated_track_is_in_the_shape_the_reduction_reads() -> None:

    from marp_inference_worker.reduction import keyframes

    accumulator = TrackAccumulator(track_buffer=240)
    _observe_span(accumulator, track_id=1, first_frame=0, last_frame=60, class_name="Lingcod")

    ended = next(iter(accumulator.finish_range()))

    # The keys the ported reduction indexes into.
    assert ended.track["class_name"] == "Lingcod"
    assert set(ended.track["frames"][0]) >= {"frame", "time", "bbox"}
    assert len(ended.track["frames"][0]["bbox"]) == 4

    # And it actually reduces, which is the real proof the shape is right.
    reduced = keyframes.reduce_to_keyframes_v3_dirpad(ended.track)
    assert reduced[0]["type"] == "start"
    assert reduced[0]["comname"] == "Lingcod"
