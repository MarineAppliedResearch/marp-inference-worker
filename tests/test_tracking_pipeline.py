# test_tracking_pipeline.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Tests for the detect -> track -> reduce pipeline in the MARP Inference Worker.
#
# These drive the REAL vendored ByteTrack over synthetic detections. No GPU and
# no model are needed, because the detector is substituted -- but the tracker,
# the accumulator, the reduction and the observation shaping are all the real
# ones, which is the point: the join between the stages is where this pipeline
# would break, and a test that mocked the tracker could not see it.
#
# ByteTrack is a vendored source tree rather than a wheel, and it needs
# cython-bbox, which needs a C compiler on Windows. A machine without it should
# fail here rather than skip: a skipped suite looks green, and "the tracker is
# missing" is exactly the kind of deployment fault that must not pass quietly.

# Path types the temporary results file.
from pathlib import Path

from marp_inference_worker.reduction import keyframes
from marp_inference_worker.tracking import byte_tracker_adapter
from marp_inference_worker.tracking.observations import (
    build_observation,
    frame_to_media_position,
    frame_to_timecode,
    pick_observation_time,
)
from marp_inference_worker.tracking.track_accumulator import TrackAccumulator


# Frame geometry the synthetic detections are expressed against.
_FRAME_WIDTH = 1920
_FRAME_HEIGHT = 1080
_FRAME_RATE = 30.0


# test_vendored_bytetrack_is_importable_and_usable()
# Verifies the tracker this pipeline requires actually works.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves A2's "ByteTrack is required, not optional" answer is satisfied on this
# machine. It is deliberately not skipped when ByteTrack is absent: a prerequisite
# missing should fail, because a skipped suite looks green.
#
# This test is also what caught the vendored copy being unusable: ByteTrack uses
# np.float, np.int and np.bool, which numpy removed in 1.24, so the import
# succeeded and the first tracked detection raised AttributeError. The adapter
# restores those aliases; without it, this test fails on the update() call.
def test_vendored_bytetrack_is_importable_and_usable() -> None:

    tracker, args = byte_tracker_adapter.create_tracker({})

    # Built with the live pipeline's settings.
    assert args.as_dict["track_thresh"] == 0.30
    assert args.as_dict["match_thresh"] == 0.70
    assert args.as_dict["track_buffer"] == 240

    # And it actually tracks something, which is the assertion that matters --
    # constructing the tracker never touched the removed numpy aliases.
    detections = byte_tracker_adapter.detections_to_array(
        [{"bbox_xyxy": [100.0, 100.0, 200.0, 250.0], "confidence": 0.9, "class_id": 0}]
    )
    tracked = tracker.update(
        detections, [_FRAME_HEIGHT, _FRAME_WIDTH], (_FRAME_HEIGHT, _FRAME_WIDTH)
    )

    assert len(tracked) == 1
    assert tracked[0].track_id >= 1


# test_empty_frame_produces_a_correctly_shaped_array()
# Verifies the empty-frame case.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves a frame with nothing on it does not crash the tracker. An empty list
# would give ByteTrack's own indexing the wrong shape and raise, rather than
# reporting no tracks.
def test_empty_frame_produces_a_correctly_shaped_array() -> None:

    array = byte_tracker_adapter.detections_to_array([])

    assert array.shape == (0, 5)

    tracker, _ = byte_tracker_adapter.create_tracker({})
    assert list(tracker.update(array, [_FRAME_HEIGHT, _FRAME_WIDTH],
                               (_FRAME_HEIGHT, _FRAME_WIDTH))) == []


# test_tracker_settings_come_from_the_job_params()
# Verifies that a job can tune the tracker.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves the settings are the job's, not the worker's. MARP tunes for recall,
# and which thresholds achieve that is a per-survey decision.
def test_tracker_settings_come_from_the_job_params() -> None:

    _, args = byte_tracker_adapter.create_tracker(
        {"track_thresh": 0.15, "track_buffer": 60, "an_unrelated_param": "ignored"}
    )

    assert args.as_dict["track_thresh"] == 0.15
    assert args.as_dict["track_buffer"] == 60

    # Unrecognized params are ignored rather than set as attributes, so a typo
    # in a job spec cannot become a tracker setting nobody meant.
    assert "an_unrelated_param" not in args.as_dict


# _moving_animal_detections(frame_index)
# Builds one frame's detections for an animal crossing the frame.
# Inputs: the frame index.
# Output: a detection list in the shape the detector produces.
# Use this to drive the tracker over a plausible track.
def _moving_animal_detections(frame_index: int) -> list[dict]:

    # Moves right and down at a steady rate, as a fish crossing an ROV
    # transect does. Pixel coordinates, as the detector emits.
    x1 = 200.0 + frame_index * 4.0
    y1 = 300.0 + frame_index * 6.0
    return [
        {
            "bbox_xyxy": [x1, y1, x1 + 120.0, y1 + 90.0],
            "confidence": 0.85,
            "class_id": 0,
        }
    ]


# test_full_pipeline_produces_one_observation_for_one_animal(tmp_path)
# Verifies detect -> track -> reduce end to end.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R10: the pipeline is the three stages, joined. The tracker assigns an
# id, the accumulator gathers the track, the reduction turns it into keyframes
# labelled start/middle/end, and the result is an observation in the terms MARP
# already stores.
def test_full_pipeline_produces_one_observation_for_one_animal(tmp_path: Path) -> None:

    tracker, args = byte_tracker_adapter.create_tracker({"track_buffer": 30})
    accumulator = TrackAccumulator(track_buffer=args.as_dict["track_buffer"])

    # Run the real tracker over 90 frames of one animal.
    for frame_index in range(90):
        detections = _moving_animal_detections(frame_index)

        tracked = tracker.update(
            byte_tracker_adapter.detections_to_array(detections),
            [_FRAME_HEIGHT, _FRAME_WIDTH],
            (_FRAME_HEIGHT, _FRAME_WIDTH),
        )

        for track in tracked:
            x1, y1, x2, y2 = (float(value) for value in track.tlbr)

            # The class is recovered by matching the track's box back to the
            # detection that produced it, as the live script did.
            best_iou = 0.0
            for detection in detections:
                best_iou = max(
                    best_iou, byte_tracker_adapter.calculate_iou((x1, y1, x2, y2),
                                                                 detection["bbox_xyxy"])
                )
            assert best_iou > 0.4, "tracker box did not match its own detection"

            accumulator.observe(
                track_id=int(track.track_id),
                class_name="Rockfish",
                frame_index=frame_index,
                frame_time_s=frame_index / _FRAME_RATE,
                bbox_normalized=(
                    (x1 + x2) / 2 / _FRAME_WIDTH,
                    (y1 + y2) / 2 / _FRAME_HEIGHT,
                    (x2 - x1) / _FRAME_WIDTH,
                    (y2 - y1) / _FRAME_HEIGHT,
                ),
                confidence=0.85,
            )

    # The range ends, so the track ends.
    ended = list(accumulator.finish_range())
    assert len(ended) == 1, f"expected one track, got {len(ended)}"

    # Reduce it with the rule the job would have named.
    reduce = keyframes.get_reduction("v3_dirpad", "1")
    reduced = reduce(ended[0].track)

    # Reduced, labelled, and far fewer than 90.
    assert 2 <= len(reduced) < 90
    assert reduced[0]["type"] == "start"
    assert reduced[-1]["type"] == "end"

    # Shape it as MARP stores it.
    observation = build_observation(
        frames=ended[0].track["frames"],
        keyframes=reduced,
        video_source_name="20240727_185645 Fwd.mp4",
        jellyfin_item_id="item-1",
        frame_rate=_FRAME_RATE,
        data_type="Fish",
        reduction_name="v3_dirpad",
        reduction_version="1",
        track_id=ended[0].track_id,
        end_reason=ended[0].reason,
    )

    # Every field the live script posted, present and plausible.
    assert observation["comname"] == "Rockfish"
    assert observation["count"] == 1
    assert observation["video_source"] == "20240727_185645 Fwd.mp4"
    assert observation["keyframes"] == reduced

    # The reduction that made it is recorded with it (R10b).
    assert observation["reduction"] == {"name": "v3_dirpad", "version": "1"}


# test_observation_carries_the_live_scripts_field_set()
# Verifies the observation payload against the working precedent.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves the decision that tracking already reaches MARP and the precedent must
# be preserved. These are the exact keys object_tracking_live.py posts to
# /api/observation, minus the three the worker cannot know -- and the Jellyfin
# item id is carried so the coordinator can supply those three itself.
def test_observation_carries_the_live_scripts_field_set() -> None:

    frames = [
        {"frame": index, "time": index / _FRAME_RATE, "bbox": (0.5, 0.85, 0.06, 0.05),
         "confidence": 0.9}
        for index in range(60)
    ]
    reduced = keyframes.reduce_to_keyframes_v3_dirpad(
        {"class_name": "Lingcod", "frames": frames}
    )

    observation = build_observation(
        frames=frames,
        keyframes=reduced,
        video_source_name="20240727_185645 Fwd.mp4",
        jellyfin_item_id="jf-item-42",
        frame_rate=_FRAME_RATE,
        data_type="Fish",
        reduction_name="v3_dirpad",
        reduction_version="1",
        track_id=7,
        end_reason="aged_out",
    )

    # The fields carried over from the live payload.
    for field in (
        "comname",
        "count",
        "tc",
        "frame",
        "video_source",
        "mediaPosition",
        "actualPosition",
        "keyframes",
    ):
        assert field in observation, field

    # session_id, taxserial and videoLocation are absent on purpose: the worker
    # does not know the session, mapping a name to a taxserial is a database
    # decision, and videoLocation on a distributed worker would be a Jellyfin
    # stream url. All three are values the coordinator already holds.
    for field in ("session_id", "taxserial", "videoLocation"):
        assert field not in observation, field

    # And the item id it needs to supply them is carried back.
    assert observation["jellyfin_item_id"] == "jf-item-42"

    # `frame` is the sub-second index within its own second, as MARP stores it.
    assert int(observation["frame"]) < int(_FRAME_RATE)


# test_timecode_helpers_match_the_live_script()
# Verifies the two timecode formats.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves the formats MARP parses are unchanged. These strings are read by
# existing queries and tools outside this workspace, so the format is a
# contract, not a presentation choice.
def test_timecode_helpers_match_the_live_script() -> None:

    # One hour, two minutes, three seconds at 30fps.
    frame = int((3600 + 120 + 3) * 30)
    assert frame_to_timecode(frame, 30.0) == "01:02:03"
    assert frame_to_media_position(frame, 30.0) == "01:02:03.000"

    # A frame part way through a second keeps its milliseconds.
    assert frame_to_media_position(frame + 15, 30.0) == "01:02:03.500"

    # Zero is the start, not an error.
    assert frame_to_timecode(0, 30.0) == "00:00:00"


# test_fish_observation_time_is_the_first_bottom_crossing()
# Verifies the Fish survey rule.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves a scientific convention, ported rather than reinvented. A fish is
# counted the first time its centre crosses into the bottom fifth of frame,
# which is where an ROV transect's counting line effectively is.
def test_fish_observation_time_is_the_first_bottom_crossing() -> None:

    # Centre descends steadily from the top of frame to the bottom.
    frames = [
        {"frame": index, "time": index / _FRAME_RATE,
         "bbox": (0.5, index * 0.02, 0.06, 0.05), "confidence": 0.9}
        for index in range(50)
    ]

    chosen = pick_observation_time(frames, "Fish")

    # The first frame whose centre is past 0.8, and not a later one.
    assert chosen is not None
    assert chosen["bbox"][1] > 0.8
    assert frames[frames.index(chosen) - 1]["bbox"][1] <= 0.8


# test_observation_time_falls_back_to_the_middle_frame()
# Verifies the fallback.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves a track that never satisfies the survey rule still gets a time rather
# than being dropped -- the live script's behaviour, and it matters because an
# animal that never crosses the line was still seen.
def test_observation_time_falls_back_to_the_middle_frame() -> None:

    # Stays in the top half throughout, so no crossing ever happens.
    frames = [
        {"frame": index, "time": index / _FRAME_RATE,
         "bbox": (0.5, 0.2, 0.06, 0.05), "confidence": 0.9}
        for index in range(41)
    ]

    chosen = pick_observation_time(frames, "Fish")
    assert chosen is frames[20]


# test_invertebrate_rule_uses_the_bottom_trapezoid()
# Verifies the Invert survey rule.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves the second convention is distinct from the first and was ported too:
# inverts are counted in a trapezoid that narrows toward the bottom of frame,
# not on a horizontal line.
def test_invertebrate_rule_uses_the_bottom_trapezoid() -> None:

    # Inside the trapezoid: bottom half, and close to centre horizontally.
    inside = [
        {"frame": 10, "time": 10 / _FRAME_RATE, "bbox": (0.5, 0.7, 0.06, 0.05),
         "confidence": 0.9}
    ]
    assert pick_observation_time(inside, "Invert") is inside[0]

    # Bottom half but far off-centre, so outside the narrowed trapezoid: the
    # rule does not match and the middle-frame fallback is used instead.
    outside = [
        {"frame": index, "time": index / _FRAME_RATE, "bbox": (0.95, 0.9, 0.06, 0.05),
         "confidence": 0.9}
        for index in range(5)
    ]
    assert pick_observation_time(outside, "Invert") is outside[2]


# test_results_are_written_as_one_json_object_per_line(tmp_path)
# Verifies the results file format.
# Inputs: pytest temporary directory.
# Output: pytest pass/fail result.
#
# Proves R11's file format is streamable in both directions. A single JSON array
# would have to be held whole in memory to write and to read, which for a
# ten-hour video's observations defeats the point of the file.
def test_results_are_written_as_one_json_object_per_line(tmp_path: Path) -> None:

    import json

    frames = [
        {"frame": index, "time": index / _FRAME_RATE, "bbox": (0.5, 0.85, 0.06, 0.05),
         "confidence": 0.9}
        for index in range(60)
    ]
    reduced = keyframes.reduce_to_keyframes_v3_dirpad({"class_name": "Lingcod", "frames": frames})

    results_path = tmp_path / "observations.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for track_id in (1, 2, 3):
            handle.write(
                json.dumps(
                    build_observation(
                        frames=frames,
                        keyframes=reduced,
                        video_source_name="v.mp4",
                        jellyfin_item_id="item-1",
                        frame_rate=_FRAME_RATE,
                        data_type="Fish",
                        reduction_name="v3_dirpad",
                        reduction_version="1",
                        track_id=track_id,
                        end_reason="aged_out",
                    )
                )
                + "\n"
            )

    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    # Every line stands alone as valid JSON.
    for line in lines:
        assert json.loads(line)["comname"] == "Lingcod"


# test_infer_stream_is_a_generator()
# Verifies inference is streaming, not accumulating.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R8 structurally. Ultralytics' predict() with stream=False accumulates
# every Results object for the whole input, which for a ten-hour video is the
# whole video in RAM. infer_stream must therefore be a generator that yields per
# frame, and its caller must not collect the frames first.
def test_infer_stream_is_a_generator() -> None:

    import inspect

    from marp_inference_worker.engines.ultralytics_engine import UltralyticsEngine

    # A generator function, so nothing is produced until it is iterated.
    assert inspect.isgeneratorfunction(UltralyticsEngine.infer_stream)

    # It feeds single frames rather than handing Ultralytics the whole source.
    source = inspect.getsource(UltralyticsEngine.infer_stream)
    assert "source=frame.image" in source

    # The frame reader is a generator too, so the two compose without either
    # one materializing the video.
    from marp_inference_worker.media import frame_range_reader

    assert inspect.isgeneratorfunction(frame_range_reader.iter_frame_range)

    # And the tracking engine consumes the reader directly rather than listing
    # it, which is the mistake that would quietly undo all of the above.
    from marp_inference_worker.engines import tracking_engine

    engine_source = inspect.getsource(tracking_engine.TrackingEngine.run)
    assert "list(iter_frame_range" not in engine_source
    assert "frame_stream = iter_frame_range(" in engine_source


# test_an_observation_from_a_job_with_no_item_id_records_null()
# Verifies what the observation carries when the provenance field is absent.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves A8's "optional, echoed exactly as received" clause at the row level. A
# job handed a bare url has no item id, and the worker must not invent one: the
# key stays present so the output shape does not change per job, and its value
# is null. A placeholder here would be worse than nothing, because the
# coordinator resolves session_id and videoLocation from this field and would
# resolve a fabricated one to the wrong video.
def test_an_observation_from_a_job_with_no_item_id_records_null() -> None:

    frames = [
        {"frame": index, "time": index / _FRAME_RATE, "bbox": (0.5, 0.85, 0.06, 0.05),
         "confidence": 0.9}
        for index in range(60)
    ]
    reduced = keyframes.reduce_to_keyframes_v3_dirpad(
        {"class_name": "Lingcod", "frames": frames}
    )

    observation = build_observation(
        frames=frames,
        keyframes=reduced,
        video_source_name="20240727_185645 Fwd.mp4",
        # What the spec carries for a job submitted with a bare url.
        jellyfin_item_id=None,
        frame_rate=_FRAME_RATE,
        data_type="Fish",
        reduction_name="v3_dirpad",
        reduction_version="1",
        track_id=7,
        end_reason="aged_out",
    )

    # Present and null, not absent and not a placeholder string.
    assert "jellyfin_item_id" in observation
    assert observation["jellyfin_item_id"] is None

    # And the field the coordinator actually needs to identify the video is
    # unaffected -- source_name is required, so a row is never anonymous.
    assert observation["video_source"] == "20240727_185645 Fwd.mp4"

    # It still serializes, which is the point of a null over a missing key.
    import json

    assert json.loads(json.dumps(observation))["jellyfin_item_id"] is None


# _confidence_for(frame_index)
# The detection score this test hands the pipeline for a given frame.
# Inputs: the frame index.
# Output: a score that names its own frame in its last three digits.
# Use this so a confidence that arrives attributed to the wrong frame is
# unmistakable: 0.470 could only have come from frame 70. Above ByteTrack's
# 0.30 track_thresh, or no track would ever be initiated.
def _confidence_for(frame_index: int) -> float:

    return 0.40 + frame_index / 1000.0


# _descending_animal_detections(frame_index)
# Builds one frame's detections for an animal descending through frame.
# Inputs: the frame index.
# Output: a detection list in the shape the detector produces.
# Use this when the Fish survey rule has to fire: the centre crosses y > 0.8
# part way through, so pick_observation_time lands on a mid-track frame rather
# than on the first, the last or the middle one.
def _descending_animal_detections(frame_index: int) -> list[dict]:

    # Pixel coordinates, as the detector emits. Descends 11px a frame, so the
    # normalized centre passes 0.8 around frame 70 of 78.
    x1 = 400.0 + frame_index * 2.0
    y1 = 60.0 + frame_index * 11.0
    return [
        {
            "bbox_xyxy": [x1, y1, x1 + 120.0, y1 + 90.0],
            "confidence": _confidence_for(frame_index),
            "class_id": 0,
        }
    ]


# _NamedClasses
# The one thing the tracking engine asks of a loaded model when attaching a
# class to a track: a `names` mapping. Substituted so these tests need no GPU
# and no weights while the tracker, the matcher and the accumulator stay real.
class _NamedClasses:

    # The class id the synthetic detections carry.
    names = {0: "Rockfish"}


# _run_pipeline_over(frame_count, detections_visible_from)
# Drives the real tracker and the real engine matcher over synthetic detections.
# Inputs: how many frames to run, and the frame from which the engine is handed
# no detections to match its tracks against.
# Output: the list of EndedTrack the accumulator closed at the range end.
# Use this so both confidence tests exercise TrackingEngine._observe_tracks --
# the code that actually decides what score a frame is recorded with -- rather
# than a copy of its matching loop, which could agree with itself while the
# engine was wrong.
def _run_pipeline_over(frame_count: int, detections_visible_from: int = 10**9) -> list:

    from marp_inference_worker.engines.tracking_engine import TrackingEngine

    engine = TrackingEngine()
    tracker, args = byte_tracker_adapter.create_tracker({"track_buffer": 30})
    accumulator = TrackAccumulator(track_buffer=args.as_dict["track_buffer"])

    for frame_index in range(frame_count):
        detections = _descending_animal_detections(frame_index)

        # The tracker always sees the real detections, so the track it reports
        # is a real track.
        tracked = tracker.update(
            byte_tracker_adapter.detections_to_array(detections),
            [_FRAME_HEIGHT, _FRAME_WIDTH],
            (_FRAME_HEIGHT, _FRAME_WIDTH),
        )

        # From detections_visible_from onward the engine is handed nothing to
        # match against, which is the predicted-track case: ByteTrack carries a
        # box forward with its Kalman filter and no detection sits behind it.
        # Real footage produces this occasionally; forcing it makes it testable.
        visible = [] if frame_index >= detections_visible_from else detections

        engine._observe_tracks(
            accumulator=accumulator,
            tracked=tracked,
            detections=visible,
            yolo_model=_NamedClasses(),
            frame_index=frame_index,
            frame_time_s=frame_index / _FRAME_RATE,
            frame_width=_FRAME_WIDTH,
            frame_height=_FRAME_HEIGHT,
        )

    return list(accumulator.finish_range())


# _observation_from(ended)
# Reduces one ended track and shapes it as MARP stores it.
# Inputs: one EndedTrack.
# Output: the observation mapping.
# Use this so the two confidence tests differ only in the pipeline they ran.
def _observation_from(ended) -> dict:

    reduce = keyframes.get_reduction("v3_dirpad", "1")
    reduced = reduce(ended.track)

    return build_observation(
        frames=ended.track["frames"],
        keyframes=reduced,
        video_source_name="20240727_185645 Fwd.mp4",
        jellyfin_item_id="item-1",
        frame_rate=_FRAME_RATE,
        data_type="Fish",
        reduction_name="v3_dirpad",
        reduction_version="1",
        track_id=ended.track_id,
        end_reason=ended.reason,
    )


# test_observation_confidence_is_the_score_at_the_observation_frame()
# Verifies the confidence on an observation is the score at its own frame.
# Inputs: none.
# Output: pytest pass/fail result.
#
# MARP's observations.confidence holds the detection score at the chosen
# observation frame -- not a mean over the track and not a max. The score and
# the frame must agree, so a reviewer can open exactly that frame and see what
# scored it.
#
# Asserted at the pipeline tier because the value has to survive three joins to
# get here: the engine's IoU match picks it, the accumulator stores it per
# frame, and the observation shaper has to read it off the frame the survey rule
# chose. A unit test on a mock would agree with whichever frame the mock used.
#
# The score names its own frame, and the test also asserts it is none of the
# five wrong answers -- first, last, min, max, mean -- because each of those
# would pass a bare "confidence is present and plausible" check.
def test_observation_confidence_is_the_score_at_the_observation_frame() -> None:

    ended = _run_pipeline_over(78)
    assert len(ended) == 1, f"expected one track, got {len(ended)}"

    observation = _observation_from(ended[0])
    frames = ended[0].track["frames"]

    # Present on every observation, alongside the frame it describes.
    assert "confidence" in observation
    observation_frame = observation["observation_frame"]

    # The survey rule chose a frame part way through, which is what makes the
    # wrong answers below distinguishable at all.
    assert frames[0]["frame"] < observation_frame < frames[-1]["frame"]

    # The score at that frame, and at no other frame.
    assert observation["confidence"] == _confidence_for(observation_frame)

    # The same value read back off the accumulated frame, so this is a fact
    # about the track rather than about the formula above.
    at_frame = next(f for f in frames if f["frame"] == observation_frame)
    assert observation["confidence"] == at_frame["confidence"]

    # None of the five plausible wrong answers.
    scores = [f["confidence"] for f in frames]
    assert observation["confidence"] != scores[0], "took the first frame's score"
    assert observation["confidence"] != scores[-1], "took the last frame's score"
    assert observation["confidence"] != min(scores), "took the minimum"
    assert observation["confidence"] != max(scores), "took the maximum"
    assert observation["confidence"] != sum(scores) / len(scores), "took the mean"


# test_observation_confidence_is_null_when_that_frame_had_no_detection()
# Verifies the predicted-track case records no score rather than a wrong one.
# Inputs: none.
# Output: pytest pass/fail result.
#
# A track can exist on a frame with no detection behind it -- ByteTrack predicts
# through gaps with its Kalman filter -- and the observation frame is chosen by
# the survey rule, which has no reason to land on a frame that was detected.
# That frame is already labelled "Unknown" rather than mislabelled; the score is
# recorded as null for the same reason. Borrowing a neighbouring frame's score
# would put a number in a scientific record that no frame measured.
#
# The key stays present so the output shape does not vary per row.
def test_observation_confidence_is_null_when_that_frame_had_no_detection() -> None:

    # Nothing matches the track from frame 66 on, and the Fish rule fires
    # around frame 70, so the observation frame is one of the undetected ones.
    ended = _run_pipeline_over(78, detections_visible_from=66)
    assert len(ended) == 1, f"expected one track, got {len(ended)}"

    observation = _observation_from(ended[0])
    frames = ended[0].track["frames"]

    assert observation["observation_frame"] >= 66
    assert "confidence" in observation
    assert observation["confidence"] is None

    # Null for this frame specifically, not a track with no scores at all:
    # the frames that did have a detection behind them still carry theirs.
    early = [f["confidence"] for f in frames if f["frame"] < 66]
    assert early, "the track never had a detected frame, so this proves nothing"
    assert all(score is not None for score in early)

    # The class name is unaffected -- it was learned at the first sighting.
    assert observation["comname"] == "Rockfish"

    # And it serializes as null, which is why a None beats a missing key.
    import json

    assert json.loads(json.dumps(observation))["confidence"] is None
