# byte_tracker_adapter.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# ByteTrack adapter for the MARP Inference Worker.
# The tracking stage of MARP's pipeline is ByteTrack, and the vendored copy under
# ByteTrack/ is the same code the live script used. This file is the only place
# that imports it, so the compatibility work and the parameter mapping live in
# one place instead of being spread through the engine.
# Tracker construction and per-frame update belong here; detection and keyframe
# reduction do not.

# numpy builds the detection array ByteTrack's update() expects.
import numpy as np

# Any types the vendored tracker and its track objects.
from typing import Any

# The vendored ByteTrack checkout, which is not an installed package.
from pathlib import Path
import sys


# Default ByteTrack settings, copied from `Args` in object_tracking_live.py.
# These are the values the live MARP pipeline ran with, so they are the defaults
# a job inherits when its params do not override them. They favour recall: MARP's
# reviewers reject false positives cheaply in mosaics, whereas a false negative
# costs somebody re-reviewing a great deal of video.
_DEFAULT_TRACKER_ARGS: dict[str, Any] = {
    "track_thresh": 0.30,
    "match_thresh": 0.70,
    "track_buffer": 240,
    "mot20": False,
}


# _restore_removed_numpy_aliases()
# Puts back the numpy aliases the vendored ByteTrack still uses.
# Inputs: none.
# Output: none.
#
# ByteTrack uses `np.float`, `np.int` and `np.bool`, which numpy deprecated in
# 1.20 and removed in 1.24. Every modern numpy therefore raises AttributeError
# inside STrack.__init__ on the first tracked detection -- the import succeeds
# and the failure only appears once a frame has something on it.
#
# The alternative was editing the vendored tree. This shim was chosen instead so
# ByteTrack/ stays byte-identical to upstream and there is exactly one place to
# delete when ByteTrack is either updated or forked properly. The aliases were
# always plain synonyms for the Python builtins, so restoring them changes no
# numeric behaviour.
def _restore_removed_numpy_aliases() -> None:

    # Only define what is missing; a numpy that still has them is left alone.
    for alias_name, builtin_type in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, alias_name):
            setattr(np, alias_name, builtin_type)


# vendored_bytetrack_path()
# Returns the path to the vendored ByteTrack checkout in this repository.
# Inputs: none.
# Output: path to the directory that contains the `yolox` package.
# Use this rather than assuming ByteTrack is pip-installed; it is a checked-in
# source tree, not a wheel.
def vendored_bytetrack_path() -> Path:

    # This file is at src/marp_inference_worker/tracking/, so the repository
    # root is four parents up.
    return Path(__file__).resolve().parents[3] / "ByteTrack"


# TrackerArgs
# The argument object ByteTrack's BYTETracker expects.
# ByteTrack reads its settings off attributes of a plain object rather than a
# mapping, so this exists purely to satisfy that shape.
class TrackerArgs:

    # __init__()
    # Builds a tracker argument object from the defaults plus any overrides.
    # Inputs: mapping of override values, typically the job's params.
    # Output: initialized TrackerArgs.
    def __init__(self, overrides: dict[str, Any] | None = None) -> None:

        # Start from the live pipeline's values, then apply the job's overrides.
        settings = dict(_DEFAULT_TRACKER_ARGS)
        if overrides:
            settings.update(
                {key: value for key, value in overrides.items() if key in _DEFAULT_TRACKER_ARGS}
            )

        # Set them as attributes, which is what BYTETracker reads.
        for key, value in settings.items():
            setattr(self, key, value)

        # Kept so a caller can report what the tracker actually ran with.
        self.as_dict = settings


# create_tracker()
# Builds a BYTETracker with the job's settings.
# Inputs: mapping of tracker overrides from the job params.
# Output: the tracker and the TrackerArgs it was built with.
# Use this once per job. A tracker holds per-video state and must not be shared
# across frame ranges -- a range boundary is a seam, and a fresh tracker is what
# makes it one (R10a).
def create_tracker(overrides: dict[str, Any] | None = None) -> tuple[Any, TrackerArgs]:

    # Put the removed numpy aliases back before any ByteTrack code runs.
    _restore_removed_numpy_aliases()

    # Add the vendored checkout to the import path if it is not already there.
    bytetrack_path = str(vendored_bytetrack_path())
    if bytetrack_path not in sys.path:
        sys.path.insert(0, bytetrack_path)

    # Imported after the path and the shim are in place, not at module import,
    # so importing this module on a machine without torch still works.
    from yolox.tracker.byte_tracker import BYTETracker

    # Build the argument object and the tracker from it.
    tracker_args = TrackerArgs(overrides)
    return BYTETracker(tracker_args), tracker_args


# detections_to_array()
# Converts normalized detections into the array ByteTrack's update() wants.
# Inputs: list of detections carrying `bbox_xyxy` and `confidence`.
# Output: an (N, 5) float array of x1, y1, x2, y2, score.
# Use this per frame. An empty frame must still produce a correctly shaped empty
# array, or ByteTrack's own indexing fails rather than reporting no tracks.
def detections_to_array(detections: list[dict[str, Any]]) -> np.ndarray:

    # A frame with nothing on it is normal and gets an empty (0, 5) array.
    if not detections:
        return np.empty((0, 5), dtype=float)

    # One row per detection, in the order ByteTrack reads them.
    return np.array(
        [
            [
                detection["bbox_xyxy"][0],
                detection["bbox_xyxy"][1],
                detection["bbox_xyxy"][2],
                detection["bbox_xyxy"][3],
                detection["confidence"],
            ]
            for detection in detections
        ],
        dtype=float,
    )


# calculate_iou()
# Intersection over union of two xyxy boxes.
# Inputs: two boxes as (x1, y1, x2, y2).
# Output: IoU in 0..1, or 0 when the union is empty.
# Verbatim from object_tracking_live.py line 83. It is how a track id is matched
# back to the detection that produced it, because ByteTrack tracks boxes and
# does not carry the class through.
def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0
