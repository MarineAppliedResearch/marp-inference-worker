# keyframes.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Keyframe reduction for the MARP Inference Worker.
# A finished track is a dense list of per-frame boxes; MARP stores a handful of
# keyframes labelled start/middle/end. This file holds the reduction rules that
# make that conversion, each registered under a name and a version.
#
# The reduction is expected to change. Which rule produced an observation is
# recorded with the observation, so a stored row stays attributable to the rule
# that made it (R10b). That is why these are registry entries and not one inline
# function: adding a v4 must not silently reinterpret rows made by v3.
#
# The v3_dirpad body below is a verbatim port of `reduce_to_keyframes_v3_dirpad`
# and `_apply_directional_pad` as defined at lines 814 and 930 of
# src/old_scripts/object_tracking_live.py -- the later of the two definitions in
# that file, which is the one Python keeps and the live path therefore used. The
# earlier definitions differ (vel_window 3 vs 2, speed_gain 0.25 vs 0.35) and are
# deliberately not merged in. Do not tidy the maths here; it is the reference.

# bisect_left finds a keyframe's position in the dense frame list.
from bisect import bisect_left

# math supplies the hypotenuse used for positional error.
import math

# Any and Callable type the registry and the free-form track mapping.
from typing import Any, Callable


# _apply_directional_pad()
# Expands an (x,y,w,h) box on the side of motion using normalized units.
# Inputs: box centre and size, velocity, and the padding coefficients.
# Output: the padded (x, y, w, h) tuple.
# Verbatim from object_tracking_live.py line 930. Growing the box toward the
# direction of travel gives a reviewer a box that still contains the animal on
# the frames between keyframes.
def _apply_directional_pad(x, y, w, h, vx, vy, base_pad, speed_gain, max_pad):

    pad_x = min(max_pad, base_pad + speed_gain * abs(vx))
    pad_y = min(max_pad, base_pad + speed_gain * abs(vy))

    # Shift and scale width based on X velocity
    if pad_x > 0:
        dx = 0.5 * pad_x * w
        w *= (1 + pad_x)
        x += dx if vx > 0 else (-dx if vx < 0 else 0)

    # Shift and scale height based on Y velocity
    if pad_y > 0:
        dy = 0.5 * pad_y * h
        h *= (1 + pad_y)
        y += dy if vy > 0 else (-dy if vy < 0 else 0)

    return x, y, w, h


# reduce_to_keyframes_v3_dirpad()
# Douglas-Peucker keyframe reducer with direction-aware velocity padding.
# Inputs: an ended-track mapping with "frames" and "class_name", plus thresholds.
# Output: list of keyframe dictionaries in the shape MARP's observation rows take.
# Verbatim from object_tracking_live.py line 814, including the defaults.
def reduce_to_keyframes_v3_dirpad(endedObs,
                                  pos_thresh=0.005,
                                  size_thresh=0.01,
                                  vel_window=3,
                                  speed_gain=0.25,
                                  base_pad=0.00,
                                  max_pad=0.12):

    frames = endedObs["frames"]
    comname = endedObs["class_name"]

    # ----------------------------------------------------------------------
    # Recursive simplifier (identical structure, just renamed error fn)
    # ----------------------------------------------------------------------
    def position_size_error_dir(frame, start, end):
        """Same math as original, distinct name for clarity."""
        t = (frame["time"] - start["time"]) / (end["time"] - start["time"] + 1e-8)
        ix = start["bbox"][0] + t * (end["bbox"][0] - start["bbox"][0])
        iy = start["bbox"][1] + t * (end["bbox"][1] - start["bbox"][1])
        iw = start["bbox"][2] + t * (end["bbox"][2] - start["bbox"][2])
        ih = start["bbox"][3] + t * (end["bbox"][3] - start["bbox"][3])
        x, y, w, h = frame["bbox"]
        pos_err = math.hypot(x - ix, y - iy)
        size_err = max(abs(w - iw), abs(h - ih))
        return max(pos_err, size_err)

    def recursive_reduce(seq):
        if len(seq) <= 2:
            return [seq[0], seq[-1]]
        start, end = seq[0], seq[-1]
        max_err, idx = 0.0, 0
        for i in range(1, len(seq) - 1):
            err = position_size_error_dir(seq[i], start, end)
            if err > max_err:
                max_err, idx = err, i
        if max_err > pos_thresh:
            left = recursive_reduce(seq[: idx + 1])
            right = recursive_reduce(seq[idx:])
            return left[:-1] + right
        else:
            return [start, end]

    reduced = recursive_reduce(frames)

    # Map frame numbers for quick raw-frame lookup
    frame_nums = [f["frame"] for f in frames]
    def find_raw_index(frame_num):
        j = bisect_left(frame_nums, frame_num)
        if j < len(frame_nums) and frame_nums[j] == frame_num:
            return j
        if j == 0: return 0
        if j == len(frame_nums): return len(frame_nums) - 1
        return j if abs(frame_nums[j] - frame_num) < abs(frame_nums[j-1] - frame_nums[j-1]) else j - 1

    # ----------------------------------------------------------------------
    # Build output list with direction-aware velocity padding
    # ----------------------------------------------------------------------
    out = []
    for i, kf in enumerate(reduced):
        x, y, w, h = kf["bbox"]

        # Estimate local velocity (vx, vy) from raw frames near this keyframe
        center_idx = find_raw_index(kf["frame"])
        i0 = max(0, center_idx - vel_window)
        i1 = min(len(frames) - 1, center_idx + vel_window)
        if i1 == i0:
            vx = vy = 0.0
        else:
            x0, y0, t0 = frames[i0]["bbox"][0], frames[i0]["bbox"][1], frames[i0]["time"]
            x1, y1, t1 = frames[i1]["bbox"][0], frames[i1]["bbox"][1], frames[i1]["time"]
            dt = (t1 - t0) if (t1 - t0) != 0 else 1e-8
            vx, vy = (x1 - x0) / dt, (y1 - y0) / dt

        # Apply directional padding based on motion
        x, y, w, h = _apply_directional_pad(x, y, w, h, vx, vy,
                                            base_pad=base_pad,
                                            speed_gain=speed_gain,
                                            max_pad=max_pad)

        out.append({
            "subset": "1",
            "comname": comname,
            "type": "start" if i == 0 else ("end" if i == len(reduced) - 1 else "middle"),
            "framenum": kf["frame"],
            "x": x, "y": y, "width": w, "height": h
        })

    return out


# Registered reductions, keyed by (name, version).
# A job spec names one of these; an unknown pair is refused at preflight rather
# than silently falling back, because falling back would mislabel the output.
_REDUCTIONS: dict[tuple[str, str], Callable[..., list[dict[str, Any]]]] = {
    ("v3_dirpad", "1"): reduce_to_keyframes_v3_dirpad,
}


# get_reduction()
# Looks up a registered reduction by name and version.
# Inputs: reduction name and version strings from the job spec.
# Output: the reduction callable.
# Use this from an engine so the spec, not the code, chooses the rule.
def get_reduction(name: str, version: str) -> Callable[..., list[dict[str, Any]]]:

    # Refuse an unknown rule outright: guessing would attribute the output to a
    # rule that did not produce it.
    key = (name, version)
    if key not in _REDUCTIONS:
        raise KeyError(f"Unknown keyframe reduction: {name} version {version}")
    return _REDUCTIONS[key]


# available_reductions()
# Lists the reductions this worker can apply.
# Inputs: none.
# Output: list of {name, version} mappings.
# Use this in the worker's /status and capabilities report, so a coordinator can
# avoid sending a job naming a rule this worker does not have.
def available_reductions() -> list[dict[str, str]]:

    # Sorted so the reported list is stable between calls.
    return [{"name": name, "version": version} for name, version in sorted(_REDUCTIONS)]
