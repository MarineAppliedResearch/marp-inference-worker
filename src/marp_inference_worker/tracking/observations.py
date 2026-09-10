# observations.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Observation shaping for the MARP Inference Worker.
# A finished, reduced track has to be expressible in the terms MARP already
# stores. object_tracking_live.py posts exactly these fields to /api/observation,
# and that write is the working precedent this must preserve: session_id, comname,
# taxserial, count, tc, frame, video_source, videoLocation, mediaPosition,
# actualPosition and keyframes[].
#
# The worker builds the payload but does not decide anything about it. Whether a
# detection becomes an observation, how a class name maps to a taxserial and
# whether a row needs human review are all coordinator decisions -- so the fields
# the worker cannot know are left out rather than guessed at, and the coordinator
# fills them from the job it issued.
#
# The timecode helpers here are ported from object_tracking_live.py unchanged.

# Any types the free-form keyframe list and payload.
from typing import Any


# frame_to_timecode()
# Converts an absolute frame index into MARP's whole-second timecode.
# Inputs: frame index and frame rate.
# Output: "HH:MM:SS".
# Verbatim from object_tracking_live.py line 1460.
def frame_to_timecode(frame_idx, fps):

    seconds = frame_idx / fps
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    return f"{hh:02}:{mm:02}:{ss:02}"


# frame_to_media_position()
# Converts an absolute frame index into MARP's millisecond media position.
# Inputs: frame index and frame rate.
# Output: "HH:MM:SS.mmm".
# Verbatim from object_tracking_live.py line 1467.
def frame_to_media_position(frame_idx, fps):

    seconds = frame_idx / fps
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    ms = (seconds - int(seconds)) * 1000
    return f"{hh:02}:{mm:02}:{ss:02}.{int(ms):03}"


# pick_observation_time()
# Chooses which frame of a track the observation is recorded at.
# Inputs: the track's dense frame list and the dataset type.
# Output: the chosen frame mapping, or None when the track had no frames.
# Verbatim from object_tracking_live.py line 1476. The rules encode how MARP
# counts: a fish is counted when it crosses near the bottom of frame, an
# invertebrate when it enters the bottom-centre trapezoid. Do not generalize
# these -- they are survey conventions, not geometry.
def pick_observation_time(frames, data_type):

    if data_type in ("Fish", "GULF_Fish"):
        # Fish: first time center crosses near bottom (y > 0.8 normalized)
        for f in frames:
            xCenter, yCenter, w, h = f["bbox"]
            if yCenter > 0.8:  # bottom 20% of screen
                return f

    elif data_type in ("Invert", "GULF_Inverts"):
        # Inverts: when center enters trapezoid at bottom half
        candidate = None
        best_bottom = -1.0

        for f in frames:
            xCenter, yCenter, h = f["bbox"][0], f["bbox"][1], f["bbox"][3]

            if yCenter < 0.5:  # must be in bottom half
                continue

            # trapezoid shrinks as you go down
            half_width = 0.5 * (1 - yCenter)
            x_min, x_max = 0.5 - half_width, 0.5 + half_width

            if x_min <= xCenter <= x_max:
                # bottom of bbox
                y_bottom = yCenter + h / 2
                if y_bottom <= 1.0 and y_bottom > best_bottom:
                    best_bottom = y_bottom
                    candidate = f

        if candidate:
            return candidate

    # Fallback: middle frame
    if frames:
        return frames[len(frames) // 2]
    return None


# build_observation()
# Builds one observation payload from one finished, reduced track.
# Inputs: the track's dense frames, its reduced keyframes, the video's source
# name and frame rate, the dataset type, the reduction that made the keyframes,
# and why the track ended.
# Output: JSON-safe observation mapping.
# Use this once per finished track. The result goes into the job's results file,
# never inline to the coordinator (R11).
def build_observation(
    frames: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    video_source_name: str,
    jellyfin_item_id: str | None,
    frame_rate: float,
    data_type: str,
    reduction_name: str,
    reduction_version: str,
    track_id: int,
    end_reason: str,
) -> dict[str, Any]:

    # Choose the frame the observation is recorded at, by MARP's survey rules.
    chosen = pick_observation_time(frames, data_type)
    chosen_frame = chosen["frame"] if chosen else frames[0]["frame"]

    # The fields below are exactly what the live script posts, with two
    # additions the worker is now able to supply and the coordinator needs:
    # which reduction produced the keyframes (R10b), and why the track ended.
    #
    # session_id, taxserial and videoLocation are deliberately absent. The worker
    # does not know which session a job belongs to; mapping a class name to a
    # taxserial is a database decision the coordinator owns; and videoLocation in
    # the live script was the operator's own local path, which on a distributed
    # worker would be a Jellyfin stream url and meaningless to MARP. The
    # coordinator fills all three from the job it issued -- which is why the
    # Jellyfin item id is carried back out below, so it can. It is optional:
    # a job handed a bare url has none, and the worker must not invent one.
    #
    # Nothing the live script recorded is dropped: every one of those three is a
    # value the coordinator already holds, not a value the worker measured.
    return {
        "comname": keyframes[0]["comname"] if keyframes else None,
        "count": 1,
        "tc": frame_to_timecode(chosen_frame, frame_rate),
        # The sub-second frame index within its own second, as MARP stores it.
        "frame": str(chosen_frame % int(frame_rate)),
        "video_source": video_source_name,
        # So the coordinator can resolve session_id and videoLocation. Opaque
        # provenance: echoed exactly as the job spec carried it, and null when
        # the job carried none, because the worker has nothing to put there.
        "jellyfin_item_id": jellyfin_item_id,
        "mediaPosition": frame_to_media_position(chosen_frame, frame_rate),
        "actualPosition": frame_to_media_position(chosen_frame, frame_rate),
        "keyframes": keyframes,
        # Provenance, so a stored row stays attributable to the rule that made it.
        "reduction": {"name": reduction_name, "version": reduction_version},
        # Valid only within this job's frame range; never compared across ranges.
        "track_id": track_id,
        "track_end_reason": end_reason,
        "observation_frame": chosen_frame,
    }
