# habitat_stretches.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Turning a habitat model's per-frame scores into habitat stretches
# (MarineAppliedResearch/MARP_API#287).
# A habitat model scores each substrate on frames a few times a second. Annotators record
# something else: stretches of time, one substrate each, overlapping where the seabed is
# mixed. This is the rule between the two, settled on the feasibility runs and applied with
# the settings stored with each model (its "decision rules"):
#   - each substrate's scores averaged over a short window of time;
#   - a substrate is present on a frame when its score reaches its cut-off;
#   - of a never-together group (Mud, Sand, Gravel), at most one per frame: the surer stays;
#   - each substrate is its own layer; breaks of the bridging gap or less do not end a
#     stretch, and stretches shorter than the minimum are dropped.
# Pure: numbers in, stretches out. Decoding video and running the model belong elsewhere.

# Types for the rules and the result.
from collections.abc import Mapping, Sequence
from typing import Any

# numpy does the arithmetic over a whole video's frames at once.
import numpy as np


# HABITAT_REDUCTION
# The rule's name and version, as a habitat run's spec names it in `reduction`.
HABITAT_REDUCTION = ("habitat_stretches", "1")


# frame_step_ms()
# How much video one scored frame stands for, from the rules.
# Inputs: the decision rules.
# Output: milliseconds between scored frames.
def frame_step_ms(rules: Mapping[str, Any]) -> float:

    return 1000.0 / float(rules["samples_per_second"])


# runs_of_frames()
# Where the scored frames break into unbroken runs: a gap of more than two frames' worth of
# time, a jump in the video or a cut between files, starts a new run.
# Inputs: frame times in ms, sorted, and the step.
# Output: list of index arrays, one per run.
def runs_of_frames(times_ms: np.ndarray, step_ms: float) -> list[np.ndarray]:

    breaks = np.where(np.diff(times_ms) > 2 * step_ms)[0] + 1
    return np.split(np.arange(len(times_ms)), breaks)


# average_scores()
# Each frame's scores averaged over a window of time centred on it, within its run.
# Inputs: frame times (ms), scores (frames x substrates), window in seconds, step in ms.
# Output: averaged scores, same shape.
def average_scores(times_ms: np.ndarray, scores: np.ndarray, window_s: float, step_ms: float) -> np.ndarray:

    # A window of nothing, or of one frame, leaves the scores as they are.
    half = int(round(window_s * 1000 / step_ms / 2))
    if half <= 0:
        return scores.copy()

    # Averaged with a box of 2*half+1 frames, divided by how many frames each box really
    # covered so a run's first and last frames are not pulled towards zero.
    out = np.empty_like(scores, dtype=float)
    kernel = np.ones(2 * half + 1)

    # The centred window over each frame of a run. numpy's "same" mode returns as many values
    # as the longer of its two inputs, so a run shorter than the window would come back too
    # long; the full convolution, cut at the centre, is the same answer at any length.
    def centred(values: np.ndarray) -> np.ndarray:

        return np.convolve(values, kernel, "full")[half:half + len(values)]

    for run in runs_of_frames(times_ms, step_ms):
        covered = centred(np.ones(len(run)))
        for c in range(scores.shape[1]):
            out[run, c] = centred(scores[run, c]) / covered
    return out


# keep_one_of_each_group()
# Of each never-together group, at most one substrate per frame: where two or more reach
# their cut-offs, the one with the highest score stays and the others are cleared.
# Inputs: scores, the substrates in column order, cut-offs by substrate, the groups.
# Output: scores with the others cleared to 0, same shape. Ties keep the first in the group.
def keep_one_of_each_group(scores: np.ndarray, substrates: Sequence[str], cut_offs: Mapping[str, float],
                           groups: Sequence[Sequence[str]]) -> np.ndarray:

    out = scores.copy()
    for group in groups:
        # Only the members this model has.
        columns = [substrates.index(name) for name in group if name in substrates]
        if len(columns) < 2:
            continue
        passing = np.stack([out[:, c] >= cut_offs[substrates[c]] for c in columns], axis=1)
        contested = passing.sum(axis=1) > 1
        if not contested.any():
            continue
        # The surest member of the group on each contested frame; argmax keeps the first on a tie.
        winner = np.argmax(np.stack([out[:, c] for c in columns], axis=1), axis=1)
        for position, c in enumerate(columns):
            out[contested & (winner != position), c] = 0.0
    return out


# stretches_for()
# The habitat stretches a set of scored frames makes, by the decision rules.
# Inputs: frame indices and times (ms) in the video, sorted by time; scores (frames x
# substrates, in the rules' substrate order); the decision rules as MARP sends them.
# Output: list of {substrate, start_ms, end_ms, start_frame, end_frame, confidence}, by start
# then substrate order. end_ms is the end of the last frame's share of time.
def stretches_for(frames: Sequence[int], times_ms: Sequence[float], scores: np.ndarray,
                  rules: Mapping[str, Any]) -> list[dict[str, Any]]:

    substrates = list(rules["substrates"])
    cut_offs = rules["cut_offs"]
    times = np.asarray(times_ms, dtype=float)
    frames = np.asarray(frames)
    if len(times) == 0:
        return []
    step = frame_step_ms(rules)

    # Averaged, then one of each never-together group per frame.
    averaged = average_scores(times, np.asarray(scores, dtype=float), float(rules["average_s"]), step)
    scored = keep_one_of_each_group(averaged, substrates, cut_offs, rules.get("exclusive") or [])

    bridge_ms = float(rules["bridge_s"]) * 1000
    minimum_ms = float(rules["min_stretch_s"]) * 1000
    out = []
    for c, name in enumerate(substrates):
        present = scored[:, c] >= cut_offs[name]

        # Runs of frames over the cut-off, broken also by a gap in the frames themselves.
        runs, start = [], None
        for i, on in enumerate(present):
            broken = i > 0 and times[i] - times[i - 1] > 2 * step
            if start is not None and (not on or broken):
                runs.append([start, i - 1])
                start = None
            if on and start is None:
                start = i
        if start is not None:
            runs.append([start, len(present) - 1])

        # A break of the bridging gap or less does not end a stretch.
        merged: list[list[int]] = []
        for run in runs:
            if merged and times[run[0]] - (times[merged[-1][1]] + step) <= bridge_ms:
                merged[-1][1] = run[1]
            else:
                merged.append(run)

        # Long enough to keep, each with the average score over its frames.
        for a, b in merged:
            end_ms = times[b] + step
            if end_ms - times[a] >= minimum_ms:
                out.append({
                    "substrate": name,
                    "start_ms": int(round(times[a])),
                    "end_ms": int(round(end_ms)),
                    "start_frame": int(frames[a]),
                    "end_frame": int(frames[b]),
                    # Over the scores the decision was made on, as the feasibility runs did.
                    "confidence": round(float(scored[a:b + 1, c].mean()), 4),
                })
    out.sort(key=lambda s: (s["start_ms"], substrates.index(s["substrate"])))
    return out
