# test_habitat_stretches.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Tests for turning habitat scores into stretches (MarineAppliedResearch/MARP_API#287).
# Each rule on its own, then the whole against the feasibility code the rules were measured
# with (sample_output.py on branch 287-habitat-classification), copied here as the reference,
# on random scores: the product must make the stretches that were scored.

# numpy builds the scores.
import numpy as np

# The module under test.
from marp_inference_worker.reduction.habitat_stretches import (
    HABITAT_REDUCTION,
    average_scores,
    keep_one_of_each_group,
    stretches_for,
)


# SUBSTRATES
# The model's outputs, in order.
SUBSTRATES = ["Rock", "Mud", "Sand", "Cobble", "Boulder"]


# rules()
# Decision rules as MARP sends them, with the settled stretch rules unless overridden.
def rules(**override):

    base = {
        "substrates": SUBSTRATES,
        "cut_offs": {"Rock": 0.5, "Mud": 0.5, "Sand": 0.5, "Cobble": 0.5, "Boulder": 0.5},
        "samples_per_second": 3, "average_s": 1, "exclusive": [["Mud", "Sand"]],
        "min_stretch_s": 2, "bridge_s": 2,
    }
    return {**base, **override}


# frames_over()
# Frame numbers and times for `seconds` of video at 3 a second, from a 25 fps video.
def frames_over(seconds: float, start_s: float = 0.0):

    times = [round((start_s + i / 3) * 1000) for i in range(int(seconds * 3))]
    return [round(t * 25 / 1000) for t in times], times


# column()
# Scores for one substrate across `n` frames, all others 0.
def column(n: int, name: str, values) -> np.ndarray:

    scores = np.zeros((n, len(SUBSTRATES)))
    scores[:, SUBSTRATES.index(name)] = values
    return scores


# test_the_rule_is_named_as_a_spec_names_it()
def test_the_rule_is_named_as_a_spec_names_it():

    assert HABITAT_REDUCTION == ("habitat_stretches", "1")


# test_one_steady_substrate_is_one_stretch_over_the_whole_time()
def test_one_steady_substrate_is_one_stretch_over_the_whole_time():

    frames, times = frames_over(10)
    out = stretches_for(frames, times, column(len(times), "Rock", 0.9), rules())
    assert len(out) == 1
    s = out[0]
    assert (s["substrate"], s["start_ms"], s["start_frame"]) == ("Rock", 0, 0)
    # The last frame stands for a third of a second after it.
    assert s["end_ms"] == round(times[-1] + 1000 / 3)
    assert s["end_frame"] == frames[-1]
    assert s["confidence"] == 0.9


# test_a_stretch_shorter_than_two_seconds_is_dropped()
def test_a_stretch_shorter_than_two_seconds_is_dropped():

    frames, times = frames_over(10)
    values = np.zeros(len(times))
    values[3:7] = 0.9   # 4 frames: 1.33 s
    assert stretches_for(frames, times, column(len(times), "Cobble", values), rules(average_s=0)) == []
    values[3:9] = 0.9   # 6 frames: 2 s
    assert len(stretches_for(frames, times, column(len(times), "Cobble", values), rules(average_s=0))) == 1


# test_a_break_of_two_seconds_or_less_is_bridged_and_a_longer_one_is_not()
def test_a_break_of_two_seconds_or_less_is_bridged_and_a_longer_one_is_not():

    frames, times = frames_over(20)
    values = np.full(len(times), 0.9)
    values[15:21] = 0.1   # six frames off: the gap between stretches is 2 s
    one = stretches_for(frames, times, column(len(times), "Sand", values), rules(average_s=0, exclusive=[]))
    assert len(one) == 1
    values[15:22] = 0.1   # seven off: 2.33 s
    two = stretches_for(frames, times, column(len(times), "Sand", values), rules(average_s=0, exclusive=[]))
    assert len(two) == 2


# test_mud_and_sand_are_never_together_and_the_surer_stays()
def test_mud_and_sand_are_never_together_and_the_surer_stays():

    scores = np.array([[0, 0.8, 0.6, 0, 0], [0, 0.55, 0.7, 0, 0], [0, 0.4, 0.9, 0, 0], [0, 0.6, 0.6, 0, 0]])
    cut = {n: 0.5 for n in SUBSTRATES}
    kept = keep_one_of_each_group(scores, SUBSTRATES, cut, [["Mud", "Sand", "Gravel"]])
    assert kept[0, 1] == 0.8 and kept[0, 2] == 0          # Mud surer
    assert kept[1, 1] == 0 and kept[1, 2] == 0.7          # Sand surer
    assert kept[2, 1] == 0.4 and kept[2, 2] == 0.9        # only Sand passed: nothing contested
    assert kept[3, 1] == 0.6 and kept[3, 2] == 0          # a tie keeps the first in the group
    # A group with one member this model has constrains nothing.
    assert np.array_equal(keep_one_of_each_group(scores, SUBSTRATES, cut, [["Mud", "Gravel"]]), scores)


# test_scores_are_averaged_over_a_second_within_unbroken_runs()
# Including a run shorter than the window, which numpy's "same" mode got wrong.
def test_scores_are_averaged_over_a_second_within_unbroken_runs():

    times = np.array([0, 333, 667, 1000, 5000, 5333])
    scores = np.array([[0.0], [0.9], [0.0], [0.0], [0.6], [0.0]])
    averaged = average_scores(times, scores, 1.0, 1000 / 3)
    # One second at 3 a second is five frames centred, as the feasibility runs computed it
    # (half the window, 1.5 frames, rounds to 2); a run's ends average over what exists.
    assert np.allclose(averaged[:4, 0], [0.3, 0.225, 0.225, 0.3])
    # The jump at 5 s starts a new run: 0.6 is not averaged with the frames before it.
    assert np.allclose(averaged[4:, 0], [0.3, 0.3])


# reference_stretches()
# sample_output.py's rules on branch 287-habitat-classification, copied for comparison:
# smooth(), the one_soft step and stretches(), with its globals as arguments. One change: its
# np.convolve(..., "same") returns too many values for a run shorter than the window, which
# its whole lines never had; the centred full convolution here is identical otherwise.
def reference_stretches(clock, p, th, gap_s, step_ms, smooth_s, min_s):

    half = int(round(smooth_s * 1000 / step_ms / 2))
    if half:
        out = np.empty_like(p)
        breaks = np.where(np.diff(clock) > 2 * step_ms)[0] + 1
        for seg in np.split(np.arange(len(clock)), breaks):
            k = np.ones(2 * half + 1)
            same = lambda v: np.convolve(v, k, "full")[half:half + len(v)]
            n = same(np.ones(len(seg)))
            for c in range(p.shape[1]):
                out[seg, c] = same(p[seg, c]) / n
        p = out
    mud, sand = 1, 2
    both = (p[:, mud] >= th[mud]) & (p[:, sand] >= th[sand])
    p = p.copy()
    p[both & (p[:, mud] < p[:, sand]), mud] = 0
    p[both & (p[:, sand] <= p[:, mud]), sand] = 0
    result = []
    for c, name in enumerate(SUBSTRATES):
        on = p[:, c] >= th[c]
        runs, start = [], None
        for i, v in enumerate(on):
            broken = i > 0 and clock[i] - clock[i - 1] > 2 * step_ms
            if start is not None and (not v or broken):
                runs.append([start, i - 1])
                start = None
            if v and start is None:
                start = i
        if start is not None:
            runs.append([start, len(on) - 1])
        merged = []
        for r in runs:
            if merged and clock[r[0]] - (clock[merged[-1][1]] + step_ms) <= gap_s * 1000:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        for a, b in merged:
            if clock[b] + step_ms - clock[a] >= min_s * 1000:
                result.append((name, int(round(clock[a])), round(float(p[a:b + 1, c].mean()), 4)))
    return sorted(result, key=lambda s: (s[1], SUBSTRATES.index(s[0])))


# test_the_stretches_are_the_ones_the_feasibility_runs_measured()
# On random scores with gaps in the frames: the same stretches, starts and confidences.
def test_the_stretches_are_the_ones_the_feasibility_runs_measured():

    rng = np.random.default_rng(287)
    step = 1000 / 3
    cut = [0.45, 0.6, 0.5, 0.4, 0.35]
    for trial in range(20):
        n = 600
        clock = np.cumsum(np.where(rng.random(n) < 0.01, 4000, step))
        # Scores that wander, so stretches form and break.
        p = np.clip(np.cumsum(rng.normal(0, 0.08, size=(n, 5)), axis=0) % 1.0, 0, 1)
        mine = stretches_for(list(range(n)), clock, p, rules(cut_offs=dict(zip(SUBSTRATES, cut))))
        theirs = reference_stretches(clock, p, cut, 2, step, 1, 2)
        assert [(s["substrate"], s["start_ms"], s["confidence"]) for s in mine] == theirs, f"trial {trial}"
