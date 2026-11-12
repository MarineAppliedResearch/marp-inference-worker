# test_habitat_engine.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Tests for the habitat inference engine (MarineAppliedResearch/MARP_API#287), on a CPU.
# A real video file is written and decoded: 20 s at 25 fps whose bottom half is red for the
# first 10 s and blue after, so the habitat crop changes colour half way. A stand-in model
# scores Rock from red and Sand from blue. What these prove is the engine: the frames it reads,
# the crop it scores, the stretches it makes and the two files it publishes. Scoring real video
# with a trained DINOv2 is the GPU run's job.

# json reads what was published.
import json

# Path types the job directory.
from pathlib import Path

# OpenCV writes the video; numpy draws it; pytest and torch for the checks and the stand-in.
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

# The engine under test and its refusal.
from marp_inference_worker.engines.base_engine import JobUnrunnable
from marp_inference_worker.engines.habitat_engine import HabitatEngine

# The fake context from the training engine's tests.
from test_habitat_training_engine import Context


# SUBSTRATES
# The stand-in's outputs, in the rules' order.
SUBSTRATES = ["Rock", "Mud", "Sand", "Cobble", "Boulder"]


# write_video()
# 20 s at 25 fps, 64 x 48: noise on top, the bottom half red for 10 s then blue.
def write_video(path: Path) -> Path:

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    rng = np.random.default_rng(0)
    for i in range(500):
        frame = rng.integers(0, 40, size=(48, 64, 3), dtype=np.uint8)
        # OpenCV's channels are blue, green, red.
        frame[24:, :, 2 if i < 250 else 0] = 220
        writer.write(frame)
    writer.release()
    return path


# ColourModel
# Rock from the crop's red, Sand from its blue; nothing else. Fed normalised images.
class ColourModel(nn.Module):

    # One parameter, so .to() and .eval() behave as for a real model.
    def __init__(self) -> None:

        super().__init__()
        self.scale = nn.Parameter(torch.tensor(4.0))

    # The mean of each channel, turned into five logits.
    def forward(self, images):

        red, blue = images[:, 0].mean(dim=(1, 2)), images[:, 2].mean(dim=(1, 2))
        zero = torch.full_like(red, -10.0)
        return torch.stack([self.scale * red, zero, self.scale * blue, zero, zero], dim=1)


# Engine
# The real engine with the stand-in model.
class Engine(HabitatEngine):

    def build_model(self, classes, input_size, weights):

        return ColourModel()


# RULES
# Decision rules as MARP sends them for a run.
RULES = {
    "substrates": SUBSTRATES,
    "cut_offs": {"Rock": 0.5, "Mud": 0.5, "Sand": 0.5, "Cobble": 0.5, "Boulder": 0.5},
    "crop": {"x0": 0.25, "y0": 0.5, "x1": 0.75, "y1": 1.0},
    "input_size": 16, "samples_per_second": 3, "average_s": 1, "exclusive": [["Mud", "Sand"]],
    "min_stretch_s": 2, "bridge_s": 2,
}


# spec_for()
# A habitat run over the whole video.
def spec_for(path: Path, **override) -> dict:

    spec = {
        "engine": "marp_habitat",
        "video": {"url": str(path), "frame_rate": 25.0, "source_name": "20240730_171520_Fwd.mp4",
                  "jellyfin_item_id": "abc"},
        "range": {"start_frame": 0, "end_frame": 500},
        "model": {"name": "habitat", "decision_rules": RULES},
        "reduction": {"name": "habitat_stretches", "version": "1"},
        "params": {"device": "cpu", "model_path": "unused.pt"},
    }
    spec.update(override)
    return spec


# test_refuses_a_run_it_cannot_make()
# Another reduction, or rules missing a field, or no video: refused before anything starts.
def test_refuses_a_run_it_cannot_make(tmp_path):

    engine = HabitatEngine()
    video = write_video(tmp_path / "v.avi")
    engine.preflight(spec_for(video))
    with pytest.raises(JobUnrunnable, match="habitat_stretches"):
        engine.preflight(spec_for(video, reduction={"name": "v3_dirpad", "version": "1"}))
    with pytest.raises(JobUnrunnable, match="cut_offs"):
        engine.preflight(spec_for(video, model={"decision_rules": {k: v for k, v in RULES.items() if k != "cut_offs"}}))
    with pytest.raises(JobUnrunnable, match="video.url"):
        engine.preflight(spec_for(video, video={}))


# test_a_run_makes_the_stretches_the_video_shows()
# Rock for the first ten seconds, Sand for the last ten, each starting at its own frame, and
# every scored frame's numbers beside them.
def test_a_run_makes_the_stretches_the_video_shows(tmp_path):

    video = write_video(tmp_path / "v.avi")
    workdir = tmp_path / "job"
    workdir.mkdir()
    ctx = Context(workdir)
    summary = Engine().run(ctx, spec_for(video))

    assert set(ctx.published) == {"habitat_stretches", "habitat_scores"}
    stretches = [json.loads(line) for line in ctx.published["habitat_stretches"].read_text(encoding="utf-8").splitlines()]
    assert [s["substrate"] for s in stretches] == ["Rock", "Sand"]
    rock, sand = stretches
    assert rock["start_ms"] == 0 and rock["start_frame"] == 0
    # The change at 10 s, to within the averaging window either side.
    assert 9000 <= rock["end_ms"] <= 11000
    assert 9000 <= sand["start_ms"] <= 11000
    assert sand["end_frame"] >= 490
    assert all(s["video_source"] == "20240730_171520_Fwd.mp4" and s["jellyfin_item_id"] == "abc" for s in stretches)

    # Three frames a second over 20 s, every one with five scores.
    scores = [json.loads(line) for line in ctx.published["habitat_scores"].read_text(encoding="utf-8").splitlines()]
    assert 58 <= len(scores) <= 61
    assert all(len(row["scores"]) == 5 for row in scores)

    assert summary["processed_range"] == {"start_frame": 0, "end_frame": 500}
    assert summary["stretches"] == 2 and summary["stopped_early"] is False


# test_a_stop_publishes_the_stretches_scored_so_far()
# A stop part way through still hands over what was scored, and says how far it got.
def test_a_stop_publishes_the_stretches_scored_so_far(tmp_path):

    video = write_video(tmp_path / "v.avi")
    workdir = tmp_path / "job"
    workdir.mkdir()
    ctx = Context(workdir, stop_after_checks=1)
    summary = Engine().run(ctx, spec_for(video))

    assert summary["stopped_early"] is True
    assert 0 < summary["processed_range"]["end_frame"] < 500
    assert "habitat_stretches" in ctx.published
