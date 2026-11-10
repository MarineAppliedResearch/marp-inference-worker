# test_habitat_training.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Tests for the habitat training recipe (MarineAppliedResearch/MARP_API#287), on a CPU.
# The recipe is given a tiny stand-in network and a handful of made-up pictures whose
# substrates are their colours, so it can learn them in seconds. What these prove is the
# recipe: reading the archive, the two stages, the per-substrate cut-offs chosen on the
# tuning split, keeping the best pass, stopping. Whether DINOv2 on real video scores as
# the feasibility runs did is the GPU run's job, recorded in .marp/verification.md.

# json and csv write the made-up archive.
import csv
import json

# numpy and pytest for the checks; torch for the stand-in network.
import numpy as np
import pytest
import torch
import torch.nn as nn

# Pillow draws the pictures.
from PIL import Image

# The module under test.
from marp_inference_worker.training.habitat_training import (
    Hooks,
    HabitatRecipe,
    StoppedEarly,
    best_cut_offs,
    layer_groups,
    make_loader,
    positive_weights,
    read_classes,
    read_split,
    score_split,
    train_habitat,
)


# CLASSES
# The five substrates, in the order the labels follow.
CLASSES = ["Rock", "Mud", "Sand", "Cobble", "Boulder"]

# COLOURS
# Each substrate's colour channel in the made-up pictures: a picture with Rock has red.
COLOURS = {"Rock": (0, 200), "Mud": (1, 200), "Sand": (2, 200)}


# write_archive()
# A small habitat dataset archive in tmp_path, laid out as MARP's build writes one.
# Inputs: the directory and frames per split.
# Output: the archive directory.
def write_archive(root, per_split: int = 24):

    # classes.json, then each split's images and labels.csv.
    (root / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        folder = root / split / "images"
        folder.mkdir(parents=True)
        with open(root / split / "labels.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", *CLASSES])
            for i in range(per_split):
                # Each picture has a random mix of Rock, Mud and Sand, shown as colour in its
                # bottom half -- where the habitat crop looks -- and noise in its top half.
                present = {name: bool(rng.integers(0, 2)) for name in CLASSES[:3]}
                pixels = rng.integers(0, 40, size=(32, 48, 3), dtype=np.uint8)
                for name, (channel, value) in COLOURS.items():
                    if present[name]:
                        pixels[16:, :, channel] = value
                Image.fromarray(pixels).save(folder / f"{i}.png")
                writer.writerow([f"images/{i}.png", *[int(present.get(n, False)) for n in CLASSES]])
    return root


# Tiny
# A stand-in for DINOv2: two "blocks" and a head, small enough to train in a second.
class Tiny(nn.Module):

    # Average colour in, five scores out.
    def __init__(self) -> None:

        super().__init__()
        self.patch_embed = nn.Linear(3, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.head = nn.Linear(8, len(CLASSES))

    # Pictures to scores.
    def forward(self, images):

        x = self.patch_embed(images.mean(dim=(2, 3)))
        for block in self.blocks:
            x = torch.relu(block(x))
        return self.head(x)


# RECIPE
# The settled recipe shrunk to run in seconds: tiny pictures, few passes, a fast rate.
RECIPE = HabitatRecipe(input_size=16, batch=8, head_epochs=2, head_lr=5e-2, epochs=6, lr=1e-2,
                       layer_decay=0.75, warmup_epochs=1.0, seed=3)

# CROP
# The habitat crop, x 0.25-0.75 and y 0.5-1.0.
CROP = (0.25, 0.5, 0.75, 1.0)


# hooks()
# Hooks that record what the recipe reports, and stop after a given number of batches.
# Inputs: how many batches before should_stop goes true, or None for never.
# Output: (hooks, record).
def hooks(stop_after: int | None = None):

    # What was reported, and how many batches were checked.
    record = {"epochs": [], "log": [], "checks": 0}

    # A stop arrives once enough batches have run.
    def should_stop() -> bool:

        record["checks"] += 1
        return stop_after is not None and record["checks"] > stop_after

    return Hooks(
        on_epoch=lambda n, stage, metrics: record["epochs"].append((n, stage, dict(metrics))),
        log=record["log"].append,
        should_stop=should_stop,
        wait_while_paused=lambda: False,
    ), record


# loaders()
# The archive's three splits as loaders, with no worker processes.
def loaders(root):

    classes = read_classes(root)
    rows = {s: read_split(root, s, classes) for s in ("train", "val", "test")}
    made = {s: make_loader(rows[s], CROP, RECIPE, train=(s == "train"), workers=0) for s in rows}
    labels = np.array([r[1] for r in rows["train"]])
    return classes, made, labels


# test_reads_the_archive_as_the_build_writes_it()
# classes.json and each split's labels.csv, in class order.
def test_reads_the_archive_as_the_build_writes_it(tmp_path):

    root = write_archive(tmp_path)
    assert read_classes(root) == CLASSES
    rows = read_split(root, "train", CLASSES)
    assert len(rows) == 24
    assert rows[0][0].exists() and len(rows[0][1]) == 5
    assert all(r[1][3] == 0 and r[1][4] == 0 for r in rows)


# test_refuses_labels_missing_a_substrate()
# A labels.csv without a column for a class cannot be trained on.
def test_refuses_labels_missing_a_substrate(tmp_path):

    root = write_archive(tmp_path)
    with pytest.raises(ValueError, match="no column for Gravel"):
        read_split(root, "train", [*CLASSES, "Gravel"])


# test_cut_offs_are_chosen_per_substrate_on_the_tuning_split()
# Each substrate gets the cut-off that gives it the best F1, independently.
def test_cut_offs_are_chosen_per_substrate_on_the_tuning_split():

    # Substrate 0 separates at 0.7, substrate 1 at 0.3.
    scores = np.array([[0.9, 0.4], [0.8, 0.35], [0.2, 0.1], [0.6, 0.2]])
    labels = np.array([[1, 1], [1, 1], [0, 0], [0, 0]])
    cut_offs = best_cut_offs(scores, labels)
    assert 0.6 < cut_offs[0] <= 0.8
    assert 0.2 < cut_offs[1] <= 0.35
    scored = score_split(scores, labels, ["A", "B"], cut_offs)
    assert scored["macro_f1"] == 1.0
    assert scored["substrates"]["A"]["frames_present"] == 2


# test_rare_substrates_weigh_more_in_the_loss()
# Positives count by how rare they are; an absent substrate gets 1, not a division by 0.
def test_rare_substrates_weigh_more_in_the_loss():

    labels = np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]])
    assert positive_weights(labels) == [1 / 3, 3.0, 1.0]


# test_deeper_blocks_learn_more_slowly()
# Layer-wise speeds: the top at the full rate, each block below 0.75 times slower.
def test_deeper_blocks_learn_more_slowly():

    groups = layer_groups(Tiny(), 1.0, 0.75)
    rates = [g["lr"] for g in groups]
    assert rates == pytest.approx([0.75 ** 3, 0.75 ** 2, 0.75, 1.0])
    assert sum(len(g["params"]) for g in groups) == len(list(Tiny().parameters()))


# test_trains_both_stages_and_keeps_the_best_pass()
# The whole recipe on the stand-in: both stages run, every pass is reported once with
# its tuning scores, and the model learns the colours well above chance.
def test_trains_both_stages_and_keeps_the_best_pass(tmp_path):

    classes, made, labels = loaders(write_archive(tmp_path, per_split=64))
    # Seeded before the network is built, so its starting weights are the same every run.
    torch.manual_seed(0)
    model = Tiny()
    hook, record = hooks()
    result = train_habitat(model, model.head, made["train"], made["val"], labels, RECIPE,
                           torch.device("cpu"), classes, hook)

    # Eight passes, numbered once across both stages.
    assert [(n, s) for n, s, _ in record["epochs"]] == [(1, "frozen"), (2, "frozen")] + [(n, "fine-tune") for n in range(3, 9)]
    assert all("val_f1/Sand" in m and "loss" in m for _, _, m in record["epochs"])

    # The kept pass is the best tuning score reported, and its weights are what the model holds.
    best = max(m["val_macro_f1"] for _, _, m in record["epochs"])
    assert result["tuning"]["macro_f1"] == best
    assert len(result["cut_offs"]) == 5
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in result["state"].items())

    # Rock, Mud and Sand are learnable from colour. Saying "present" on every frame scores
    # about 0.67 here; across five seeds the recipe averaged 0.90 to 1.0 over the three.
    learnable = [result["tuning"]["substrates"][name]["f1"] for name in ("Rock", "Mud", "Sand")]
    assert np.mean(learnable) >= 0.85, result["tuning"]
    assert min(learnable) >= 0.65, result["tuning"]


# test_a_stop_unwinds_the_recipe()
# A stop between batches raises StoppedEarly rather than finishing the pass.
def test_a_stop_unwinds_the_recipe(tmp_path):

    classes, made, labels = loaders(write_archive(tmp_path))
    model = Tiny()
    hook, record = hooks(stop_after=2)
    with pytest.raises(StoppedEarly):
        train_habitat(model, model.head, made["train"], made["val"], labels, RECIPE,
                      torch.device("cpu"), classes, hook)
    assert record["epochs"] == []


# test_a_loader_with_worker_processes_reads_frames()
# The loaders run with worker processes in a real job, and on Windows each is started fresh
# and sent the dataset by pickling. The first GPU run failed there, at its first batch, with
# "Can't get local object 'make_loader.<locals>.Frames'"; the other tests here use no worker
# processes and could not see it.
def test_a_loader_with_worker_processes_reads_frames(tmp_path):

    root = write_archive(tmp_path, per_split=4)
    classes = read_classes(root)
    loader = make_loader(read_split(root, "val", classes), CROP, RECIPE, train=False, workers=1)
    images, labels = next(iter(loader))
    assert tuple(images.shape) == (4, 3, 16, 16)
    assert tuple(labels.shape) == (4, 5)
