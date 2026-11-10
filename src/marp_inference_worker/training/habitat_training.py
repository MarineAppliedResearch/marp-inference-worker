# habitat_training.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Training a habitat substrate model (MarineAppliedResearch/MARP_API#287).
# A habitat model answers yes or no for each substrate on a frame, several at once,
# from the habitat crop of the frame. This file holds the dataset reader, the scoring
# and the two-stage recipe the feasibility runs settled: a frozen backbone with only
# the head training, then the whole model at layer-wise speeds, keeping whichever pass
# scored best on the tuning dives, with each substrate's cut-off picked there too.
# The model itself is passed in, so the recipe can be tested on a tiny network on a CPU;
# building DINOv2 and talking to MARP belong to the engine.

# csv and json read the dataset archive's label files and class list.
import csv
import json

# math shapes the learning-rate schedule.
import math

# time measures each pass, for the operator.
import time

# Path locates the archive's files.
from pathlib import Path

# Types for the callbacks and the result.
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

# numpy does the scoring arithmetic over whole splits at once.
import numpy as np


# SPLITS
# The archive's three splits: training, tuning and test dives, whole dives each.
SPLITS = ("train", "val", "test")

# CUT_OFF_GRID
# The cut-offs tried for each substrate on the tuning dives, as the feasibility runs did.
CUT_OFF_GRID = np.linspace(0.05, 0.95, 19)


# HabitatRecipe
# The settled training recipe (MARP_API#287, 2026-11-10). Fields are overridable from
# the job's params, so a run can be repeated with one setting changed.
class HabitatRecipe(NamedTuple):

    # Picture size the crop is resized to before the model sees it.
    input_size: int = 518

    # Frames per batch; 16 fits DINOv2 ViT-B at 518 px on an 8 GB card.
    batch: int = 16

    # Stage 1: passes with only the head training, and its learning rate.
    head_epochs: int = 3
    head_lr: float = 1e-3

    # Stage 2: passes over the whole model, its top learning rate, and how much slower
    # each block below the top learns (layer-wise decay).
    epochs: int = 10
    lr: float = 3e-5
    layer_decay: float = 0.75

    # Stage 2's warm-up, in passes, before a cosine slow-down stepped every batch.
    warmup_epochs: float = 1.0

    # The random seed, so a run can be repeated.
    seed: int = 1


# read_classes()
# The substrates of a habitat dataset archive, in the order its labels follow.
# Inputs: the unpacked archive directory.
# Output: the class names, e.g. ["Rock", "Mud", "Sand", "Cobble", "Boulder"].
def read_classes(root: Path) -> list[str]:

    # classes.json is a plain list, written by MARP's habitat dataset build.
    classes = json.loads((Path(root) / "classes.json").read_text(encoding="utf-8"))
    if not isinstance(classes, list) or not classes or not all(isinstance(c, str) and c for c in classes):
        raise ValueError("classes.json must be a list of substrate names")
    return classes


# read_split()
# One split's frames and their yes/no labels.
# Inputs: the archive directory, the split name and the class list.
# Output: a list of (image path, labels as floats in class order).
# Use this for each split before building its loader.
def read_split(root: Path, split: str, classes: Sequence[str]) -> list[tuple[Path, list[float]]]:

    # Each split has labels.csv: an image path relative to the split, then one 0/1 column
    # per class, named as in classes.json.
    folder = Path(root) / split
    rows = []
    with open(folder / "labels.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            missing = [c for c in classes if c not in row]
            if missing:
                raise ValueError(f"{split}/labels.csv has no column for {', '.join(missing)}")
            rows.append((folder / row["image"], [float(row[c]) for c in classes]))
    return rows


# f1_at()
# F1, precision and recall of one substrate's scores at one cut-off.
# Inputs: scores and 0/1 labels for one substrate, and the cut-off.
# Output: (f1, precision, recall).
def f1_at(scores: np.ndarray, labels: np.ndarray, cut_off: float) -> tuple[float, float, float]:

    # Present where the score reaches the cut-off.
    predicted = scores >= cut_off
    true_positive = float((predicted & (labels == 1)).sum())
    precision = true_positive / max(float(predicted.sum()), 1.0)
    recall = true_positive / max(float((labels == 1).sum()), 1.0)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return f1, precision, recall


# best_cut_offs()
# The cut-off per substrate that gives its best F1, chosen on the tuning dives.
# Inputs: scores and labels, both shaped (frames, substrates).
# Output: one cut-off per substrate, from CUT_OFF_GRID.
def best_cut_offs(scores: np.ndarray, labels: np.ndarray) -> list[float]:

    # Each substrate on its own: they are separate yes/no questions.
    return [
        float(CUT_OFF_GRID[int(np.argmax([f1_at(scores[:, c], labels[:, c], t)[0] for t in CUT_OFF_GRID]))])
        for c in range(scores.shape[1])
    ]


# score_split()
# Per-substrate F1, precision and recall at given cut-offs, and their average.
# Inputs: scores, labels, the class names and the cut-offs.
# Output: {"substrates": {name: {...}}, "macro_f1": ...}.
def score_split(scores: np.ndarray, labels: np.ndarray, classes: Sequence[str], cut_offs: Sequence[float]) -> dict[str, Any]:

    # One row per substrate, then the plain average of their F1, the number every
    # feasibility run was compared on.
    substrates = {}
    for c, name in enumerate(classes):
        f1, precision, recall = f1_at(scores[:, c], labels[:, c], cut_offs[c])
        substrates[name] = {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "cut_off": round(float(cut_offs[c]), 4),
            "frames_present": int((labels[:, c] == 1).sum()),
        }
    macro = float(np.mean([substrates[n]["f1"] for n in classes])) if classes else 0.0
    return {"substrates": substrates, "macro_f1": round(macro, 4), "frames": int(scores.shape[0])}


# positive_weights()
# How much each substrate's positives count in the loss, by how rare they are.
# Inputs: the training labels, shaped (frames, substrates).
# Output: one weight per substrate: negatives over positives, 1 where there are none.
# Use this so Rock, present on most frames, does not drown Boulder.
def positive_weights(labels: np.ndarray) -> list[float]:

    # A substrate absent from the training split gets weight 1 rather than a division by 0.
    positives = labels.sum(axis=0)
    return [float((len(labels) - p) / p) if p else 1.0 for p in positives]


# layer_groups()
# The model's parameters in groups with their own learning rate, slower deeper down.
# Inputs: the model, the top learning rate and the decay per block.
# Output: parameter groups for an optimiser.
# Use this for stage 2. A model with no `blocks` (not a ViT) is one group at the top rate.
def layer_groups(model: Any, lr: float, decay: float) -> list[dict[str, Any]]:

    # Without transformer blocks there are no layers to slow down.
    blocks = getattr(model, "blocks", None)
    if blocks is None or decay == 1.0:
        return [{"params": list(model.parameters()), "lr": lr}]

    # Depth 0 is the patch embedding, 1..n the blocks, n+1 everything above them.
    top = len(blocks) + 1

    # Which depth a parameter belongs to, by its name.
    def depth(name: str) -> int:

        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        if name.startswith(("patch_embed", "cls_token", "pos_embed", "reg_token")):
            return 0
        return top

    # Gather by depth, then give each depth its rate.
    grouped: dict[int, list] = {}
    for name, parameter in model.named_parameters():
        grouped.setdefault(depth(name), []).append(parameter)
    return [{"params": params, "lr": lr * decay ** (top - d)} for d, params in sorted(grouped.items())]


# Hooks
# What the recipe tells its caller as it goes, and asks of it. The engine connects these
# to the job context; a test passes plain functions.
class Hooks(NamedTuple):

    # A pass finished: (pass number counted over both stages, stage name, metrics).
    on_epoch: Callable[[int, str, Mapping[str, Any]], None]

    # A line for the operator.
    log: Callable[[str], None]

    # True when the job should stop; checked every batch.
    should_stop: Callable[[], bool]

    # Blocks while the machine's operator has the worker paused; checked every batch.
    wait_while_paused: Callable[[], Any]


# StoppedEarly
# Raised out of the recipe when a stop arrived; the engine reports a cancellation.
class StoppedEarly(RuntimeError):
    pass


# predict()
# Every frame's scores from a loader, as probabilities.
# Inputs: the model, a loader of (images, labels), the torch device.
# Output: (scores, labels) as numpy arrays shaped (frames, substrates).
def predict(model: Any, loader: Any, device: Any) -> tuple[np.ndarray, np.ndarray]:

    # Imported here so this module imports without torch, for the scoring tests.
    import torch

    model.eval()
    outputs, labels = [], []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for images, targets in loader:
            outputs.append(torch.sigmoid(model(images.to(device))).float().cpu())
            labels.append(targets)
    if not outputs:
        return np.zeros((0, 0)), np.zeros((0, 0))
    return torch.cat(outputs).numpy(), torch.cat(labels).numpy()


# train_habitat()
# Runs the two-stage recipe and returns the best model's weights and cut-offs.
# Inputs: the model and its head, train and tuning loaders, the training labels (for the
# loss weights), the recipe, the device, the class names and the hooks.
# Output: {"state": weights, "cut_offs": [...], "tuning": scores, "kept": stage, "history": [...]}.
# Use this from the engine; the model it is given is left holding the best weights.
def train_habitat(
    model: Any,
    head: Any,
    train_loader: Any,
    val_loader: Any,
    train_labels: np.ndarray,
    recipe: HabitatRecipe,
    device: Any,
    classes: Sequence[str],
    hooks: Hooks,
) -> dict[str, Any]:

    # torch is needed from here on.
    import torch
    import torch.nn as nn

    torch.manual_seed(recipe.seed)
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weights(train_labels), dtype=torch.float32).to(device))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    # The best seen so far, across both stages: (tuning macro F1, weights, cut-offs, stage).
    best: dict[str, Any] = {"score": -1.0, "state": None, "cut_offs": None, "stage": None, "tuning": None}

    # Every pass's numbers, for the run's record.
    history: list[dict[str, Any]] = []

    # The pass number across both stages, so MARP's epochs are numbered once.
    counted = 0

    # One stage: its parameters, rate, passes and, for stage 2, a per-batch schedule.
    def stage(name: str, groups: list[dict[str, Any]], epochs: int, warmup: float) -> None:

        nonlocal counted
        optimiser = torch.optim.AdamW(groups, weight_decay=1e-4)
        steps = max(epochs * len(train_loader), 1)
        warm = warmup * len(train_loader)

        # Warm-up then cosine, stepped per batch; without warm-up, cosine per pass.
        def factor(i: int) -> float:

            if warm and i < warm:
                return (i + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * (i - warm) / max(steps - warm, 1)))

        per_batch = warm > 0
        schedule = (torch.optim.lr_scheduler.LambdaLR(optimiser, factor) if per_batch
                    else torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, max(epochs, 1)))

        for epoch in range(epochs):
            model.train()
            started, total, seen = time.monotonic(), 0.0, 0
            for images, targets in train_loader:

                # A stop or a pause is honoured between batches.
                hooks.wait_while_paused()
                if hooks.should_stop():
                    raise StoppedEarly(f"stopped during {name} pass {epoch + 1}")

                optimiser.zero_grad()
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    loss = loss_fn(model(images.to(device)), targets.to(device))
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
                if per_batch:
                    schedule.step()
                total += float(loss.item()) * len(images)
                seen += len(images)
            if not per_batch:
                schedule.step()

            # Scored on the tuning dives with cut-offs picked there, as every feasibility run was.
            scores, labels = predict(model, val_loader, device)
            cut_offs = best_cut_offs(scores, labels)
            tuning = score_split(scores, labels, classes, cut_offs)
            counted += 1
            metrics = {
                "stage": name,
                "loss": round(total / max(seen, 1), 5),
                "val_macro_f1": tuning["macro_f1"],
                **{f"val_f1/{n}": tuning["substrates"][n]["f1"] for n in classes},
                "duration_seconds": round(time.monotonic() - started, 1),
            }
            history.append(metrics)
            hooks.on_epoch(counted, name, metrics)
            hooks.log(f"{name} pass {epoch + 1}/{epochs}: loss {metrics['loss']}, tuning macro F1 {tuning['macro_f1']}")

            # Kept only if it beats everything before it, the frozen stage included.
            if tuning["macro_f1"] > best["score"]:
                best.update(score=tuning["macro_f1"], cut_offs=cut_offs, stage=name, tuning=tuning,
                            state={k: v.detach().clone() for k, v in model.state_dict().items()})

    # Stage 1: only the head learns.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True
    stage("frozen", [{"params": list(head.parameters()), "lr": recipe.head_lr}], recipe.head_epochs, 0.0)

    # Stage 2: everything learns, more slowly deeper down.
    for parameter in model.parameters():
        parameter.requires_grad = True
    stage("fine-tune", layer_groups(model, recipe.lr, recipe.layer_decay), recipe.epochs, recipe.warmup_epochs)

    # The model leaves holding the best pass's weights.
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    if best["stage"] == "frozen":
        hooks.log("fine-tuning never beat the frozen backbone on the tuning dives; kept the frozen one")
    return {"state": best["state"], "cut_offs": best["cut_offs"], "tuning": best["tuning"],
            "kept": best["stage"], "history": history}


# make_loader()
# A torch loader over one split: each frame cropped, resized and normalised.
# Inputs: the split's rows from read_split, the crop as fractions (x0, y0, x1, y1), the
# recipe, whether it is the training split, and how many loader workers to use.
# Output: a DataLoader of (images, labels).
# Use this for each split. Training frames are flipped and their colours varied, as in the
# feasibility runs; tuning and test frames are not.
def make_loader(rows: Sequence[tuple[Path, list[float]]], crop: Sequence[float], recipe: HabitatRecipe,
                train: bool, workers: int = 4) -> Any:

    # torch, torchvision and Pillow are needed from here on.
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    # The same preparation as the feasibility training: ImageNet normalisation, which is
    # what DINOv2 was trained with.
    steps = [transforms.Resize((recipe.input_size, recipe.input_size))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.1)]
    steps += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    prepare = transforms.Compose(steps)

    # Frames
    # The split's frames, read one at a time.
    class Frames(Dataset):

        # The rows and the crop, fixed for the split.
        def __init__(self) -> None:

            self.rows = list(rows)

        def __len__(self) -> int:

            return len(self.rows)

        # One frame: the habitat crop, prepared, and its labels.
        def __getitem__(self, index: int):

            path, labels = self.rows[index]
            image = Image.open(path)
            # A full-HD frame is decoded at half size when the crop is still larger than the
            # model's input after it, which keeps the loaders ahead of the GPU.
            if image.width > 1000 and image.height // 2 * (crop[3] - crop[1]) >= recipe.input_size:
                image.draft("RGB", (image.width // 2, image.height // 2))
            image = image.convert("RGB")
            width, height = image.size
            box = (int(width * crop[0]), int(height * crop[1]), int(width * crop[2]), int(height * crop[3]))
            return prepare(image.crop(box)), torch.tensor(labels, dtype=torch.float32)

    return DataLoader(Frames(), batch_size=recipe.batch, shuffle=train, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())
