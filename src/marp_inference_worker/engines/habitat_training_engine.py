# habitat_training_engine.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Training engine for habitat substrate models (MarineAppliedResearch/MARP_API#287).
# Builds DINOv2 ViT-B/14 from the starting weights MARP sends, trains it on a habitat
# dataset archive with the recipe in training/habitat_training.py, and publishes the
# weights, the decision rules MARP registers the model with, and the end-of-run scores.
# Model construction and publishing belong here; the recipe and the dataset reader do
# not, and nothing here knows about MARP beyond the job context.

# json writes the published reports.
import json

# shutil links or copies the starting weights to a name timm reads as safetensors.
import shutil

# time measures the run.
import time

# Path locates files in the job's own directory.
from pathlib import Path

# Types for the spec and the summary.
from typing import Any, Mapping

# numpy assembles the training labels for the loss weights.
import numpy as np

# The contract every engine is invoked through.
from marp_inference_worker.engines.base_engine import BaseEngine, JobContext, JobUnrunnable

# The habitat crop, the default when a job names none.
from marp_inference_worker.media.habitat_frames import HABITAT_CROP

# Device selection is the worker's decision, not the engine's.
from marp_inference_worker.system.device import cuda_device_count, resolve_device

# The recipe, the archive reader and the scoring.
from marp_inference_worker.training.habitat_training import (
    Hooks,
    HabitatRecipe,
    StoppedEarly,
    make_loader,
    predict,
    read_classes,
    read_split,
    score_split,
    train_habitat,
)


# DINOV2_ARCHITECTURE
# The settled model (MARP_API#287): DINOv2 ViT-B/14 as timm packages it.
DINOV2_ARCHITECTURE = "vit_base_patch14_dinov2.lvd142m"

# RECIPE_PARAMS
# The recipe settings a job's params may override, and their types.
RECIPE_PARAMS = {name: type(value) for name, value in HabitatRecipe()._asdict().items()}


# recipe_from_params()
# The settled recipe with any overrides the job carries.
# Inputs: the job's params.
# Output: (recipe, overrides that were not recipe settings).
def recipe_from_params(params: Mapping[str, Any]) -> tuple[HabitatRecipe, dict[str, Any]]:

    # A recipe setting is converted to its own type; anything else is reported back as
    # ignored rather than silently dropped.
    chosen, ignored = {}, {}
    reserved = {"dataset_path", "model_path", "model_sha256", "slot_index", "slot_count", "device",
                "crop", "workers", "_job_id", "resume_checkpoint", "data_type"}
    for key, value in params.items():
        if key in RECIPE_PARAMS:
            chosen[key] = RECIPE_PARAMS[key](value)
        elif key not in reserved:
            ignored[key] = value
    return HabitatRecipe(**chosen), ignored


# crop_from_params()
# The crop a job asks for, as fractions (x0, y0, x1, y1), or the habitat crop.
def crop_from_params(params: Mapping[str, Any]) -> tuple[float, float, float, float]:

    # Given as a mapping or a list; checked to be a box inside the frame.
    raw = params.get("crop")
    if raw is None:
        return tuple(HABITAT_CROP)
    values = [raw[k] for k in ("x0", "y0", "x1", "y1")] if isinstance(raw, Mapping) else list(raw)
    x0, y0, x1, y1 = (float(v) for v in values)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise JobUnrunnable(f"the crop {raw} is not a box inside the frame")
    return (x0, y0, x1, y1)


# HabitatTrainingEngine
# Trains one habitat substrate model per job.
class HabitatTrainingEngine(BaseEngine):

    # engine_name
    # The name a job spec selects this engine by.
    @property
    def engine_name(self) -> str:

        return "marp_habitat_training"

    # describe()
    # Says what this engine is and what it needs, for /status and MARP.
    def describe(self) -> dict[str, Any]:

        return {"engine": self.engine_name, "kind": "training", "requires_gpu": True, "tasks": ["habitat"]}

    # preflight()
    # Decides whether this machine can run this job at all.
    # Inputs: the job spec mapping.
    # Output: none on success; raises JobUnrunnable with a reason otherwise.
    def preflight(self, spec: Mapping[str, Any]) -> None:

        params = dict(spec.get("params") or {})

        # A run that takes an hour on a GPU takes days on a CPU, so a CPU is refused,
        # unless the job asks for it by name (the tests do, on a tiny model).
        if cuda_device_count() <= 0 and str(params.get("device") or "").lower() != "cpu":
            raise JobUnrunnable("habitat training needs a CUDA GPU and this machine reports none")

        # The archive is unpacked by the parent before this runs.
        root = params.get("dataset_path")
        if not root or not (Path(root) / "classes.json").is_file():
            raise JobUnrunnable("the job carried no habitat dataset: classes.json is missing")
        for split in ("train", "val", "test"):
            if not (Path(root) / split / "labels.csv").is_file():
                raise JobUnrunnable(f"the habitat dataset has no {split} split")

        if not params.get("model_path"):
            raise JobUnrunnable("the job carried no starting weights")

    # build_model()
    # DINOv2 ViT-B/14 with one output per substrate, from the starting weights given.
    # Inputs: the number of substrates, the input size and the weights file.
    # Output: (model, its head).
    # A method, so a test can substitute a small network for the real one.
    def build_model(self, classes: int, input_size: int, weights: Path, workdir: Path) -> tuple[Any, Any]:

        # timm reads a file as safetensors by its name, and MARP's model cache names
        # every file .pt; so the weights are linked under a .safetensors name first.
        import timm

        named = workdir / "dinov2-start.safetensors"
        if not named.exists():
            try:
                named.hardlink_to(weights)
            except OSError:
                shutil.copyfile(weights, named)

        # Built exactly as the feasibility runs built it: patches of 14, so the input is
        # padded to a whole number of them and the position grid fitted to the input size.
        model = timm.create_model(
            DINOV2_ARCHITECTURE, pretrained=True, num_classes=classes, img_size=input_size,
            dynamic_img_pad=True, dynamic_img_size=True, pretrained_cfg_overlay={"file": str(named)},
        )
        return model, model.head

    # run()
    # Trains one model to completion, cancellation or failure.
    # Inputs: the job context and the job spec.
    # Output: a JSON-safe summary; the weights and reports go out through publish_artifact.
    def run(self, ctx: JobContext, spec: Mapping[str, Any]) -> dict[str, Any]:

        # torch is needed from here on.
        import torch

        started = time.monotonic()
        params = dict(spec.get("params") or {})
        recipe, ignored = recipe_from_params(params)
        crop = crop_from_params(params)
        device = torch.device(resolve_device(params.get("device"), int(params.get("slot_index") or 0)))
        workdir = Path(ctx.checkpoint_dir)
        workers = int(params.get("workers", 4))

        # Every setting the run uses, once, before training starts.
        ctx.report_settings(self.engine_name, {**recipe._asdict(), "crop": list(crop), "device": str(device)}, ignored)

        # The archive's substrates and splits.
        root = Path(params["dataset_path"])
        classes = read_classes(root)
        rows = {split: read_split(root, split, classes) for split in ("train", "val", "test")}
        loaders = {split: make_loader(rows[split], crop, recipe, split == "train", workers) for split in rows}
        train_labels = np.array([labels for _, labels in rows["train"]], dtype=float)
        total = recipe.head_epochs + recipe.epochs
        ctx.log(f"habitat training: {', '.join(f'{s} {len(rows[s])} frames' for s in rows)}; substrates {classes}")

        # The model, from the starting weights MARP sent.
        ctx.report_progress(0, total, "epochs", phase="loading_model")
        model, head = self.build_model(len(classes), recipe.input_size, Path(params["model_path"]), workdir)

        # The recipe, reporting each pass as a training epoch.
        ctx.report_progress(0, total, "epochs", phase="training")

        # One pass finished: its numbers, and how far through the run it is.
        def on_epoch(step: int, stage: str, metrics: Mapping[str, Any]) -> None:

            ctx.report_metrics(step, "training", metrics)
            ctx.report_progress(step, total, "epochs", phase="training")

        hooks = Hooks(on_epoch=on_epoch, log=ctx.log, should_stop=ctx.should_stop, wait_while_paused=ctx.wait_while_paused)
        try:
            result = train_habitat(model, head, loaders["train"], loaders["val"], train_labels, recipe, device, classes, hooks)
        except StoppedEarly as stopped:
            ctx.log(str(stopped))
            return {"engine": self.engine_name, "stopped_early": True, "duration_seconds": round(time.monotonic() - started, 1)}

        # The kept model on the test dives, at the cut-offs chosen on the tuning dives.
        ctx.report_progress(total, total, "epochs", phase="testing")
        scores, labels = predict(model, loaders["test"], device)
        test = score_split(scores, labels, classes, result["cut_offs"])
        ctx.log(f"habitat training: kept the {result['kept']} pass; test macro F1 {test['macro_f1']}")

        # Weights, decision rules and scores, in that order.
        ctx.report_progress(total, total, "epochs", phase="publishing")
        weights = workdir / "habitat-weights.pt"
        torch.save(result["state"], weights)
        published = {"weights": ctx.publish_artifact(weights, "weights")}
        rules = {
            "substrates": classes,
            "cut_offs": dict(zip(classes, result["cut_offs"])),
            "crop": dict(zip(("x0", "y0", "x1", "y1"), crop)),
            "input_size": recipe.input_size,
        }
        published["decision_rules"] = self._publish_json(ctx, rules, "decision_rules")
        published["final_metrics"] = self._publish_json(ctx, final_metrics_report(test), "final_metrics")

        return {
            "engine": self.engine_name,
            "architecture": DINOV2_ARCHITECTURE,
            "kept": result["kept"],
            "tuning_macro_f1": result["tuning"]["macro_f1"],
            "test_macro_f1": test["macro_f1"],
            "published": published,
            "duration_seconds": round(time.monotonic() - started, 1),
        }

    # _publish_json()
    # Writes one report into the job's directory and hands it over.
    # Inputs: the context, the report and its role.
    # Output: the sha256 the runner recorded.
    def _publish_json(self, ctx: JobContext, report: Mapping[str, Any], role: str) -> str:

        # allow_nan=False makes a stray NaN fail here rather than in MARP's parser.
        path = Path(ctx.checkpoint_dir) / f"{role}.json"
        path.write_text(json.dumps(report, indent=1, allow_nan=False), encoding="utf-8")
        return ctx.publish_artifact(path, role)


# final_metrics_report()
# The end-of-run scores in the shape MARP stores a training run's final metrics in.
# Inputs: score_split's result on the test dives.
# Output: a report with one `multilabel` family: the macro F1 overall, and each substrate's
# F1, precision and recall at its cut-off, with how many test frames had it.
def final_metrics_report(test: Mapping[str, Any]) -> dict[str, Any]:

    # Per substrate, then the plain average of their F1 as the overall figure.
    per_class = [
        {"class": name, "f1": row["f1"], "precision": row["precision"], "recall": row["recall"],
         "support": row["frames_present"], "images": test["frames"]}
        for name, row in test["substrates"].items()
    ]
    return {
        "schema": 1,
        "task": "habitat",
        "split": "test",
        "metrics": [{"family": "multilabel", "overall": {"f1": test["macro_f1"]}, "per_class": per_class}],
    }
