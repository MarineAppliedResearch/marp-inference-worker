# test_habitat_training_engine.py
# Created: 2026-11-10
# Author: Isaac Travers
#
# Tests for the habitat training engine (MarineAppliedResearch/MARP_API#287), on a CPU.
# The engine is run end to end with the stand-in network from test_habitat_training.py in
# place of DINOv2, against a fake job context that keeps what it is told and what it is
# handed. What these prove is the engine: what it refuses, what it reports, and the three
# files MARP registers a habitat model from. Building DINOv2 from its real weights on a GPU
# is the recorded GPU run's job.

# json reads what was published.
import json

# Path types the job directory.
from pathlib import Path

# pytest for refusals; torch for the stand-in.
import pytest
import torch

# The engine under test, its report shape, and the contract's refusal.
from marp_inference_worker.engines.base_engine import JobContext, JobUnrunnable
from marp_inference_worker.engines.habitat_training_engine import (
    HabitatTrainingEngine,
    final_metrics_report,
    recipe_from_params,
)

# The made-up archive and stand-in network.
from test_habitat_training import CLASSES, Tiny, write_archive


# Context
# A JobContext that records everything and publishes into a dict.
class Context:

    # A job directory, and empty records.
    def __init__(self, workdir: Path, stop_after_checks: int | None = None) -> None:

        self._workdir = workdir
        self.progress, self.metrics, self.logs, self.settings = [], [], [], []
        self.published: dict[str, Path] = {}
        self._stop_after = stop_after_checks
        self._checks = 0

    @property
    def params(self):

        return {}

    @property
    def checkpoint_dir(self) -> Path:

        return self._workdir

    @property
    def resume_from(self):

        return None

    def log(self, message, level="info") -> None:

        self.logs.append(message)

    def report_progress(self, done, total, unit, phase=None) -> None:

        self.progress.append((done, total, unit, phase))

    def report_metrics(self, step, phase, metrics) -> None:

        self.metrics.append((step, phase, dict(metrics)))

    def report_settings(self, engine, settings, ignored) -> None:

        self.settings.append((engine, dict(settings), dict(ignored)))

    def publish_artifact(self, path, kind) -> str:

        self.published[kind] = Path(path)
        return f"sha-{kind}"

    def should_stop(self) -> bool:

        self._checks += 1
        return self._stop_after is not None and self._checks > self._stop_after

    def wait_while_paused(self, while_waiting=None) -> bool:

        return False


# Engine
# The real engine with the stand-in network in place of DINOv2.
class Engine(HabitatTrainingEngine):

    # A seeded stand-in, so the run is the same every time.
    def build_model(self, classes, input_size, weights, workdir):

        torch.manual_seed(0)
        model = Tiny()
        return model, model.head


# spec_for()
# A job spec over a made-up archive, small enough for a CPU.
def spec_for(archive: Path, weights: Path, **params) -> dict:

    return {"engine": "marp_habitat_training", "params": {
        "dataset_path": str(archive), "model_path": str(weights), "device": "cpu", "workers": 0,
        "input_size": 16, "batch": 8, "head_epochs": 2, "head_lr": 5e-2, "epochs": 4, "lr": 1e-2, "seed": 3,
        **params}}


# job()
# A made-up archive, a starting-weights file and a fresh context.
def job(tmp_path, **params):

    (tmp_path / "archive").mkdir()
    archive = write_archive(tmp_path / "archive", per_split=32)
    weights = tmp_path / "start.pt"
    weights.write_bytes(b"not used by the stand-in")
    workdir = tmp_path / "job"
    workdir.mkdir()
    return spec_for(archive, weights, **params), workdir


# test_the_context_satisfies_the_contract()
# The fake is a JobContext, so the engine is tested against the real interface.
def test_the_context_satisfies_the_contract(tmp_path):

    assert isinstance(Context(tmp_path), JobContext)


# test_refuses_a_job_it_cannot_run()
# No dataset, no splits, or no starting weights: refused before anything starts.
def test_refuses_a_job_it_cannot_run(tmp_path):

    spec, _ = job(tmp_path)
    engine = HabitatTrainingEngine()
    with pytest.raises(JobUnrunnable, match="classes.json"):
        engine.preflight({"params": {**spec["params"], "dataset_path": str(tmp_path)}})
    with pytest.raises(JobUnrunnable, match="starting weights"):
        engine.preflight({"params": {**spec["params"], "model_path": None}})
    (tmp_path / "archive" / "test" / "labels.csv").unlink()
    with pytest.raises(JobUnrunnable, match="no test split"):
        engine.preflight(spec)


# test_settings_are_the_recipe_unless_overridden_and_unknown_ones_are_reported()
# The settled recipe, a job's overrides converted, and anything unknown reported back.
def test_settings_are_the_recipe_unless_overridden_and_unknown_ones_are_reported():

    recipe, ignored = recipe_from_params({"epochs": "4", "lr": 1e-4, "mosaic": 1.0, "dataset_path": "x"})
    assert (recipe.epochs, recipe.lr, recipe.input_size, recipe.layer_decay) == (4, 1e-4, 518, 0.75)
    assert ignored == {"mosaic": 1.0}


# test_a_run_publishes_the_weights_the_rules_and_the_scores()
# A whole run: every pass reported, then the three files MARP registers a model from.
def test_a_run_publishes_the_weights_the_rules_and_the_scores(tmp_path):

    spec, workdir = job(tmp_path)
    ctx = Context(workdir)
    engine = Engine()
    engine.preflight(spec)
    summary = engine.run(ctx, spec)

    # Six passes reported as training epochs, numbered once.
    assert [step for step, phase, _ in ctx.metrics] == [1, 2, 3, 4, 5, 6]
    assert all(phase == "training" for _, phase, _ in ctx.metrics)
    assert ctx.settings[0][0] == "marp_habitat_training"

    # The weights, loadable into the model that made them.
    assert set(ctx.published) == {"weights", "decision_rules", "final_metrics"}
    state = torch.load(ctx.published["weights"], weights_only=True)
    Tiny().load_state_dict(state)

    # The decision rules MARP registers the model with: substrates in output order, a
    # cut-off each, the habitat crop and the input size.
    rules = json.loads(ctx.published["decision_rules"].read_text(encoding="utf-8"))
    assert rules["substrates"] == CLASSES
    assert set(rules["cut_offs"]) == set(CLASSES) and all(0 < v < 1 for v in rules["cut_offs"].values())
    assert rules["crop"] == {"x0": 0.25, "y0": 0.5, "x1": 0.75, "y1": 1.0}
    assert rules["input_size"] == 16

    # The scores on the test split, in the shape MARP stores final metrics in.
    report = json.loads(ctx.published["final_metrics"].read_text(encoding="utf-8"))
    assert report["split"] == "test" and report["task"] == "habitat"
    family = report["metrics"][0]
    assert family["family"] == "multilabel"
    assert [row["class"] for row in family["per_class"]] == CLASSES
    assert family["overall"]["f1"] == summary["test_macro_f1"]


# test_a_stop_publishes_nothing()
# A stop part way through is a cancellation with nothing handed over.
def test_a_stop_publishes_nothing(tmp_path):

    spec, workdir = job(tmp_path)
    ctx = Context(workdir, stop_after_checks=3)
    summary = Engine().run(ctx, spec)
    assert summary["stopped_early"] is True
    assert ctx.published == {}


# test_final_metrics_are_one_multilabel_family()
# The report MARP reads: one multilabel family, overall and per substrate.
def test_final_metrics_are_one_multilabel_family():

    report = final_metrics_report({"macro_f1": 0.7, "frames": 10, "substrates": {
        "Rock": {"f1": 0.9, "precision": 0.8, "recall": 1.0, "frames_present": 6}}})
    assert report["metrics"] == [{"family": "multilabel", "overall": {"f1": 0.7}, "per_class": [
        {"class": "Rock", "f1": 0.9, "precision": 0.8, "recall": 1.0, "support": 6, "images": 10}]}]
