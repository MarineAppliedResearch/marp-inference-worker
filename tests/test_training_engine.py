# test_training_engine.py
# Created: 2026-10-20
# Author: Isaac Travers
#
# Training engine tests for the MARP Inference Worker.
# Everything here runs with no GPU and without importing Ultralytics: it covers
# the decisions the engine makes before and around a run -- what it refuses, what
# settings it will actually pass, and what it reports as ignored.
#
# The run itself is not here and cannot be. Training needs a CUDA device and a
# real dataset, so what proves it is a run on the hardware, recorded in
# .marp/verification.md. Pretending otherwise with a mocked trainer would assert
# that our own fake was called.

# pytest provides raises and monkeypatch.
import pytest

# sys and types build the stand-in `ultralytics` module one test installs, so a
# whole run() can be exercised without a GPU or real weights.
import sys
import types

# Path builds the fixture dataset and weights.
from pathlib import Path

# The engine under test, and the refusal it raises.
from marp_inference_worker.engines.base_engine import JobUnrunnable
from marp_inference_worker.engines.training_engine import TrainingEngine
from marp_inference_worker.engines import training_engine as training_module


# a_runnable_spec()
# A training spec whose dataset and weights exist on disk.
# Inputs: tmp_path and any params to override.
# Output: the spec mapping.
def a_runnable_spec(tmp_path: Path, **params) -> dict:

    data_yaml = tmp_path / "ds" / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("names:\n- Lingcod\nnc: 1\ntrain: t\nval: v\n", encoding="utf-8")

    weights = tmp_path / "yolo11m.pt"
    weights.write_bytes(b"not-really-weights")

    base = {"data_yaml": str(data_yaml), "model_path": str(weights)}
    base.update(params)
    return {"engine": "marp_training", "params": base}


# test_the_engine_describes_itself_as_training()
# Verifies /status can tell this engine apart from an inference one.
# Inputs: none.
# Output: pytest pass/fail result.
def test_the_engine_describes_itself_as_training() -> None:

    described = TrainingEngine().describe()

    assert described["name"] == "marp_training"
    assert described["kind"] == "training"
    assert described["requires_gpu"] is True


# test_training_is_refused_on_a_machine_with_no_gpu()
# Verifies a CPU machine fails immediately and says why.
# Inputs: monkeypatch and tmp_path.
# Output: pytest pass/fail result.
#
# Training on a CPU is not a slower version of the job, it is a different one: a
# run that takes an hour takes days, and the volunteer's machine is unusable
# throughout. Refusing at preflight is what turns that into a job MARP can give
# to another machine instead (R14).
def test_training_is_refused_on_a_machine_with_no_gpu(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "cuda_device_count", lambda: 0)

    with pytest.raises(JobUnrunnable) as refused:
        TrainingEngine().preflight(a_runnable_spec(tmp_path))

    assert "GPU" in str(refused.value)


# test_training_is_refused_when_the_dataset_never_arrived()
# Verifies a missing dataset is named here rather than inside Ultralytics.
# Inputs: monkeypatch and tmp_path.
# Output: pytest pass/fail result.
def test_training_is_refused_when_the_dataset_never_arrived(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "cuda_device_count", lambda: 1)

    spec = a_runnable_spec(tmp_path)
    spec["params"]["data_yaml"] = str(tmp_path / "gone" / "data.yaml")

    with pytest.raises(JobUnrunnable) as refused:
        TrainingEngine().preflight(spec)

    assert "data.yaml" in str(refused.value)


# test_a_job_with_no_dataset_at_all_is_refused()
# Verifies an inference-shaped spec cannot reach the training engine.
# Inputs: monkeypatch and tmp_path.
# Output: pytest pass/fail result.
def test_a_job_with_no_dataset_at_all_is_refused(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "cuda_device_count", lambda: 1)

    spec = a_runnable_spec(tmp_path)
    del spec["params"]["data_yaml"]

    with pytest.raises(JobUnrunnable):
        TrainingEngine().preflight(spec)


# test_a_runnable_job_passes_preflight()
# The positive control.
# Inputs: monkeypatch and tmp_path.
# Output: pytest pass/fail result.
#
# Without this the refusal tests above would all keep passing if preflight
# started refusing everything, which is also what a broken preflight looks like.
def test_a_runnable_job_passes_preflight(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "cuda_device_count", lambda: 1)

    TrainingEngine().preflight(a_runnable_spec(tmp_path))


# test_only_known_settings_reach_ultralytics()
# Verifies the allow-list, and that nothing is dropped in silence.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# `model.train()` takes well over a hundred arguments and several of them change
# what the result means -- `project` and `name` decide where the weights land,
# `resume` reinterprets the whole call, `device` is the worker's decision rather
# than the submitter's. A job that asks for one of those gets told it was
# ignored rather than quietly having its way.
def test_only_known_settings_reach_ultralytics(tmp_path) -> None:

    spec = a_runnable_spec(
        tmp_path,
        epochs=7,
        imgsz=512,
        lr0=0.001,
        project="/tmp/somewhere-else",
        resume=True,
        slot_index=1,
    )

    settings, ignored = TrainingEngine()._training_settings(spec["params"])

    assert settings["epochs"] == 7
    assert settings["imgsz"] == 512
    assert settings["lr0"] == pytest.approx(0.001)

    # Refused, and said so.
    assert "project" in ignored
    assert "resume" in ignored

    # The worker's own keys are neither settings nor ignored submitter settings.
    for internal in ("data_yaml", "model_path", "slot_index"):
        assert internal not in settings
        assert internal not in ignored


# test_a_run_has_defaults_that_are_ours()
# Verifies what runs is decided here rather than by the installed library.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
def test_a_run_has_defaults_that_are_ours(tmp_path) -> None:

    settings, _ = TrainingEngine()._training_settings(a_runnable_spec(tmp_path)["params"])

    assert settings["epochs"] == 100
    assert settings["imgsz"] == 640
    assert settings["patience"] == 50


# test_a_setting_of_the_wrong_type_is_reported_not_guessed()
# Verifies a bad value is refused rather than coerced into something plausible.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
def test_a_setting_of_the_wrong_type_is_reported_not_guessed(tmp_path) -> None:

    spec = a_runnable_spec(tmp_path, epochs="lots")

    settings, ignored = TrainingEngine()._training_settings(spec["params"])

    assert ignored["epochs"] == "lots"
    assert settings["epochs"] == 100


# FakeTrainer
# The handful of attributes the epoch callback reads off Ultralytics' trainer.
# A fake rather than the real thing because the callback's job is bookkeeping,
# and what it has to get right is the sequence of calls -- which a real trainer
# would take twenty minutes and a GPU to produce once.
class FakeTrainer:

    def __init__(self, epoch: int, metrics: dict, fitness=None, best_fitness=None) -> None:

        # Zero-based, as Ultralytics counts.
        self.epoch = epoch
        self.metrics = metrics
        self.fitness = fitness
        self.best_fitness = best_fitness


# RecordingContext
# Enough of a JobContext for the callback, keeping what it was told.
class RecordingContext:

    def __init__(self) -> None:

        self.progress: list[tuple] = []
        self.metrics: list[tuple] = []
        self.logs: list[str] = []

    def report_progress(self, done, total, unit, phase=None, tracks=None) -> None:
        self.progress.append((done, total, unit, phase))

    def report_metrics(self, step, phase, metrics) -> None:
        self.metrics.append((step, phase, dict(metrics)))

    def log(self, message, level="info") -> None:
        self.logs.append(message)

    def should_stop(self) -> bool:
        return False

    def wait_while_paused(self, while_waiting=None) -> bool:
        return False


# test_the_final_validation_is_not_counted_as_an_epoch()
# Verifies a two-epoch run reports two epochs.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# **This is a defect the first real run found.** Ultralytics calls
# `on_fit_epoch_end` once per epoch and then once more, after it validates the
# best weights, with `trainer.epoch` already advanced. The engine counted that
# as another epoch, so a two-epoch run reported "3/2 epochs", and the coordinator
# would have written an `epochs` row for an epoch that never ran and a
# `total_epochs` of 3.
#
# The sequence below is exactly what the run produced: epoch indices 0 and 1 for
# the two real epochs, then a third call at index 2 carrying the final
# validation's metrics.
def test_the_final_validation_is_not_counted_as_an_epoch(tmp_path) -> None:

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=2, run_root=tmp_path)

    state.on_epoch_end(FakeTrainer(0, {"metrics/precision(B)": 0.80}))
    state.on_epoch_end(FakeTrainer(1, {"metrics/precision(B)": 0.88}))

    # The extra call, after training has finished.
    state.on_epoch_end(FakeTrainer(2, {"metrics/precision(B)": 0.91, "metrics/mAP50(B)": 0.92}))

    assert state.epochs_done == 2
    assert [step for step, _, _ in ctx.metrics] == [1, 2]

    # The final figures are still the run's result, just not another epoch.
    assert state.last_metrics["metrics/mAP50(B)"] == 0.92


# test_a_repeated_epoch_is_not_counted_twice()
# Verifies the same epoch arriving again does not advance the count.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# Some Ultralytics versions fire the final validation with the *last* index
# rather than one past it, which is the same defect wearing a different number.
def test_a_repeated_epoch_is_not_counted_twice(tmp_path) -> None:

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=5, run_root=tmp_path)

    state.on_epoch_end(FakeTrainer(0, {"metrics/precision(B)": 0.80}))
    state.on_epoch_end(FakeTrainer(0, {"metrics/precision(B)": 0.81}))

    assert state.epochs_done == 1
    assert len(ctx.metrics) == 1


# test_every_epoch_of_a_full_run_is_reported()
# The positive control.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# Without it the two tests above would keep passing if the callback stopped
# counting anything at all, which is also what a broken callback looks like.
def test_every_epoch_of_a_full_run_is_reported(tmp_path) -> None:

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=4, run_root=tmp_path)

    for index in range(4):
        state.on_epoch_end(FakeTrainer(index, {"metrics/precision(B)": 0.5 + index / 10}))

    assert state.epochs_done == 4
    assert [step for step, _, _ in ctx.metrics] == [1, 2, 3, 4]


# test_the_loss_weights_reach_ultralytics()
# Verifies box, cls and dfl are honoured rather than reported ignored.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# These are the knobs for the trade MARP actually cares about -- recall against
# false positives, because a biologist rejecting a bad tile in a review mosaic
# is cheap and a missed animal costs re-reviewing video. The dashboard has
# offered them on its detection form since the training page was wired; the
# worker did not accept them, so a run that set them trained at the defaults and
# said nothing a reader of the record would notice.
def test_the_loss_weights_reach_ultralytics(tmp_path) -> None:

    spec = a_runnable_spec(tmp_path, box=9.0, cls=1.25, dfl=2.0)

    settings, ignored = TrainingEngine()._training_settings(spec["params"])

    assert settings["box"] == pytest.approx(9.0)
    assert settings["cls"] == pytest.approx(1.25)
    assert settings["dfl"] == pytest.approx(2.0)

    for weight in ("box", "cls", "dfl"):
        assert weight not in ignored


# test_mask_ratio_is_an_integer_divisor()
# Verifies mask_ratio is cast as the int Ultralytics lists it as.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# It is a downsample divisor, used as `h // mask_ratio`. A float would reach
# Ultralytics and fail somewhere inside the segmentation dataloader rather than
# here, so the cast is what turns a wrong value into a settings report.
def test_mask_ratio_is_an_integer_divisor(tmp_path) -> None:

    settings, ignored = TrainingEngine()._training_settings(
        a_runnable_spec(tmp_path, mask_ratio="8")["params"]
    )

    assert settings["mask_ratio"] == 8
    assert isinstance(settings["mask_ratio"], int)
    assert "mask_ratio" not in ignored


# test_the_hazardous_arguments_are_still_refused()
# Verifies widening the allow-list did not widen it past its reason for being.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# The allow-list exists because a handful of `model.train()` arguments change
# what the result means rather than how well it trains. Adding four loss weights
# is a different kind of change and must not drag those in behind it; this is
# the test that says so out loud, where the one above only says the four arrived.
def test_the_hazardous_arguments_are_still_refused(tmp_path) -> None:

    spec = a_runnable_spec(
        tmp_path, project="/tmp/elsewhere", name="theirs", resume=True, device="cuda:3"
    )

    settings, ignored = TrainingEngine()._training_settings(spec["params"])

    for hazard in ("project", "name", "resume"):
        assert hazard in ignored
        assert hazard not in settings

    # `device` is in the engine's own key list rather than the allow-list, so it
    # is neither a setting nor an ignored one. run() reads it separately, hands
    # it to resolve_device() and passes the answer to train() itself -- leaving
    # it here would pass `device` twice and let the request through unresolved.
    assert "device" not in settings
    assert "device" not in ignored


# test_the_settings_report_says_which_device_and_which_build()
# Verifies the two facts only the worker can state reach MARP's record.
# Inputs: none.
# Output: pytest pass/fail result.
#
# `training_runs.compute_device` and `training_runs.train_script_commit` are
# columns nothing filled. Only the worker knows either: it resolves the device
# itself, deliberately not from the submitter's request, and it is the thing
# that knows which build it is. Both are reported as `source: "worker"`, which
# is what lets a reader tell a resolved device from a requested one.
def test_the_settings_report_says_which_device_and_which_build() -> None:

    provenance = training_module._run_provenance("cuda:1")

    assert provenance["compute_device"]["value"] == "cuda:1"
    assert provenance["compute_device"]["source"] == "worker"

    commit = provenance["train_script_commit"]
    assert commit["source"] == "worker"
    assert isinstance(commit["value"], str) and commit["value"]


# StubTrainedModel
# A YOLO stand-in that records the arguments train() was called with.
# Real Ultralytics is not used here: it would want a CUDA device, real weights
# and twenty minutes to answer a question about argument plumbing.
class StubTrainedModel:

    def __init__(self, weights_path) -> None:

        self.weights_path = weights_path
        self.train_kwargs: dict = {}
        self.callbacks: dict = {}

    def add_callback(self, event, handler) -> None:
        self.callbacks[event] = handler

    def train(self, **kwargs) -> None:
        self.train_kwargs = kwargs


# RunContext
# Enough of a JobContext for a whole run(), keeping every report it was given.
class RunContext(RecordingContext):

    def __init__(self, checkpoint_dir) -> None:

        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.settings_reports: list[tuple] = []

    def report_settings(self, engine, settings, ignored) -> None:
        self.settings_reports.append((engine, dict(settings), dict(ignored)))

    def publish_artifact(self, path, kind) -> str:
        return "0" * 64


# a_stubbed_ultralytics()
# Installs a fake `ultralytics` module and returns the model it will hand out.
# Inputs: pytest's monkeypatch.
# Output: the StubTrainedModel run() will train.
#
# run() imports ultralytics inside the function, so replacing the entry in
# sys.modules is enough and nothing has to be injected through the engine.
def a_stubbed_ultralytics(monkeypatch) -> StubTrainedModel:

    model = StubTrainedModel(None)

    module = types.ModuleType("ultralytics")

    # YOLO(path) in run() -- records which weights it was pointed at.
    def _yolo(weights):
        model.weights_path = weights
        return model

    module.YOLO = _yolo
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return model


# test_a_run_reports_its_provenance_and_passes_its_loss_weights()
# Verifies the wiring, not just the two helpers either side of it.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# The two tests above can both pass while run() never calls either function.
# This one goes through run() itself with a stubbed Ultralytics, so it can see
# the settings report MARP will actually receive and the arguments the trainer
# will actually get. It is the tier the change has to be observed at; the
# helpers' own tests only say what the helpers would produce if asked.
def test_a_run_reports_its_provenance_and_passes_its_loss_weights(tmp_path, monkeypatch) -> None:

    model = a_stubbed_ultralytics(monkeypatch)
    spec = a_runnable_spec(tmp_path, epochs=1, box=9.0, cls=1.25, dfl=2.0, mask_ratio=8)

    ctx = RunContext(tmp_path / "attempt")
    ctx.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary = TrainingEngine().run(ctx, spec)

    # What MARP is told about how the run ran.
    assert len(ctx.settings_reports) == 1
    _engine, reported, _ignored = ctx.settings_reports[0]

    assert reported["compute_device"]["source"] == "worker"
    assert reported["compute_device"]["value"] == summary["device"]
    assert reported["train_script_commit"]["source"] == "worker"
    assert reported["train_script_commit"]["value"] == summary["train_script_commit"]

    # The loss weights are the submitter's, and are reported as such.
    assert reported["box"] == {"value": 9.0, "type": "float", "source": "job"}

    # And they reached the trainer rather than only the record.
    assert model.train_kwargs["box"] == pytest.approx(9.0)
    assert model.train_kwargs["cls"] == pytest.approx(1.25)
    assert model.train_kwargs["dfl"] == pytest.approx(2.0)
    assert model.train_kwargs["mask_ratio"] == 8

    # The worker's decisions are still the worker's: a device argument is
    # passed by run() itself, never merged in from the job's settings.
    assert model.train_kwargs["device"] == summary["device"]


# FakeParamGroup/FakeOptimizer
# The two attributes the resolved-settings reader touches on a torch optimizer.
# Named after the optimizer class, because the class name IS the reported value.
class FakeAdamW:

    def __init__(self, groups) -> None:

        self.param_groups = groups


class FakeSGD:

    def __init__(self, groups) -> None:

        self.param_groups = groups


# test_what_the_run_resolved_is_read_off_the_optimizer()
# Verifies the reported optimizer and lr0 come from the built optimizer.
# Inputs: none.
# Output: pytest pass/fail result.
#
# `build_optimizer` resolves `optimizer='auto'` into a name, a learning rate and
# a momentum and keeps all three in LOCAL variables -- `trainer.args.optimizer`
# still reads "auto" afterwards and `trainer.args.lr0` still reads whatever was
# configured. Reading `args` back would record the request while claiming it was
# the run. This asserts the reader goes to the optimizer instead.
def test_what_the_run_resolved_is_read_off_the_optimizer() -> None:

    trainer = FakeTrainer(0, {})
    trainer.args = types.SimpleNamespace(optimizer="auto", lr0=0.01, momentum=0.937)
    trainer.optimizer = FakeAdamW([{"initial_lr": 0.002, "betas": (0.9, 0.999)}])

    resolved = training_module._resolved_training_settings(trainer)

    # The request said "auto" and 0.01; neither is what ran.
    assert resolved["optimizer"]["value"] == "FakeAdamW"
    assert resolved["lr0"]["value"] == pytest.approx(0.002)
    assert resolved["momentum"]["value"] == pytest.approx(0.9)

    for entry in resolved.values():
        assert entry["source"] == "resolved"


# test_momentum_is_found_wherever_the_optimizer_keeps_it()
# Verifies the SGD spelling of momentum is read as well as the Adam one.
# Inputs: none.
# Output: pytest pass/fail result.
#
# SGD-family optimizers have a `momentum` key; Adam-family ones keep it as the
# first of `betas`. A reader that knew only one of the two would silently drop
# the value for half of the optimizers Ultralytics can pick.
def test_momentum_is_found_wherever_the_optimizer_keeps_it() -> None:

    trainer = FakeTrainer(0, {})
    trainer.optimizer = FakeSGD([{"initial_lr": 0.01, "momentum": 0.937}])

    resolved = training_module._resolved_training_settings(trainer)

    assert resolved["momentum"]["value"] == pytest.approx(0.937)
    assert resolved["optimizer"]["value"] == "FakeSGD"


# test_nothing_is_invented_when_it_cannot_be_read()
# Verifies an unreadable value is left out rather than filled from a default.
# Inputs: none.
# Output: pytest pass/fail result.
#
# This is the point of the whole exercise. The dashboard deliberately does not
# send Auto, because a guess written into a scientific record is worse than a
# null. A reader that fell back to a default would reintroduce exactly that, and
# would do it invisibly.
def test_nothing_is_invented_when_it_cannot_be_read() -> None:

    trainer = FakeTrainer(0, {})
    trainer.args = types.SimpleNamespace(optimizer="auto", lr0=0.01, momentum=0.937)

    # No optimizer at all: nothing is knowable, so nothing is said.
    trainer.optimizer = None
    assert training_module._resolved_training_settings(trainer) == {}

    # An optimizer whose groups carry no initial_lr: the name is knowable and
    # is reported; the learning rate is not and is omitted, NOT taken from
    # trainer.args.lr0, which is sitting right there reading 0.01.
    trainer.optimizer = FakeSGD([{}])
    resolved = training_module._resolved_training_settings(trainer)

    assert resolved["optimizer"]["value"] == "FakeSGD"
    assert "lr0" not in resolved
    assert "momentum" not in resolved


# test_the_resolved_report_goes_out_before_the_first_epoch()
# Verifies the second settings report fires from on_train_start.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# Timing is the requirement, not just content: a run that dies at epoch 3 still
# has to answer "what was it trying?". Reporting at the end would lose exactly
# the runs somebody needs to diagnose.
def test_the_resolved_report_goes_out_before_the_first_epoch(tmp_path) -> None:

    ctx = RunContext(tmp_path)
    state = training_module._RunState(
        ctx=ctx, total_epochs=4, run_root=tmp_path, engine_name="marp_training"
    )

    trainer = FakeTrainer(0, {})
    trainer.optimizer = FakeSGD([{"initial_lr": 0.01, "momentum": 0.937}])

    state.on_train_start(trainer)

    assert state.epochs_done == 0
    assert len(ctx.settings_reports) == 1

    engine, reported, ignored = ctx.settings_reports[0]
    assert engine == "marp_training"
    assert reported["lr0"]["value"] == pytest.approx(0.01)

    # The resolved report carries no ignored params; the first report owns those.
    assert ignored == {}


# test_a_broken_resolved_report_does_not_kill_the_run()
# Verifies bookkeeping never costs a volunteer's GPU run.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
def test_a_broken_resolved_report_does_not_kill_the_run(tmp_path) -> None:

    class Exploding(RunContext):
        def report_settings(self, engine, settings, ignored) -> None:
            raise RuntimeError("coordinator went away")

    ctx = Exploding(tmp_path)
    state = training_module._RunState(ctx=ctx, total_epochs=1, run_root=tmp_path)

    trainer = FakeTrainer(0, {})
    trainer.optimizer = FakeSGD([{"initial_lr": 0.01}])

    # Must not raise.
    state.on_train_start(trainer)

    assert any("could not report" in line for line in ctx.logs)


# test_the_reader_works_against_a_real_ultralytics_optimizer()
# The control: everything above uses a fake optimizer shaped the way this file
# believes torch shapes them. This one builds a real one.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Without it, every test above would keep passing if Ultralytics changed where
# it keeps the resolved learning rate -- the fakes would still match the old
# shape and the engine would silently report nothing. It uses Ultralytics' own
# build_optimizer on a toy torch model: no dataset, no GPU, no training.
def test_the_reader_works_against_a_real_ultralytics_optimizer() -> None:

    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")

    from torch import nn, optim
    from ultralytics.cfg import get_cfg
    from ultralytics.engine.trainer import BaseTrainer

    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8))

    # optimizer='auto' is the dashboard's default, and the case that matters.
    args = get_cfg(overrides={})
    owner = types.SimpleNamespace(args=args, data={"nc": 1})
    optimizer = BaseTrainer.build_optimizer(
        owner, model=model, name=args.optimizer, lr=args.lr0,
        momentum=args.momentum, decay=0.0005, iterations=500,
    )
    # The scheduler is what stamps initial_lr onto the param groups, so the
    # engine only ever sees an optimizer that has been through this.
    optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _x: 1.0)

    trainer = FakeTrainer(0, {})
    trainer.args = args
    trainer.optimizer = optimizer

    resolved = training_module._resolved_training_settings(trainer)

    # Ultralytics left the request untouched: this is the lie we must not tell.
    assert args.optimizer == "auto"

    # And the reader found the real thing anyway.
    assert resolved["optimizer"]["value"] != "auto"
    assert resolved["optimizer"]["value"] == type(optimizer).__name__
    assert resolved["lr0"]["value"] > 0
    assert resolved["lr0"]["value"] != pytest.approx(args.lr0)
