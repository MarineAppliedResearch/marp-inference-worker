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

# time builds a trainer that has been running for a known number of seconds,
# so the derived rate and ETA can be asserted against real arithmetic.
import time

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

    # `engine`, the key every other engine in this worker uses. This asserted
    # `name` and passed, while test_status, which iterates engines by
    # entry["engine"], was red -- a test agreeing with its own module rather
    # than with the thing that consumes it.
    assert described["engine"] == "marp_training"
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

    # `compute_device` is the CARD, not the slot -- MARP's column asks for
    # hardware. The slot travels beside it, and that is what matches the
    # summary's device string.
    assert reported["compute_device_index"]["value"] == summary["device"]
    assert reported["train_script_commit"]["source"] == "worker"
    assert reported["train_script_commit"]["value"] == summary["train_script_commit"]

    # The loss weights are the submitter's, and are reported as such.
    assert reported["box"] == {"value": 9.0, "type": "real", "source": "job"}

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


# a_trainer_at_epoch_end()
# A FakeTrainer carrying the three places Ultralytics keeps epoch numbers.
# Inputs: validator metrics, the epoch's mean losses, the epoch duration.
# Output: the fake trainer.
#
# `label_loss_items` is the trainer's own method in Ultralytics and is what
# names the losses, so the fake has to provide it rather than a flat dict --
# otherwise the test would assert against names this file invented.
def a_trainer_at_epoch_end(epoch=0, metrics=None, losses=None, epoch_time=None):

    trainer = FakeTrainer(epoch, dict(metrics or {}))
    trainer.tloss = dict(losses) if losses is not None else None
    trainer.epoch_time = epoch_time
    trainer.label_loss_items = lambda items, prefix="train": {
        f"{prefix}/{name}": round(float(value), 5) for name, value in items.items()
    }
    return trainer


# test_an_epoch_reports_its_losses_and_duration()
# Verifies the numbers that were reaching the database as NULL now arrive.
# Inputs: none.
# Output: pytest pass/fail result.
#
# The first real run from the dashboard wrote precision and recall and left
# box_loss, cls_loss, dfl_loss and duration_seconds empty. The losses never
# reach `trainer.metrics` at all -- Ultralytics keeps them in `tloss` and only
# brings the two together when it writes its own CSV -- and the duration is a
# plain attribute.
def test_an_epoch_reports_its_losses_and_duration() -> None:

    trainer = a_trainer_at_epoch_end(
        metrics={"metrics/precision(B)": 0.19, "metrics/mAP50(B)": 0.42},
        losses={"box_loss": 1.25, "cls_loss": 0.8, "dfl_loss": 0.95},
        epoch_time=27.1234,
    )

    reported = training_module._epoch_metrics(trainer)

    assert reported["train/box_loss"] == pytest.approx(1.25)
    assert reported["train/cls_loss"] == pytest.approx(0.8)
    assert reported["train/dfl_loss"] == pytest.approx(0.95)
    assert reported["duration_seconds"] == pytest.approx(27.123)

    # And the validator's numbers are still carried, unchanged.
    assert reported["metrics/precision(B)"] == pytest.approx(0.19)
    assert reported["metrics/mAP50(B)"] == pytest.approx(0.42)


# test_a_missing_source_contributes_nothing_not_zero()
# Verifies an absent number is absent rather than reported as 0.
# Inputs: none.
# Output: pytest pass/fail result.
#
# `tloss` is None through a NaN-recovery epoch, and `epoch_time` is unset
# before the first one finishes. Reporting 0.0 for either would put a wrong
# number in a scientific record and look exactly like a real measurement.
def test_a_missing_source_contributes_nothing_not_zero() -> None:

    trainer = a_trainer_at_epoch_end(metrics={"metrics/recall(B)": 0.4})

    reported = training_module._epoch_metrics(trainer)

    assert reported == {"metrics/recall(B)": 0.4}
    assert "duration_seconds" not in reported
    assert not any(name.endswith("_loss") for name in reported)


# test_a_real_loss_of_zero_is_still_reported()
# The control for the test above.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Without it, dropping every falsy value would pass -- and that would silently
# discard a genuine zero, which is a measurement and not an absence.
def test_a_real_loss_of_zero_is_still_reported() -> None:

    trainer = a_trainer_at_epoch_end(losses={"cls_loss": 0.0}, epoch_time=0.0)

    reported = training_module._epoch_metrics(trainer)

    assert reported["train/cls_loss"] == 0.0
    assert reported["duration_seconds"] == 0.0


# test_the_whole_epoch_record_reaches_the_context()
# Verifies the callback reports the assembled mapping, not just the validator's.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# `_epoch_metrics` could be perfect while `on_epoch_end` still forwarded
# `trainer.metrics`, which is exactly the bug being fixed. This watches what
# actually leaves the engine.
def test_the_whole_epoch_record_reaches_the_context(tmp_path) -> None:

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=2, run_root=tmp_path)

    state.on_epoch_end(
        a_trainer_at_epoch_end(
            epoch=0,
            metrics={"metrics/mAP50-95(B)": 0.31},
            losses={"box_loss": 1.1},
            epoch_time=27.0,
        )
    )

    assert len(ctx.metrics) == 1
    step, phase, reported = ctx.metrics[0]

    assert (step, phase) == (1, "training")
    assert reported["metrics/mAP50-95(B)"] == pytest.approx(0.31)
    assert reported["train/box_loss"] == pytest.approx(1.1)
    assert reported["duration_seconds"] == pytest.approx(27.0)


# test_the_loss_names_are_ultralytics_own()
# The control: every test above uses a fake label_loss_items shaped the way
# this file believes Ultralytics shapes it. This one uses the real method.
# Inputs: none.
# Output: pytest pass/fail result.
#
# The loss names are task-derived -- Ultralytics reads them off the criterion on
# the first batch, so segmentation has a `seg_loss` that detection does not.
# Keeping a list of names here would be a second copy that goes stale; this
# asserts we are using theirs, and would fail if the `train/` prefix or the
# labelling moved.
def test_the_loss_names_are_ultralytics_own() -> None:

    pytest.importorskip("ultralytics")
    from ultralytics.engine.trainer import BaseTrainer

    trainer = FakeTrainer(0, {})
    trainer.tloss = {"box_loss": 1.5, "cls_loss": 0.6, "dfl_loss": 0.9}
    trainer.epoch_time = 12.0
    trainer.loss_names = tuple(trainer.tloss)

    # The real method, bound to the fake -- it reads only loss_names and the
    # items it is handed.
    trainer.label_loss_items = types.MethodType(BaseTrainer.label_loss_items, trainer)

    reported = training_module._epoch_metrics(trainer)

    assert reported["train/box_loss"] == pytest.approx(1.5)
    assert reported["train/dfl_loss"] == pytest.approx(0.9)


# test_a_trainer_that_cannot_label_its_losses_still_reports_the_rest()
# Verifies one unreadable source does not cost the whole epoch's record.
# Inputs: none.
# Output: pytest pass/fail result.
def test_a_trainer_that_cannot_label_its_losses_still_reports_the_rest() -> None:

    trainer = FakeTrainer(0, {"metrics/precision(B)": 0.5})
    trainer.tloss = {"box_loss": 1.0}
    trainer.epoch_time = 9.0

    def exploding(_items, prefix="train"):
        raise RuntimeError("loss names moved")

    trainer.label_loss_items = exploding

    reported = training_module._epoch_metrics(trainer)

    assert reported["metrics/precision(B)"] == pytest.approx(0.5)
    assert reported["duration_seconds"] == pytest.approx(9.0)
    assert not any(name.endswith("_loss") for name in reported)


# test_an_epoch_says_where_it_sat_in_its_run()
# Verifies the two offsets, measured from the start of training.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Offsets rather than wall-clock instants is Isaac's decision: a volunteer's
# clock can be wrong by any amount, so an absolute timestamp from the worker
# would claim an agreement with MARP's other timestamps that nothing checks.
def test_an_epoch_says_where_it_sat_in_its_run() -> None:

    trainer = a_trainer_at_epoch_end(losses={"box_loss": 1.0}, epoch_time=27.5)
    trainer.train_time_start = 1_800_000_000.0
    trainer.epoch_time_start = 1_800_000_100.0

    reported = training_module._epoch_metrics(trainer)

    assert reported["end_offset_s"] == pytest.approx(100.0)
    assert reported["start_offset_s"] == pytest.approx(72.5)


# test_the_offsets_agree_with_the_duration_exactly()
# Verifies end - start is the duration, to the rounding.
# Inputs: none.
# Output: pytest pass/fail result.
#
# **This identity is the whole reason the start is derived rather than sampled.**
# Reading our own clock in `on_train_epoch_start` would give a start a few
# milliseconds from the trainer's, and these three numbers would then disagree
# with each other in the record for no benefit.
def test_the_offsets_agree_with_the_duration_exactly() -> None:

    for duration in (13.75, 0.001, 27.123, 600.5):
        trainer = a_trainer_at_epoch_end(epoch_time=duration)
        trainer.train_time_start = 1_800_000_000.0
        trainer.epoch_time_start = 1_800_000_000.0 + 3 * duration

        reported = training_module._epoch_metrics(trainer)

        assert reported["end_offset_s"] - reported["start_offset_s"] == pytest.approx(
            reported["duration_seconds"]
        ), f"identity broke at duration={duration}"


# test_the_first_epoch_starts_at_the_beginning_of_training_not_the_job()
# Verifies, and pins, that setup time is NOT inside these offsets.
# Inputs: none.
# Output: pytest pass/fail result.
#
# `train_time_start` and `epoch_time_start` are both set at `_do_train` entry,
# microseconds apart and AFTER `_setup_train` has built the dataloaders, loaded
# the model and built the optimizer. So epoch 1 begins at roughly zero however
# long setup took.
#
# This is worth a test rather than only a comment because the opposite was
# expected during design -- that a first epoch starting at 180s would show a
# three-minute setup. It does not, and anyone who later makes it do so should
# have to delete this test deliberately.
def test_the_first_epoch_starts_at_the_beginning_of_training_not_the_job() -> None:

    # A run whose setup took three minutes: the trainer's clocks still start
    # together, because Ultralytics sets them after setup has finished.
    started = 1_800_000_180.0
    trainer = a_trainer_at_epoch_end(epoch_time=20.0)
    trainer.train_time_start = started
    trainer.epoch_time_start = started + 20.0

    reported = training_module._epoch_metrics(trainer)

    assert reported["start_offset_s"] == pytest.approx(0.0)
    assert reported["end_offset_s"] == pytest.approx(20.0)


# test_neither_offset_is_reported_without_a_duration()
# Verifies half an interval is withheld rather than shipped alone.
# Inputs: none.
# Output: pytest pass/fail result.
def test_neither_offset_is_reported_without_a_duration() -> None:

    trainer = a_trainer_at_epoch_end()
    trainer.train_time_start = 1_800_000_000.0
    trainer.epoch_time_start = 1_800_000_100.0

    reported = training_module._epoch_metrics(trainer)

    assert "start_offset_s" not in reported
    assert "end_offset_s" not in reported


# test_no_offsets_without_an_origin()
# Verifies an offset is never measured from an origin that was not there.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Without `train_time_start` the only thing left to measure from would be zero,
# which would silently turn every epoch's offset into a wall-clock instant --
# exactly what Isaac chose against.
def test_no_offsets_without_an_origin() -> None:

    trainer = a_trainer_at_epoch_end(epoch_time=20.0)
    trainer.epoch_time_start = 1_800_000_100.0

    reported = training_module._epoch_metrics(trainer)

    assert reported["duration_seconds"] == pytest.approx(20.0)
    assert "start_offset_s" not in reported
    assert "end_offset_s" not in reported


# The only type names MARP's settings route accepts.
# Written out here rather than imported from the engine on purpose: a test that
# imported the engine's own mapping would agree with whatever the engine says,
# which is exactly the failure this pins. These four are MARP's contract, copied
# from its 400: "must be one of: real, int, bool, text".
CONTRACT_TYPE_NAMES = {"real", "int", "bool", "text"}


# test_every_reported_setting_type_is_one_the_coordinator_accepts()
# Verifies the whole report speaks MARP's vocabulary, not Python's.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# **This is the test the defect needed and did not have.** Run 205 resolved
# optimizer=AdamW, lr0=0.002 and momentum=0.9 perfectly and MARP stored none of
# them: one entry typed "float" makes the route refuse the WHOLE report with a
# 400, so the attempt recorded no settings at all. Every piece worked and two
# words did not match.
#
# No test of our own behaviour could see that, because the engine agreed with
# itself. So this one asserts against the contract's vocabulary, named here.
def test_every_reported_setting_type_is_one_the_coordinator_accepts(tmp_path, monkeypatch) -> None:

    a_stubbed_ultralytics(monkeypatch)

    # One of each shape the allow-list can produce: float, int, bool and str.
    spec = a_runnable_spec(
        tmp_path, epochs=1, lr0=0.002, imgsz=512, cos_lr=True, optimizer="AdamW"
    )

    ctx = RunContext(tmp_path / "attempt")
    ctx.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    TrainingEngine().run(ctx, spec)

    _engine, reported, _ignored = ctx.settings_reports[0]

    # Every entry, including the worker's own provenance ones.
    for name, entry in reported.items():
        assert entry["type"] in CONTRACT_TYPE_NAMES, (
            f"{name} was reported as {entry['type']!r}, which MARP refuses"
        )

    # And the four shapes landed where they should rather than all on text.
    assert reported["lr0"]["type"] == "real"
    assert reported["imgsz"]["type"] == "int"
    assert reported["cos_lr"]["type"] == "bool"
    assert reported["optimizer"]["type"] == "text"


# test_pythons_type_names_never_reach_the_report()
# The named negative: the two words that actually broke run 205.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# The test above would pass if a future entry were typed, say, "numeric" and the
# set were widened to match. This one names "float" and "str" specifically,
# because those are what Python hands you and what a new report will reach for.
def test_pythons_type_names_never_reach_the_report(tmp_path, monkeypatch) -> None:

    a_stubbed_ultralytics(monkeypatch)
    spec = a_runnable_spec(tmp_path, epochs=1, lr0=0.002, optimizer="AdamW")

    ctx = RunContext(tmp_path / "attempt")
    ctx.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    TrainingEngine().run(ctx, spec)

    _engine, reported, _ignored = ctx.settings_reports[0]
    emitted = {entry["type"] for entry in reported.values()}

    assert "float" not in emitted
    assert "str" not in emitted


# test_the_resolved_report_speaks_the_contracts_vocabulary_too()
# Verifies the second report, which is the one run 205 actually died on.
# Inputs: none.
# Output: pytest pass/fail result.
def test_the_resolved_report_speaks_the_contracts_vocabulary_too() -> None:

    trainer = FakeTrainer(0, {})
    trainer.optimizer = FakeSGD([{"initial_lr": 0.002, "momentum": 0.9}])

    resolved = training_module._resolved_training_settings(trainer)

    assert resolved["optimizer"]["type"] == "text"
    assert resolved["lr0"]["type"] == "real"
    assert resolved["momentum"]["type"] == "real"

    for entry in resolved.values():
        assert entry["type"] in CONTRACT_TYPE_NAMES


# test_an_unmappable_value_travels_as_text_rather_than_costing_the_report()
# Verifies one odd value cannot take every setting down with it.
# Inputs: none.
# Output: pytest pass/fail result.
#
# A refused report is all-or-nothing, so the expensive failure is not a wrong
# entry but a rejected batch. A type outside the four is carried as text.
def test_an_unmappable_value_travels_as_text_rather_than_costing_the_report() -> None:

    entry = training_module._setting_entry([10, 20], "job")

    assert entry["type"] == "text"
    assert entry["value"] == "[10, 20]"


# test_a_boolean_is_not_reported_as_an_integer()
# Verifies the one trap in Python's type system is handled.
# Inputs: none.
# Output: pytest pass/fail result.
#
# bool is a subclass of int, so an isinstance chain that tests int first reports
# True as an integer. The mapping is keyed by type name precisely so it cannot.
def test_a_boolean_is_not_reported_as_an_integer() -> None:

    assert training_module._setting_entry(True, "job")["type"] == "bool"
    assert training_module._setting_entry(1, "job")["type"] == "int"


# test_compute_device_names_the_card_not_the_slot()
# Verifies the column gets what it asks for.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
#
# MARP's schema comments that column 'Hardware used for training (e.g. "RTX
# 6000 Ada", "A100 GPU")'. The worker used to send "cuda:0", which answers a
# different question: it says which index the job was pinned to, not what the
# model was trained on. Two runs on two different cards both report "cuda:0",
# so the column could not have done its job even once something read it.
def test_compute_device_names_the_card_not_the_slot(monkeypatch) -> None:

    monkeypatch.setattr(
        training_module, "describe_cuda_devices",
        lambda: [{"index": 0, "name": "NVIDIA RTX 6000 Ada"},
                 {"index": 1, "name": "NVIDIA A100"}],
    )

    assert training_module._device_hardware("cuda:1") == "NVIDIA A100"

    provenance = training_module._run_provenance("cuda:1")
    assert provenance["compute_device"]["value"] == "NVIDIA A100"

    # And the slot is not lost -- which card misbehaved still matters.
    assert provenance["compute_device_index"]["value"] == "cuda:1"


# test_a_run_with_no_card_reports_the_device_it_had()
# Verifies the fallbacks name something true rather than nothing.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
#
# A CPU run has no card to name, and a driver that will not answer leaves
# describe_cuda_devices() without a name for that index. In both cases the
# device string is the most specific true thing available.
def test_a_run_with_no_card_reports_the_device_it_had(monkeypatch) -> None:

    monkeypatch.setattr(training_module, "describe_cuda_devices", lambda: [])

    assert training_module._device_hardware("cpu") == "cpu"
    assert training_module._device_hardware("cuda:0") == "cuda:0"

    # An index this machine does not have falls back rather than guessing at
    # whichever card happens to be first.
    monkeypatch.setattr(
        training_module, "describe_cuda_devices",
        lambda: [{"index": 0, "name": "NVIDIA A100"}],
    )
    assert training_module._device_hardware("cuda:3") == "cuda:3"


# test_naming_the_card_never_costs_the_run()
# Verifies a driver that throws does not take the report down with it.
# Inputs: monkeypatch.
# Output: pytest pass/fail result.
def test_naming_the_card_never_costs_the_run(monkeypatch) -> None:

    def exploding():
        raise RuntimeError("driver reset")

    monkeypatch.setattr(training_module, "describe_cuda_devices", exploding)

    assert training_module._device_hardware("cuda:0") == "cuda:0"
    assert training_module._run_provenance("cuda:0")["compute_device"]["value"] == "cuda:0"


# The only `metric_family` values MARP's enum accepts.
# Written out here, copied from the API side's built enum, and deliberately NOT
# imported from the engine -- a test that imported our own constant would agree
# with whatever we emit, which is the failure this pins. MARP does not validate
# this field, so a wrong word would not throw: it would store a row nobody
# renders and a table that silently never appears.
CONTRACT_METRIC_FAMILIES = {"box", "seg", "classify"}

# The two roles MARP looks up by name. Everything else is opaque to it and is
# stored under whatever the worker sent.
CONTRACT_EXACT_ROLES = {"final_metrics", "metrics_curves"}


# FakeFamily
# The attributes the end-of-run readers touch on an Ultralytics Metric.
class FakeFamily:

    def __init__(self, ap_class_index, p, r, f1, ap50, ap, curves=None) -> None:

        self.ap_class_index = ap_class_index
        self.p, self.r, self.f1, self.ap50, self.ap = p, r, f1, ap50, ap
        self.curves_results = curves or []

    def mean_results(self):
        return [0.81, 0.74, 0.79, 0.52]

    def fitness(self):
        return 0.52


# FakeMetrics
# The validator's metrics holder: one family plus the class-id-indexed counts.
class FakeMetrics:

    def __init__(self, names, family, nt_per_class=None, nt_per_image=None) -> None:

        self.names = names
        self.box = family
        self.nt_per_class = nt_per_class
        self.nt_per_image = nt_per_image


# FakeConfusion
# Ultralytics' ConfusionMatrix, as far as the reader is concerned.
class FakeConfusion:

    def __init__(self, names, matrix) -> None:

        self.names = names
        self.matrix = matrix


# FakeValidator
class FakeValidator:

    def __init__(self, confusion, metrics) -> None:

        self.confusion_matrix = confusion
        self.metrics = metrics


# a_finished_trainer()
# A trainer as it looks at on_train_end, with three classes and two evaluated.
# Inputs: optional overrides.
# Output: the fake trainer.
#
# Three in the vocabulary and two evaluated is the shape that matters: it is
# the case where the confusion matrix's axes and the per-class rows disagree,
# which is the defect this whole artifact shape was corrected for.
def a_finished_trainer(matrix=None, curves=None, names=None):

    import numpy as np

    names = names if names is not None else {0: "Lingcod", 1: "Rockfish", 2: "Cookie star"}
    family = FakeFamily(
        ap_class_index=np.array([0, 2]),           # class 1 never evaluated
        p=[0.90, 0.60], r=[0.80, 0.50], f1=[0.85, 0.55],
        ap50=[0.88, 0.40], ap=[0.61, 0.22],
        curves=curves,
    )
    metrics = FakeMetrics(
        names=names,
        family=family,
        nt_per_class=[142, 0, 7],                  # indexed by CLASS ID
        nt_per_image=[37, 0, 5],
    )
    if matrix is None:
        matrix = np.array([[8.0, 1, 1, 0], [0, 9, 1, 0], [2, 0, 8, 0], [1, 0, 2, 0]])

    trainer = FakeTrainer(0, {})
    trainer.args = types.SimpleNamespace(task="detect")
    trainer.data = {"val": "images/val"}
    trainer.validator = FakeValidator(FakeConfusion(names, matrix), metrics)
    return trainer


# test_the_matrix_axes_carry_the_full_vocabulary()
# Verifies the correction that this artifact shape exists for.
# Inputs: none.
# Output: pytest pass/fail result.
#
# **`ConfusionMatrix.names` is the FULL vocabulary; `ap_class_index` is only
# what validation scored.** A species absent from validation is on the matrix
# axes and has no metrics row. Labelling the axes from the evaluated list would
# shift every label after it -- silently, and precisely on the rare species a
# reader is looking for.
def test_the_matrix_axes_carry_the_full_vocabulary() -> None:

    report = training_module._final_metrics_report(a_finished_trainer())
    matrix = report["confusion_matrix"]

    assert matrix["classes"] == ["Lingcod", "Rockfish", "Cookie star"]

    # Rockfish was never evaluated: on the axes, absent from the rows.
    scored = [row["class"] for row in report["metrics"][0]["per_class"]]
    assert scored == ["Lingcod", "Cookie star"]
    assert "Rockfish" not in scored


# test_background_is_derived_from_the_matrix_not_the_task()
# Verifies background_index is checkable rather than trusted.
# Inputs: none.
# Output: pytest pass/fail result.
def test_background_is_derived_from_the_matrix_not_the_task() -> None:

    import numpy as np

    report = training_module._final_metrics_report(a_finished_trainer())
    matrix = report["confusion_matrix"]

    # Always exactly len(classes), so a reader can verify rather than trust.
    assert matrix["background_index"] == len(matrix["classes"]) == 3
    assert len(matrix["matrix"]) == len(matrix["classes"]) + 1

    # A square matrix has no background row, and the key is ABSENT rather than
    # null -- the same absence rule as everywhere else in this work.
    square = a_finished_trainer(matrix=np.zeros((3, 3)))
    assert "background_index" not in training_module._final_metrics_report(
        square
    )["confusion_matrix"]


# test_the_orientation_is_stated_rather_than_assumed()
# Verifies the transpose trap is named in the payload.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Ultralytics fills matrix[predicted][true], the transpose of how a person reads
# one. A reader that assumes the other way round produces a confident lie about
# which species is confused with which, so the array says which way it is.
def test_the_orientation_is_stated_rather_than_assumed() -> None:

    report = training_module._final_metrics_report(a_finished_trainer())

    assert report["confusion_matrix"]["orientation"] == "rows_predicted_cols_true"


# test_per_class_counts_come_from_the_class_id_not_the_position()
# Verifies the dual-index trap is handled.
# Inputs: none.
# Output: pytest pass/fail result.
#
# `p`/`r`/`f1` are indexed by position within the evaluated set; `nt_per_class`
# and `nt_per_image` are indexed by CLASS ID. Ultralytics' own code reads both
# in one loop. Cookie star is class 2 but evaluated position 1, so a reader that
# used the position would give it Rockfish's counts.
def test_per_class_counts_come_from_the_class_id_not_the_position() -> None:

    rows = training_module._final_metrics_report(
        a_finished_trainer()
    )["metrics"][0]["per_class"]

    cookie = next(row for row in rows if row["class"] == "Cookie star")

    # Class id 2 -> support 7, images 5. Position 1 would have given 0 and 0.
    assert cookie["support"] == 7
    assert cookie["images"] == 5
    assert cookie["precision"] == pytest.approx(0.60)


# test_the_family_is_a_word_marp_accepts()
# The vocabulary test, against the contract rather than our own output.
# Inputs: none.
# Output: pytest pass/fail result.
#
# MARP does not validate this field: a wrong word is stored, renders nowhere,
# and the table silently never appears. Nothing would throw, so only a test
# naming the permitted values can catch it.
def test_the_family_is_a_word_marp_accepts() -> None:

    report = training_module._final_metrics_report(a_finished_trainer())

    for block in report["metrics"]:
        assert block["family"] in CONTRACT_METRIC_FAMILIES

    assert report["metrics"][0]["family"] == "box"

    # And the constant itself, since segmentation has no fake here yet.
    assert {name for _attr, name in training_module._METRIC_FAMILIES} <= (
        CONTRACT_METRIC_FAMILIES
    )
    assert ("seg", "seg") in training_module._METRIC_FAMILIES
    assert not any(name == "mask" for _attr, name in training_module._METRIC_FAMILIES)


# test_no_aggregate_f1_is_invented()
# Verifies the one number Ultralytics does not expose stays absent.
# Inputs: none.
# Output: pytest pass/fail result.
#
# mean_results() returns four values and none is an F1. 2PR/(P+R) of the means
# is a third quantity Ultralytics never computes, and it would sit in a column
# whose other rows hold a real measurement.
def test_no_aggregate_f1_is_invented() -> None:

    overall = training_module._final_metrics_report(
        a_finished_trainer()
    )["metrics"][0]["overall"]

    assert "f1" not in overall
    assert overall["map50"] == pytest.approx(0.79)
    assert overall["fitness"] == pytest.approx(0.52)


# test_a_nan_becomes_null_rather_than_breaking_the_file()
# Verifies an unscored class cannot cost the whole artifact.
# Inputs: none.
# Output: pytest pass/fail result.
#
# **json.dumps writes NaN as a bare `NaN`, which is not valid JSON.** Ultralytics
# yields NaN for a class it could not score, so without this one unmeasured
# species would make the entire report unparseable -- the run 205 failure in a
# new place.
def test_a_nan_becomes_null_rather_than_breaking_the_file() -> None:

    import json as _json
    import numpy as np

    trainer = a_finished_trainer()
    trainer.validator.metrics.box.p = [float("nan"), 0.60]

    report = training_module._final_metrics_report(trainer)
    rows = report["metrics"][0]["per_class"]

    assert rows[0]["precision"] is None

    # The real check: it survives a strict encode.
    _json.dumps(report, allow_nan=False)


# test_only_the_confidence_curves_are_sent()
# Verifies the PR curve is dropped, and why.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Ultralytics reuses ONE px array for two meanings: the PR curve's x axis is
# recall, the other three are confidence. It distinguishes them only by the
# label it attaches. Storing the PR curve against a confidence axis would record
# a recall value as a confidence.
def test_only_the_confidence_curves_are_sent() -> None:

    px = [0.0, 0.5, 1.0]
    curves = [
        [px, [[0.9, 0.5, 0.1]], "Recall", "Precision"],      # the PR curve
        [px, [[0.8, 0.6, 0.2]], "Confidence", "F1"],
        [px, [[0.7, 0.5, 0.3]], "Confidence", "Precision"],
        [px, [[0.6, 0.4, 0.2]], "Confidence", "Recall"],
    ]

    report = training_module._curves_report(a_finished_trainer(curves=curves), "val")

    assert set(report["curves"]) == {"precision", "recall", "f1"}

    # The confidence axis is stored once, not per curve.
    assert report["confidence"] == [0.0, 0.5, 1.0]

    # Precision came from the Confidence-labelled curve, not the PR one.
    assert report["curves"]["precision"] == [[0.7, 0.5, 0.3]]

    # The curve rows are the EVALUATED classes, parallel to the arrays.
    assert report["classes"] == ["Lingcod", "Cookie star"]
    assert report["family"] == "box"


# test_one_unreadable_section_does_not_cost_the_others()
# Verifies the run 205 lesson is built in.
# Inputs: none.
# Output: pytest pass/fail result.
#
# An all-or-nothing payload turns one bad value into a total loss. A matrix that
# cannot be read must still leave the per-class rows.
def test_one_unreadable_section_does_not_cost_the_others() -> None:

    trainer = a_finished_trainer()
    trainer.validator.confusion_matrix = None

    report = training_module._final_metrics_report(trainer)

    assert "confusion_matrix" not in report
    assert len(report["metrics"][0]["per_class"]) == 2


# test_a_run_that_measured_nothing_sends_nothing()
# Verifies an empty report is not written.
# Inputs: none.
# Output: pytest pass/fail result.
#
# A file saying `{"schema": 1}` would claim a run was scored when it was not.
def test_a_run_that_measured_nothing_sends_nothing() -> None:

    trainer = a_finished_trainer()
    trainer.validator.confusion_matrix = None
    trainer.validator.metrics = None

    assert training_module._final_metrics_report(trainer) is None


# PublishingContext
# A RunContext that remembers every role it was asked to publish.
class PublishingContext(RunContext):

    def __init__(self, checkpoint_dir) -> None:

        super().__init__(checkpoint_dir)
        self.published: list[str] = []

    def publish_artifact(self, path, kind) -> str:
        assert path.is_file(), f"{path} was published but does not exist"
        self.published.append(kind)
        return "0" * 64


# test_the_plot_roles_are_the_real_filename_stems()
# Verifies case is preserved and nothing is filtered.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# MARP stores the role as sent and serves it back under that name, and the
# dashboard keys off it, so `PR_curve` must not become `pr_curve` -- that would
# make the record disagree with the run directory for no gain. There is no
# allow-list on either side, which is the point: a plot a later Ultralytics adds
# arrives rather than being dropped by a filter nobody updated.
def test_the_plot_roles_are_the_real_filename_stems(tmp_path) -> None:

    run_dir = tmp_path / "train" / "run"
    (run_dir / "weights").mkdir(parents=True)
    for name in (
        "confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png",
        "F1_curve.png", "labels_correlogram.jpg", "val_batch0_pred.jpg",
        "results.csv", "a_plot_nobody_has_seen_yet.png",
    ):
        (run_dir / name).write_bytes(b"x")
    # Weights are published separately and are ~130 MB; they must not come here.
    (run_dir / "weights" / "best.pt").write_bytes(b"x")

    ctx = PublishingContext(tmp_path)
    state = training_module._RunState(ctx=ctx, total_epochs=1, run_root=tmp_path / "train")

    trainer = a_finished_trainer()
    trainer.save_dir = run_dir
    state._publish_run_files(trainer)

    assert "PR_curve" in ctx.published
    assert "F1_curve" in ctx.published
    assert "pr_curve" not in ctx.published
    assert "results" in ctx.published

    # A stem neither side anticipated arrives rather than being filtered.
    assert "a_plot_nobody_has_seen_yet" in ctx.published

    # And the weights did not.
    assert "best" not in ctx.published


# test_the_two_looked_up_roles_are_exactly_lowercase()
# Verifies the two names MARP matches literally.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# Every other role is opaque to MARP and stored as sent. These two are looked up
# by name, so their spelling is a contract rather than a label.
def test_the_two_looked_up_roles_are_exactly_lowercase(tmp_path) -> None:

    ctx = PublishingContext(tmp_path)
    state = training_module._RunState(ctx=ctx, total_epochs=1, run_root=tmp_path)

    state.on_train_end(a_finished_trainer(curves=[
        [[0.0, 1.0], [[0.5, 0.4]], "Confidence", "F1"],
        [[0.0, 1.0], [[0.6, 0.3]], "Confidence", "Precision"],
        [[0.0, 1.0], [[0.7, 0.2]], "Confidence", "Recall"],
    ]))

    assert "final_metrics" in ctx.published
    assert "metrics_curves" in ctx.published
    assert CONTRACT_EXACT_ROLES <= set(ctx.published)


# test_bookkeeping_never_fails_a_finished_run()
# Verifies a publishing fault cannot throw away a completed run.
# Inputs: tmp_path.
# Output: pytest pass/fail result.
#
# By the time this callback runs the training is done and the weights are
# already published. A fault here would discard a complete result over a report.
def test_bookkeeping_never_fails_a_finished_run(tmp_path) -> None:

    class Exploding(PublishingContext):
        def publish_artifact(self, path, kind) -> str:
            raise RuntimeError("coordinator went away")

    ctx = Exploding(tmp_path)
    state = training_module._RunState(ctx=ctx, total_epochs=1, run_root=tmp_path)

    # Must not raise.
    state.on_train_end(a_finished_trainer())

    assert any("could not publish" in line for line in ctx.logs)


# a_classification_spec()
# A job whose dataset is a classify directory rather than a data.yaml.
# Inputs: tmp_path, whether to give it a train/ split, and any param overrides.
# Output: the spec mapping.
#
# Ultralytics classify takes a DIRECTORY and reads its class vocabulary from the
# subfolder names; there is no names list anywhere in a classify dataset.
def a_classification_spec(tmp_path: Path, with_train: bool = True, **params) -> dict:

    root = tmp_path / "cls-dataset"
    if with_train:
        for split in ("train", "val"):
            for name in ("Lingcod", "Rockfish"):
                (root / split / name).mkdir(parents=True, exist_ok=True)
                (root / split / name / "a.jpg").write_bytes(b"x")
    else:
        # Images, but no split: exactly the shape Ultralytics would silently
        # split 80/20 for itself.
        (root / "Lingcod").mkdir(parents=True, exist_ok=True)
        (root / "Lingcod" / "a.jpg").write_bytes(b"x")

    weights = tmp_path / "yolo11n-cls.pt"
    weights.write_bytes(b"not-really-weights")

    base = {"dataset_path": str(root), "model_path": str(weights)}
    base.update(params)
    return {"engine": "marp_training", "params": base}


# test_a_classification_dataset_with_no_split_is_refused()
# Verifies the worker refuses rather than letting Ultralytics invent a split.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# **This is the most important check in the classification work.** Handed a
# directory with images and no `train/`, Ultralytics does not fail -- it
# silently splits 80/20 at random (`split_classify_dataset`,
# data/utils.py:718-726) and logs it at INFO.
#
# MARP splits deliberately and temporally. A random split puts frames from the
# same dive seconds apart on both sides of it, so the model is scored on
# near-duplicates of what it trained on: accuracy that looks excellent and means
# nothing, with nothing downstream saying so. Refusing before training starts is
# the only honest answer, and it costs a volunteer minutes rather than hours.
def test_a_classification_dataset_with_no_split_is_refused(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "cuda_device_count", lambda: 1)

    spec = a_classification_spec(tmp_path, with_train=False)

    with pytest.raises(JobUnrunnable) as refusal:
        TrainingEngine().preflight(spec)

    # The reason has to say what is wrong and why it is not just fixed up.
    message = str(refusal.value)
    assert "train/" in message
    assert "80/20" in message


# test_a_classification_dataset_with_a_split_is_accepted()
# The control for the refusal above.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# Without it, a preflight that refused every classification dataset would pass
# the test above and nobody would notice until a real job.
def test_a_classification_dataset_with_a_split_is_accepted(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "cuda_device_count", lambda: 1)

    TrainingEngine().preflight(a_classification_spec(tmp_path))


# test_a_classification_run_is_given_the_directory_not_a_yaml()
# Verifies what actually reaches Ultralytics.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# The preflight accepting it means nothing if run() still passes a data.yaml
# that does not exist. This watches the argument the trainer receives.
def test_a_classification_run_is_given_the_directory_not_a_yaml(tmp_path, monkeypatch) -> None:

    model = a_stubbed_ultralytics(monkeypatch)
    spec = a_classification_spec(tmp_path, epochs=1)

    ctx = RunContext(tmp_path / "attempt")
    ctx.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary = TrainingEngine().run(ctx, spec)

    passed = Path(model.train_kwargs["data"])
    assert passed.is_dir()
    assert passed.name == "cls-dataset"

    # And the summary names the dataset rather than a yaml's parent.
    assert Path(summary["dataset"]).name == "cls-dataset"


# test_a_detection_run_still_gets_its_data_yaml()
# The control: adding classification must not move detection.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
def test_a_detection_run_still_gets_its_data_yaml(tmp_path, monkeypatch) -> None:

    model = a_stubbed_ultralytics(monkeypatch)
    spec = a_runnable_spec(tmp_path, epochs=1)

    ctx = RunContext(tmp_path / "attempt")
    ctx.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    TrainingEngine().run(ctx, spec)

    passed = Path(model.train_kwargs["data"])
    assert passed.is_file()
    assert passed.name == "data.yaml"


# FakeClassifyMetrics
# ClassifyMetrics as far as the reader is concerned: two accuracies, a fitness,
# and no box, no seg, no per-class anything.
class FakeClassifyMetrics:

    def __init__(self, top1=0.82, top5=0.97) -> None:

        self.top1 = top1
        self.top5 = top5
        self.fitness = 0.895
        self.names = {0: "Lingcod", 1: "Rockfish"}


# test_a_classification_run_reports_its_accuracies()
# Verifies classify produces a metrics block at all.
# Inputs: none.
# Output: pytest pass/fail result.
#
# `_metric_families` iterates box and seg; ClassifyMetrics has neither, so
# without a classify branch a classification run would send a confusion matrix
# and NO metrics block -- which looks like a run that measured nothing.
#
# The key names come from a real one-epoch run: `accuracy_top1` and
# `accuracy_top5`, with no (B)/(M) suffix, because that suffix distinguishes box
# from mask and classification is neither.
def test_a_classification_run_reports_its_accuracies() -> None:

    blocks = training_module._metric_families(FakeClassifyMetrics())

    assert len(blocks) == 1
    assert blocks[0]["family"] == "classify"
    assert blocks[0]["overall"]["accuracy_top1"] == pytest.approx(0.82)
    assert blocks[0]["overall"]["accuracy_top5"] == pytest.approx(0.97)

    # The family is still a word MARP accepts.
    assert blocks[0]["family"] in CONTRACT_METRIC_FAMILIES


# test_a_classification_block_has_no_per_class()
# Verifies the absence is deliberate rather than an empty list.
# Inputs: none.
# Output: pytest pass/fail result.
#
# ClassifyMetrics has no per-class quantities at all -- no precision, recall or
# AP per class. The confusion matrix is the only per-class information a
# classify run produces, so an empty `per_class` would imply rows were measured
# and came back empty.
def test_a_classification_block_has_no_per_class() -> None:

    block = training_module._metric_families(FakeClassifyMetrics())[0]

    assert "per_class" not in block


# test_a_detection_family_still_wins_over_the_classify_branch()
# Verifies the classify branch is a fallback, not a competitor.
# Inputs: none.
# Output: pytest pass/fail result.
def test_a_detection_family_still_wins_over_the_classify_branch() -> None:

    blocks = training_module._metric_families(
        a_finished_trainer().validator.metrics
    )

    assert [block["family"] for block in blocks] == ["box"]


# test_the_engine_does_not_claim_to_be_a_detector()
# Verifies /status stops asserting a task the engine cannot know.
# Inputs: none.
# Output: pytest pass/fail result.
#
# **The task is decided by the checkpoint.** Ultralytics reads it off the .pt
# and ignores any task argument for one, so the engine does not learn the task
# until a job names its weights. describe() used to say "detect" flatly, which
# was false for every segmentation run.
def test_the_engine_does_not_claim_to_be_a_detector() -> None:

    described = TrainingEngine().describe()

    assert "task" not in described
    assert set(described["tasks"]) == {"detect", "segment", "classify"}


# a_training_trainer()
# A trainer as it looks mid-epoch, with the attributes the sampler reads.
# Inputs: elapsed seconds since the epoch started, and batches per epoch.
# Output: the fake trainer.
def a_training_trainer(elapsed=2.0, batches=100):

    trainer = FakeTrainer(0, {})
    trainer.args = types.SimpleNamespace(task="detect", imgsz=640)
    trainer.tloss = {"box_loss": 1.2, "cls_loss": 0.8, "dfl_loss": 0.9}
    trainer.label_loss_items = lambda items, prefix="train": {
        f"{prefix}/{k}": round(float(v), 5) for k, v in items.items()
    }
    trainer.train_loader = [None] * batches
    trainer.epoch_time_start = time.time() - elapsed
    trainer._get_memory = lambda: 7.25
    return trainer


# test_a_live_sample_carries_the_console_line()
# Verifies the sample holds what a person watching the CLI would see.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
def test_a_live_sample_carries_the_console_line(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "_SAMPLE_INTERVAL_S", 0.0)

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=10, run_root=tmp_path)

    trainer = a_training_trainer(elapsed=2.0, batches=100)
    state.on_batch_end(trainer)

    step, phase, sample = ctx.metrics[-1]

    assert phase == training_module._BATCH_PHASE
    assert step == 1
    assert sample["epoch"] == 1 and sample["epochs_total"] == 10
    assert sample["batch"] == 1 and sample["batches_total"] == 100
    assert sample["train/box_loss"] == pytest.approx(1.2)
    assert sample["gpu_memory_gb"] == pytest.approx(7.25)
    assert sample["imgsz"] == 640
    assert "at" in sample


# test_the_pace_is_arithmetic_on_measurements()
# Verifies it/s and the ETA are derived, not estimated.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# Elapsed-since-epoch-start and the batch counts are both real, so the division
# is arithmetic rather than a guess. Ten batches in two seconds is five a
# second, and ninety left at five a second is eighteen seconds.
def test_the_pace_is_arithmetic_on_measurements(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "_SAMPLE_INTERVAL_S", 0.0)

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=1, run_root=tmp_path)

    trainer = a_training_trainer(elapsed=2.0, batches=100)
    for _ in range(10):
        state.on_batch_end(trainer)

    _step, _phase, sample = ctx.metrics[-1]

    assert sample["iterations_per_second"] == pytest.approx(5.0, rel=0.05)
    assert sample["epoch_eta_seconds"] == pytest.approx(18.0, rel=0.05)


# test_no_pace_before_the_first_batch_finishes()
# Verifies a rate over no elapsed work is omitted rather than reported as zero.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# A rate measured over zero batches is not slow, it is undefined. Reporting 0.0
# would draw a chart starting at the floor on every epoch.
def test_no_pace_before_the_first_batch_finishes(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "_SAMPLE_INTERVAL_S", 0.0)

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=1, run_root=tmp_path)

    # on_epoch_start forces a sample with no batches done yet.
    state.on_epoch_start(a_training_trainer())

    _step, _phase, sample = ctx.metrics[-1]

    assert "iterations_per_second" not in sample
    assert "epoch_eta_seconds" not in sample


# test_samples_are_throttled_but_a_phase_change_always_reports()
# Verifies a fast card cannot flood, and a phase change is never lost.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# Throttled by TIME rather than batch count, because batches are not a unit of
# time: "every N batches" gives a chart whose density depends on image size.
def test_samples_are_throttled_but_a_phase_change_always_reports(tmp_path, monkeypatch) -> None:

    # Long enough that nothing in this test can clear it.
    monkeypatch.setattr(training_module, "_SAMPLE_INTERVAL_S", 3600.0)

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=1, run_root=tmp_path)
    trainer = a_training_trainer()

    state.on_epoch_start(trainer)
    before = len(ctx.metrics)

    for _ in range(500):
        state.on_batch_end(trainer)

    # Five hundred batches, none of them sampled.
    assert len(ctx.metrics) == before

    # But the counters still moved, so the next sample is accurate.
    assert state._global_step == 500
    assert state._batch_in_epoch == 500


# test_the_batch_counter_resets_each_epoch_and_the_step_does_not()
# Verifies the two counters mean different things.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# `batch` is what the console counts and restarts every epoch; `step` is the
# run's x axis and must not, or every epoch would overwrite the last.
def test_the_batch_counter_resets_each_epoch_and_the_step_does_not(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "_SAMPLE_INTERVAL_S", 0.0)

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=2, run_root=tmp_path)
    trainer = a_training_trainer()

    state.on_epoch_start(trainer)
    for _ in range(3):
        state.on_batch_end(trainer)

    state.on_epoch_start(trainer)
    state.on_batch_end(trainer)

    _step, _phase, sample = ctx.metrics[-1]

    assert sample["batch"] == 1
    assert state._global_step == 4


# test_the_live_phases_are_not_the_epoch_phase()
# Verifies the two series can be told apart by a word, not by magnitude.
# Inputs: none.
# Output: pytest pass/fail result.
#
# **The per-epoch rows already use report_metrics with phase "training" and
# step = the epoch number.** If live samples shared that phase, the two series
# would be distinguishable only by how big `step` is -- which works on every
# long run and collapses the first time a one-epoch run makes both equal 1.
def test_the_live_phases_are_not_the_epoch_phase() -> None:

    assert training_module._BATCH_PHASE != "training"
    assert training_module._VAL_BATCH_PHASE != "training"
    assert training_module._BATCH_PHASE != training_module._VAL_BATCH_PHASE


# test_a_validation_sample_counts_the_validators_batches()
# Verifies the validation pause gets its own bar.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
#
# The validator is handed to this callback, not the trainer -- it has its own
# zero-based batch_i and dataloader, which is the only reason a bar is possible.
def test_a_validation_sample_counts_the_validators_batches(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "_SAMPLE_INTERVAL_S", 0.0)

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=1, run_root=tmp_path)

    validator = types.SimpleNamespace(batch_i=4, dataloader=[None] * 20)
    state.on_val_batch_end(validator)

    _step, phase, sample = ctx.metrics[-1]

    assert phase == training_module._VAL_BATCH_PHASE
    # batch_i is zero-based and a person counts from one.
    assert sample["batch"] == 5
    assert sample["batches_total"] == 20


# test_one_unreadable_field_does_not_cost_the_sample()
# Verifies a sample degrades rather than disappearing.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
def test_one_unreadable_field_does_not_cost_the_sample(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "_SAMPLE_INTERVAL_S", 0.0)

    ctx = RecordingContext()
    state = training_module._RunState(ctx=ctx, total_epochs=1, run_root=tmp_path)

    trainer = a_training_trainer()

    def exploding():
        raise RuntimeError("driver reset")

    trainer._get_memory = exploding

    state.on_batch_end(trainer)

    _step, _phase, sample = ctx.metrics[-1]

    assert "gpu_memory_gb" not in sample
    assert sample["train/box_loss"] == pytest.approx(1.2)
    assert sample["batch"] == 1


# test_sampling_never_fails_the_run()
# Verifies a chart cannot throw away GPU hours.
# Inputs: tmp_path and monkeypatch.
# Output: pytest pass/fail result.
def test_sampling_never_fails_the_run(tmp_path, monkeypatch) -> None:

    monkeypatch.setattr(training_module, "_SAMPLE_INTERVAL_S", 0.0)

    class Exploding(RecordingContext):
        def report_metrics(self, step, phase, metrics) -> None:
            raise RuntimeError("coordinator went away")

    state = training_module._RunState(
        ctx=Exploding(), total_epochs=1, run_root=tmp_path
    )

    # None of these may raise.
    state.on_epoch_start(a_training_trainer())
    state.on_batch_end(a_training_trainer())
    state.on_val_batch_end(types.SimpleNamespace(batch_i=0, dataloader=[None]))
