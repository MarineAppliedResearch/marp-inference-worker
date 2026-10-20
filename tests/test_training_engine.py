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
