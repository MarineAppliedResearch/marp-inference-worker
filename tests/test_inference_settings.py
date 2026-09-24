# test_inference_settings.py
# Created: 2026-09-23
# Author: Isaac Travers
#
# Unit tests for the record of how an attempt ran (MARP_API#232).
#
# The report is built by a pure function from what the engine has already
# resolved, so it is tested here without a GPU, a model or a video. What these
# tests guard is that every value recorded is the value that actually ran --
# including the defaults, which is where a record goes quietly wrong.
import json
from io import StringIO
from pathlib import Path

from marp_inference_worker.engines.tracking_engine import (
    _SETTING_TYPES,
    inference_settings_report,
)
from marp_inference_worker.jobs.context import ChildJobContext


# What Ultralytics 8.4.145 falls back to, injected so the report's own logic is
# tested apart from the library. The real values are checked separately below.
_DEFAULTS = {"imgsz": 640, "iou": 0.7, "augment": False, "agnostic_nms": False,
             "max_det": 300, "half": False}

# The engine's own ByteTrack defaults, as TrackerArgs merges them.
_TRACKER = {"track_thresh": 0.30, "match_thresh": 0.70, "track_buffer": 240, "mot20": False}


# _report()
# Builds a report with the given overrides of an otherwise default job.
# Inputs: keyword overrides for any of the function's arguments.
# Output: (settings, ignored).
def _report(**overrides):

    arguments = {
        "params": {},
        "confidence": 0.15,
        "predict_options": {},
        "tracker_settings": dict(_TRACKER),
        "model_overrides": {},
        "unknown": [],
        "ultralytics_defaults": _DEFAULTS,
    }
    arguments.update(overrides)
    return inference_settings_report(**arguments)


# test_an_unset_imgsz_is_the_models_training_size_not_ultralytics_default()
# Verifies the default recorded is the one predict() actually used.
# Inputs: none.
# Output: pytest pass/fail result.
#
# The case this function exists for. Model.predict() merges the checkpoint's own
# overrides first, and a loaded .pt keeps its training imgsz there -- so a job
# setting no imgsz runs at the model's size. Recording DEFAULT_CFG's 640 would be
# a false record of a 1280 run.
def test_an_unset_imgsz_is_the_models_training_size_not_ultralytics_default() -> None:

    settings, _ = _report(params={"conf": 0.001}, confidence=0.001,
                          model_overrides={"imgsz": 1280})

    assert settings["imgsz"] == {"value": 1280, "type": "int", "source": "default"}
    assert settings["confidence"] == {"value": 0.001, "type": "real", "source": "job"}


# test_every_setting_is_reported_under_its_catalogue_type()
# Verifies nothing is missing and every value is in its declared type.
# Inputs: none.
# Output: pytest pass/fail result.
#
# MARP refuses a value whose type disagrees with its catalogue entry, so a type
# drifting here would lose the whole report to the fallback log line.
def test_every_setting_is_reported_under_its_catalogue_type() -> None:

    settings, _ = _report()

    assert set(settings) == set(_SETTING_TYPES)

    python_type = {"real": float, "int": int, "bool": bool}
    for name, entry in settings.items():
        assert entry["type"] == _SETTING_TYPES[name]
        assert type(entry["value"]) is python_type[entry["type"]], name


# test_what_the_job_set_is_the_jobs_and_the_rest_is_default()
# Verifies each value says where it came from.
# Inputs: none.
# Output: pytest pass/fail result.
def test_what_the_job_set_is_the_jobs_and_the_rest_is_default() -> None:

    settings, _ = _report(
        params={"imgsz": 1280, "iou": 0.2, "track_thresh": 0.01},
        predict_options={"imgsz": 1280, "iou": 0.2},
        tracker_settings={**_TRACKER, "track_thresh": 0.01},
    )

    assert {name: entry["source"] for name, entry in settings.items()} == {
        "confidence": "default",
        "imgsz": "job",
        "iou": "job",
        "augment": "default",
        "agnostic_nms": "default",
        "max_det": "default",
        "half": "default",
        "track_thresh": "job",
        "match_thresh": "default",
        "track_buffer": "default",
        "mot20": "default",
        "class_match_iou": "engine",
    }
    assert settings["track_thresh"]["value"] == 0.01


# test_a_value_is_recorded_as_the_type_it_ran_as()
# Verifies the conversions that keep a report acceptable without falsifying it.
# Inputs: none.
# Output: pytest pass/fail result.
#
# A square checkpoint imgsz of [1280, 1280] is one number; a track_buffer of
# 300.0 behaves as 300; a real setting that happens to be whole is still real.
def test_a_value_is_recorded_as_the_type_it_ran_as() -> None:

    settings, _ = _report(
        model_overrides={"imgsz": [1280, 1280]},
        tracker_settings={**_TRACKER, "track_buffer": 300.0, "match_thresh": 1},
    )

    assert settings["imgsz"]["value"] == 1280
    assert settings["track_buffer"]["value"] == 300
    assert type(settings["track_buffer"]["value"]) is int
    assert settings["match_thresh"]["value"] == 1.0
    assert type(settings["match_thresh"]["value"]) is float


# test_keys_nothing_honours_are_reported_with_what_the_job_asked_for()
# Verifies an ignored setting is visible, with its value.
# Inputs: none.
# Output: pytest pass/fail result.
#
# A setting silently not applied is what made a run at the deep-sea settings
# indistinguishable from one at the defaults.
def test_keys_nothing_honours_are_reported_with_what_the_job_asked_for() -> None:

    _, ignored = _report(params={"track_threshh": 0.5, "tracker": None},
                         unknown=["track_threshh", "tracker"])

    assert ignored == {"track_threshh": 0.5, "tracker": None}


# test_the_real_ultralytics_defaults_are_read_when_none_are_injected()
# Verifies the fallback reads the installed library rather than a copy.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Not skipped when Ultralytics is absent: it is a declared dependency, and a
# skipped test looks green.
def test_the_real_ultralytics_defaults_are_read_when_none_are_injected() -> None:

    from ultralytics.cfg import DEFAULT_CFG

    settings, _ = _report(ultralytics_defaults=None)

    assert settings["imgsz"]["value"] == DEFAULT_CFG.imgsz
    assert settings["iou"]["value"] == DEFAULT_CFG.iou
    assert settings["max_det"]["value"] == DEFAULT_CFG.max_det


# test_the_context_writes_the_report_as_a_settings_event()
# Verifies the report leaves the child in the shape the parent forwards.
# Inputs: the pytest temporary directory.
# Output: pytest pass/fail result.
def test_the_context_writes_the_report_as_a_settings_event(tmp_path: Path) -> None:

    stream = StringIO()
    ctx = ChildJobContext(params={}, checkpoint_dir=tmp_path / "work",
                          stop_file=tmp_path / "STOP", stream=stream)

    settings, ignored = _report()
    ctx.report_settings("marp_tracking", settings, ignored)

    [event] = [json.loads(line) for line in stream.getvalue().splitlines()]

    assert event["kind"] == "settings"
    assert event["engine"] == "marp_tracking"
    assert event["settings"] == settings
    assert event["ignored"] == {}
    assert isinstance(event["seq"], int)
