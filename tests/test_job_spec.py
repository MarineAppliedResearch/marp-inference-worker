# test_job_spec.py
# Created: 2026-10-15
# Author: Isaac Travers
#
# Job spec validation tests for the MARP Inference Worker.
# The spec is the contract with the coordinator, and it now describes two shapes
# of work rather than one. These tests hold both ends of that: a training job
# can be expressed at all, and an inference job is still refused when it is
# incomplete -- which is the half that a looser schema would have quietly lost.
#
# Pure validation. No engine, no coordinator, no network.

# pytest provides the raises helper the refusal cases need.
import pytest

# ValidationError is what pydantic raises out of model_validate.
from pydantic import ValidationError

# The models under test.
from marp_inference_worker.jobs.job_spec import JobSpec


# An inference job, complete, as the coordinator sends one.
INFERENCE = {
    "engine": "marp_tracking",
    "model": {"name": "rockfish.pt", "sha256": "a" * 64},
    "video": {"url": "http://media.invalid/v.mp4", "source_name": "v.mp4"},
    "range": {"start_frame": 0, "end_frame": 1000},
    "reduction": {"name": "v3_dirpad", "version": 1},
}


# A training job: a dataset to learn from, and weights to start from.
TRAINING = {
    "engine": "marp_training",
    "model": {"name": "yolo11m.pt", "sha256": "b" * 64},
    "dataset": {"url": "http://marp.invalid/datasets/7/archive",
                "sha256": "c" * 64,
                "name": "CAMPA_2024_Rockfish_v4",
                "dataset_id": 7},
    "params": {"epochs": 200, "imgsz": 640},
}


# test_an_inference_job_still_validates()
# Verifies the existing shape is untouched.
# Inputs: none.
# Output: pytest pass/fail result.
def test_an_inference_job_still_validates() -> None:

    spec = JobSpec.model_validate(INFERENCE)

    assert spec.video is not None
    assert spec.range.end_frame == 1000
    assert spec.dataset is None


# test_a_training_job_needs_no_video_or_range()
# Verifies a training job can be expressed at all.
# Inputs: none.
# Output: pytest pass/fail result.
#
# `video` and `range` were required fields, so this payload could not be
# validated at all before -- which is what made training impossible to express
# rather than merely unimplemented.
def test_a_training_job_needs_no_video_or_range() -> None:

    spec = JobSpec.model_validate(TRAINING)

    assert spec.dataset.dataset_id == 7
    assert spec.video is None
    assert spec.range is None
    assert spec.reduction is None
    assert spec.params["epochs"] == 200


# test_an_inference_job_is_still_refused_when_incomplete()
# Verifies the optional fields did not loosen the inference contract.
# Inputs: none.
# Output: pytest pass/fail result.
#
# This is the test that matters most. Making `video`, `range` and `reduction`
# optional so a training job could exist would, on its own, let a half-built
# inference lease through to fail inside a child process as something that reads
# like an engine fault.
@pytest.mark.parametrize("missing", ["video", "range", "reduction"])
def test_an_inference_job_is_still_refused_when_incomplete(missing: str) -> None:

    payload = {key: value for key, value in INFERENCE.items() if key != missing}

    with pytest.raises(ValidationError) as refused:
        JobSpec.model_validate(payload)

    assert missing in str(refused.value)


# test_a_job_cannot_be_both_shapes()
# Verifies an incoherent spec is named rather than silently resolved.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Picking one of the two would mean a coordinator bug ran as a job rather than
# failing at the boundary, and whichever was picked would be wrong half the time.
def test_a_job_cannot_be_both_shapes() -> None:

    payload = {**TRAINING, "video": INFERENCE["video"], "range": INFERENCE["range"]}

    with pytest.raises(ValidationError) as refused:
        JobSpec.model_validate(payload)

    assert "dataset" in str(refused.value)


# test_a_dataset_must_say_where_it_is()
# Verifies an empty locator is refused at the boundary.
# Inputs: none.
# Output: pytest pass/fail result.
def test_a_dataset_must_say_where_it_is() -> None:

    payload = {**TRAINING, "dataset": {**TRAINING["dataset"], "url": ""}}

    with pytest.raises(ValidationError):
        JobSpec.model_validate(payload)
