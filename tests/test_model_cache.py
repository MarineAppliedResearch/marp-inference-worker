# test_model_cache.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Tests for model cache path planning in the MARP Inference Worker.
# This file verifies deterministic cache keys and artifact paths before real
# model downloading and hash verification are implemented.

# Model cache utilities are tested directly because they define local paths.
from marp_inference_worker.models import model_cache

# ModelSpec validates representative model metadata used by cache planning.
from marp_inference_worker.models.model_spec import ModelSpec


# test_cache_key_uses_model_id_without_hash()
# Verifies that model ID is used as the cache key when no hash is available.
# Inputs: none.
# Output: pytest pass/fail result based on computed cache key.
# Use this to protect early cache behavior for models without hashes.
def test_cache_key_uses_model_id_without_hash() -> None:

    # Create a model spec that represents an artifact without a hash.
    model_spec = ModelSpec(
        model_id="demo_yolo",
        engine="ultralytics",
        model_arch="yolo11",
        task="detect",
        artifact={
            "url": "https://model-server/models/demo_yolo.pt",
            "format": "ultralytics_pt",
            "sha256": None,
        },
    )

    # Confirm the cache key falls back to the model ID.
    assert model_cache.get_cache_key(model_spec) == "demo_yolo"


# test_cache_key_uses_hash_when_present()
# Verifies that the artifact hash becomes part of the cache key when present.
# Inputs: none.
# Output: pytest pass/fail result based on computed cache key.
# Use this to prevent different model artifacts from colliding in cache.
def test_cache_key_uses_hash_when_present() -> None:

    # Create a model spec that includes a fake artifact hash.
    model_spec = ModelSpec(
        model_id="demo_yolo",
        engine="ultralytics",
        model_arch="yolo11",
        task="detect",
        artifact={
            "url": "https://model-server/models/demo_yolo.pt",
            "format": "ultralytics_pt",
            "sha256": "abc123",
        },
    )

    # Confirm the cache key includes both model identity and artifact hash.
    assert model_cache.get_cache_key(model_spec) == "demo_yolo_abc123"


# test_artifact_cache_path_uses_cache_root_key_and_filename()
# Verifies that artifact paths are built under models/cache.
# Inputs: none.
# Output: pytest pass/fail result based on computed path suffix.
# Use this to protect the local artifact cache layout.
def test_artifact_cache_path_uses_cache_root_key_and_filename() -> None:

    # Create a model spec with a URL filename that should become the local filename.
    model_spec = ModelSpec(
        model_id="demo_yolo",
        engine="ultralytics",
        model_arch="yolo11",
        task="detect",
        artifact={
            "url": "https://model-server/models/demo_yolo.pt",
            "format": "ultralytics_pt",
            "sha256": None,
        },
    )

    # Compute the local cache path for the artifact.
    artifact_path = model_cache.get_artifact_cache_path(model_spec)

    # Confirm the path points into the expected cache location.
    assert str(artifact_path).endswith("models\\cache\\demo_yolo\\demo_yolo.pt")