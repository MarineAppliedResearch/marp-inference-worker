# engine_registry.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Engine registry for the MARP Inference Worker.
# This file maps public engine names from model specs to engine implementations.
# Keep engine selection logic here so model management code does not need to
# import or branch across every supported runtime directly.

# BaseEngine defines the common interface returned by the registry.
from marp_inference_worker.engines.base_engine import BaseEngine

# MockEngine provides the first lightweight engine implementation.
from marp_inference_worker.engines.mock_engine import MockEngine

# UltralyticsEngine loads real Ultralytics-compatible YOLO model artifacts.
from marp_inference_worker.engines.ultralytics_engine import UltralyticsEngine


# Registered engine instances keyed by public engine name.
# TODO: Add CustomTorchEngine as it is implemented.
_ENGINES_BY_NAME: dict[str, BaseEngine] = {
    "mock": MockEngine(),
    "ultralytics": UltralyticsEngine(),
}


# get_engine()
# Returns the engine implementation for a public engine name.
# Inputs: engine name string from ModelSpec.engine.
# Output: BaseEngine implementation.
# Use this when model loading needs to dispatch to an engine-specific adapter.
def get_engine(engine_name: str) -> BaseEngine:

    # Normalize the engine name so API input casing does not affect lookup.
    normalized_engine_name = engine_name.lower()

    # Return the matching engine when it has been registered.
    if normalized_engine_name in _ENGINES_BY_NAME:
        return _ENGINES_BY_NAME[normalized_engine_name]

    # Fail clearly when a model requests an unsupported engine.
    raise ValueError(f"Unsupported inference engine: {engine_name}")