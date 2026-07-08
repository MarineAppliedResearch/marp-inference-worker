# mock_engine.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Mock inference engine for the MARP Inference Worker.
# This file provides a lightweight engine implementation for tests and early
# API development. It should not perform real inference or load real model
# runtimes.

# Path identifies the cached artifact file passed to the mock loader.
from pathlib import Path

# BaseEngine defines the shared interface all engines must implement.
from marp_inference_worker.engines.base_engine import BaseEngine

# ModelSpec defines the validated model metadata passed into engines.
from marp_inference_worker.models.model_spec import ModelSpec


# MockEngine
# Implements the engine interface without loading a real ML runtime.
# This class is useful for tests and early API contract development.
# Replace mock behavior with real engines when testing actual inference.
class MockEngine(BaseEngine):

    # engine_name
    # Returns the public engine name for the mock engine.
    # Inputs: none.
    # Output: mock engine identifier.
    # Use this for tests and placeholder model-loading flows.
    @property
    def engine_name(self) -> str:

        # Return the engine name used in ModelSpec.engine.
        return "mock"

    # load_model()
    # Pretends to load a cached model artifact.
    # Inputs: validated ModelSpec and local cached artifact path.
    # Output: dictionary representing a fake loaded model handle.
    # Use this until real engines are wired into the worker.
    def load_model(self, model_spec: ModelSpec, artifact_path: Path) -> dict[str, object]:

        # Return structured fake loaded-model metadata for tests and status views.
        return {
            "engine": self.engine_name,
            "model_id": model_spec.model_id,
            "artifact_path": str(artifact_path),
            "loaded": True,
        }