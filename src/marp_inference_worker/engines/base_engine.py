# base_engine.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Base engine interface for the MARP Inference Worker.
# This file defines the common behavior expected from all inference engines.
# Engine-specific modules should implement this interface so model management
# code can load and run models without knowing engine-specific details.

# ABC tools define an abstract base class for all inference engines.
from abc import ABC, abstractmethod

# Path identifies the cached artifact file that should be loaded by an engine.
from pathlib import Path

# ModelSpec defines the validated model metadata passed into engines.
from marp_inference_worker.models.model_spec import ModelSpec


# BaseEngine
# Defines the shared interface used by all model runtime engines.
# Engines should load cached artifacts and return engine-specific model handles.
# Keep this class small so Ultralytics, custom Torch, and future engines can fit it.
class BaseEngine(ABC):

    # engine_name
    # Returns the public engine name handled by this engine implementation.
    # Inputs: none.
    # Output: engine name string used in ModelSpec.engine.
    # Use this so the registry can match model specs to engine classes.
    @property
    @abstractmethod
    def engine_name(self) -> str:

        # Subclasses must define their public engine name.
        raise NotImplementedError

    # load_model()
    # Loads a cached model artifact into this engine runtime.
    # Inputs: validated ModelSpec and local cached artifact path.
    # Output: engine-specific loaded model handle or metadata object.
    # Use this when /models/load needs to prepare a model for inference.
    @abstractmethod
    def load_model(self, model_spec: ModelSpec, artifact_path: Path) -> object:

        # Subclasses must implement actual model-loading behavior.
        raise NotImplementedError