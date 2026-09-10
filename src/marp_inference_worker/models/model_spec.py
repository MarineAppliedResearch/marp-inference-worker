# model_spec.py
# Created: 2026-07-11
# Author: Isaac Travers
#
# Model schema definitions for the MARP Inference Worker.
# This file defines the public model-load request contract used by API routes,
# model caching, and engine adapters. Keep model metadata validation here.

# Pydantic provides request/response validation models.
from pydantic import BaseModel, Field


# ArtifactSpec
# Describes where a model artifact is located and how it should be interpreted.
# Inputs: artifact URL/path, format, and optional artifact hash.
# Output: validated artifact object inside ModelSpec.
# Use this to separate transport metadata from runtime model settings.
class ArtifactSpec(BaseModel):

    # Source URL or local path for this model artifact.
    url: str

    # Artifact format name used by loader code.
    format: str

    # Optional immutable hash used for cache identity.
    sha256: str | None = None


# LoadSettings
# Optional runtime settings used when loading a model into an engine.
# Inputs: device, precision, and optional provider-specific options.
# Output: validated load-settings object.
# Use this so load-time options are grouped under one schema field.
class LoadSettings(BaseModel):

    # Optional runtime device hint such as auto, cpu, or cuda:0.
    device: str | None = None


# LabelSpec
# Defines one class label mapping for model output IDs.
# Inputs: class ID, display name, and optional external identifier.
# Output: validated label mapping object.
# Use this when the coordinator needs explicit class-name control.
class LabelSpec(BaseModel):

    # Numeric class identifier emitted by an inference engine.
    class_id: int

    # Human-readable class name for this class ID.
    class_name: str

    # Optional external ID for cross-system class mapping.
    external_id: str | None = None


# ModelSpec
# Public request schema for loading a model into the worker.
# Inputs: model identity, engine, artifact metadata, and optional settings.
# Output: validated model-load contract object.
# Use this as the single source of truth for model-load API validation.
class ModelSpec(BaseModel):

    # Stable identifier for this model in the worker runtime.
    model_id: str

    # Public engine name used by the engine registry.
    engine: str

    # Model architecture descriptor from the coordinator.
    model_arch: str

    # Task type such as detect, segment, or classify.
    task: str

    # Artifact locator and format details.
    artifact: ArtifactSpec

    # Optional load-time settings for the chosen engine.
    load_settings: LoadSettings | None = None

    # Optional class-name mappings provided by coordinator metadata.
    labels: list[LabelSpec] = Field(default_factory=list)
