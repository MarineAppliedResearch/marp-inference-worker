# model_manager.py
# Created: 2026-07-11
# Author: Isaac Travers
#
# Model management service for the MARP Inference Worker.
# This file tracks loaded model specs and delegates cache + engine operations
# to dedicated modules. Keep route handlers thin by centralizing model/runtime
# behavior in this manager.

# Path helps pass normalized local artifact paths to engines.
from pathlib import Path

# Engine registry resolves model engine names to engine implementations.
from marp_inference_worker.engines import engine_registry

# Model cache ensures artifacts are available on local disk.
from marp_inference_worker.models import model_cache

# ModelSpec validates model metadata and label mappings.
from marp_inference_worker.models.model_spec import ModelSpec


# In-memory loaded model specs indexed by model ID.
_loaded_model_specs_by_id: dict[str, ModelSpec] = {}


# In-memory JSON-safe loaded model payloads used by list endpoints.
_loaded_model_payloads_by_id: dict[str, dict[str, object]] = {}


# clear_loaded_models()
# Clears loaded-model state for tests and local development resets.
# Inputs: none.
# Output: none.
# Use this in tests so model-loading state does not leak between cases.
def clear_loaded_models() -> None:

    # Reset all in-memory loaded model state.
    _loaded_model_specs_by_id.clear()
    _loaded_model_payloads_by_id.clear()


# get_loaded_models()
# Returns JSON-safe loaded model payloads for API responses.
# Inputs: none.
# Output: list of model dictionaries.
# Use this from status/model routes to report current worker state.
def get_loaded_models() -> list[dict[str, object]]:

    # Preserve insertion order so responses are stable in tests.
    return list(_loaded_model_payloads_by_id.values())


# _class_names_by_id_from_spec()
# Builds class-name mappings from optional model label metadata.
# Inputs: validated ModelSpec object.
# Output: dictionary keyed by class ID.
# Use this so engines can prefer coordinator-provided class names.
def _class_names_by_id_from_spec(model_spec: ModelSpec) -> dict[int, str]:

    # Convert optional labels list into class-name lookup map.
    class_names_by_id: dict[int, str] = {}
    for label in model_spec.labels:
        class_names_by_id[label.class_id] = label.class_name
    return class_names_by_id


# load_model()
# Caches and loads a model through the configured runtime engine.
# Inputs: validated ModelSpec object.
# Output: dictionary with model payload, cache state, and engine metadata.
# Use this as the service entry point for POST /models/load.
def load_model(model_spec: ModelSpec) -> dict[str, object]:

    # Ensure the model artifact is present in local cache storage.
    cache_state = model_cache.ensure_artifact_cached(model_spec)

    # Resolve the engine implementation for this model spec. Loading a model
    # for the frame routes needs the frame capability, not just any engine.
    engine = engine_registry.get_frame_engine(model_spec.engine)

    # Ask the engine to load from the local cached artifact path.
    engine_state = engine.load_model(
        model_spec=model_spec,
        artifact_path=Path(str(cache_state["artifact_path"])),
    )

    # Store validated model metadata for later inference dispatch.
    _loaded_model_specs_by_id[model_spec.model_id] = model_spec

    # Store JSON-safe model payload for API list responses.
    model_payload = model_spec.model_dump(mode="json")
    _loaded_model_payloads_by_id[model_spec.model_id] = model_payload

    # Return API-safe load result payload.
    return {
        "model": model_payload,
        "cache": cache_state,
        "engine": engine_state,
    }


# infer_frame()
# Runs model inference on one image source using a loaded model.
# Inputs: model ID, image source, confidence threshold, and image return flag.
# Output: normalized inference result dictionary.
# Use this as the service entry point for JSON frame-inference routes.
def infer_frame(
    model_id: str,
    image_source: str,
    confidence: float,
    return_annotated_image: bool = False,
) -> dict[str, object]:

    # Fail clearly when inference is requested before model load.
    if model_id not in _loaded_model_specs_by_id:
        raise ValueError(f"Model is not loaded: {model_id}")

    # Resolve engine and class labels from stored model metadata.
    # get_frame_engine() checks the capability by class and raises with the
    # engine name when it is absent, so no method-name probing is needed here.
    model_spec = _loaded_model_specs_by_id[model_id]
    engine = engine_registry.get_frame_engine(model_spec.engine)
    class_names_by_id = _class_names_by_id_from_spec(model_spec)

    # Delegate frame inference to the selected engine implementation.
    return engine.infer_frame(
        model_id=model_id,
        image_source=image_source,
        confidence=confidence,
        class_names_by_id=class_names_by_id,
        return_annotated_image=return_annotated_image,
    )


# infer_frame_image()
# Runs model inference and returns an annotated image byte payload.
# Inputs: model ID, image source, and confidence threshold.
# Output: encoded image bytes.
# Use this as the service entry point for binary frame-image routes.
def infer_frame_image(
    model_id: str,
    image_source: str,
    confidence: float,
) -> bytes:

    # Fail clearly when inference is requested before model load.
    if model_id not in _loaded_model_specs_by_id:
        raise ValueError(f"Model is not loaded: {model_id}")

    # Resolve engine and class labels from stored model metadata.
    # get_frame_engine() checks the capability by class and raises with the
    # engine name when it is absent, so no method-name probing is needed here.
    model_spec = _loaded_model_specs_by_id[model_id]
    engine = engine_registry.get_frame_engine(model_spec.engine)
    class_names_by_id = _class_names_by_id_from_spec(model_spec)

    # Delegate image inference to the selected engine implementation.
    return engine.infer_frame_image(
        model_id=model_id,
        image_source=image_source,
        confidence=confidence,
        class_names_by_id=class_names_by_id,
    )
