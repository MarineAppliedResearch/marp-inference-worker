# test_engine_registry.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Tests for the MARP Inference Worker engine registry.
# This file verifies that public engine names resolve to the correct engine
# implementations and unsupported engines fail clearly.

# Pytest checks expected exceptions from unsupported engine lookups.
import pytest

# Engine registry is tested directly because it controls runtime selection.
from marp_inference_worker.engines import engine_registry


# test_engine_registry_returns_mock_engine()
# Verifies that the mock engine can be resolved by name.
# Inputs: none.
# Output: pytest pass/fail result based on returned engine name.
# Use this to protect the first registered engine contract.
def test_engine_registry_returns_mock_engine() -> None:

    # Resolve the mock engine through the public registry.
    engine = engine_registry.get_engine("mock")

    # Confirm the returned engine reports the expected public name.
    assert engine.engine_name == "mock"


# test_engine_registry_returns_ultralytics_engine()
# Verifies that the Ultralytics engine can be resolved by name.
# Inputs: none.
# Output: pytest pass/fail result based on returned engine name.
# Use this to protect the real YOLO engine registry contract.
def test_engine_registry_returns_ultralytics_engine() -> None:

    # Resolve the Ultralytics engine through the public registry.
    engine = engine_registry.get_engine("ultralytics")

    # Confirm the returned engine reports the expected public name.
    assert engine.engine_name == "ultralytics"
    

# test_engine_registry_rejects_unknown_engine()
# Verifies that unsupported engine names fail clearly.
# Inputs: none.
# Output: pytest pass/fail result based on raised ValueError.
# Use this to avoid silently accepting unsupported model runtimes.
def test_engine_registry_rejects_unknown_engine() -> None:

    # Confirm an unknown engine raises a clear error.
    with pytest.raises(ValueError):
        engine_registry.get_engine("not_real")