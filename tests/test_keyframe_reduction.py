# test_keyframe_reduction.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Tests for keyframe reduction in the MARP Inference Worker.
# The reduction is the step that turns a dense track into the handful of
# keyframes MARP stores, and it is a scientific rule rather than an
# implementation detail. These tests protect two different things: that the
# ported code is the code that was ported, and that a stored observation stays
# attributable to the rule that made it.

# ast compares the ported function's source against the legacy definition.
import ast

# inspect reads the ported function's source text.
import inspect

# Path locates the legacy script the reduction was ported from.
from pathlib import Path

# Pytest checks the expected refusal of an unknown reduction.
import pytest

# The reduction module under test.
from marp_inference_worker.reduction import keyframes


# The legacy script the live reduction was ported from.
_LEGACY_SCRIPT = (
    Path(__file__).resolve().parents[1] / "src" / "old_scripts" / "object_tracking_live.py"
)


# _strip_docstrings()
# Removes every docstring from a parsed tree, recursively.
# Inputs: a parsed ast node.
# Output: the same node, with docstring statements removed in place.
#
# Needed because the ported code carries this repository's `#` comment style
# instead of the legacy file's docstrings, and a docstring is a real statement
# in the tree. Without this the comparison below asserts prose style rather than
# arithmetic, which is not what "verbatim" is protecting.
def _strip_docstrings(node: ast.AST) -> ast.AST:

    # Every scope that can hold a docstring.
    for child in ast.walk(node):
        if not isinstance(child, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        # A docstring is a bare string expression in the first statement slot.
        body = getattr(child, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                # Leave a `pass` if the docstring was the whole body, so the
                # result stays parseable.
                child.body = body[1:] or [ast.Pass()]

    return node


# _normalized_source()
# Returns a function's source as a normalized, docstring-free syntax tree dump.
# Inputs: the source text of exactly one function definition.
# Output: normalized source text.
# Use this on both sides of the comparison so only the code is compared.
def _normalized_source(source: str) -> str:

    # dedent, because inspect.getsource of a module-level function is already
    # flush left but this keeps the helper safe either way.
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    return ast.unparse(_strip_docstrings(tree)).strip()


# _legacy_function_source()
# Returns the source of the LAST definition of a named function in the legacy file.
# Inputs: the function name.
# Output: the source text of its final definition.
#
# The last definition, deliberately. object_tracking_live.py defines
# reduce_to_keyframes_v3_dirpad twice, at lines 628 and 814, and
# _apply_directional_pad twice, at 780 and 930. Python keeps the later one, so
# the later one is what the live path actually ran -- and they differ
# numerically (vel_window 3 vs 2, speed_gain 0.25 vs 0.35). Picking the first
# match here would make this test assert the wrong reference.
def _legacy_function_source(name: str) -> str:

    source = _LEGACY_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Collect every top-level definition of that name, in file order.
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert definitions, f"{name} not found in {_LEGACY_SCRIPT}"

    # The last one wins, as it does at import time. Normalized the same way as
    # the ported side, so only the code is compared.
    return _normalized_source(ast.unparse(definitions[-1]))


# test_ported_reduction_is_the_line_814_definition()
# Verifies the ported reduction is source-identical to the legacy live one.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R10b's "verbatim" clause, and it is the only test that can. Comparing
# outputs on sample tracks would pass for a reduction that had been subtly
# rewritten; comparing the normalized syntax tree catches a changed constant, a
# reordered branch or a merged variant.
#
# Docstrings and comments are excluded, which is the right granularity: this
# asserts the maths is unchanged, not the prose around it. The ported code
# carries this repository's `#` comment style rather than the legacy
# docstrings, and that difference is mandated, not accidental.
def test_ported_reduction_is_the_line_814_definition() -> None:

    ported = _normalized_source(inspect.getsource(keyframes.reduce_to_keyframes_v3_dirpad))
    legacy = _legacy_function_source("reduce_to_keyframes_v3_dirpad")

    assert ported == legacy


# test_ported_directional_pad_is_the_line_930_definition()
# Verifies the ported padding helper is source-identical to the legacy live one.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves the same clause for the other half of the pair. The two definitions in
# the legacy file are identical in this case, but asserting it stops a future
# edit here from drifting silently.
def test_ported_directional_pad_is_the_line_930_definition() -> None:

    ported = _normalized_source(inspect.getsource(keyframes._apply_directional_pad))
    legacy = _legacy_function_source("_apply_directional_pad")

    assert ported == legacy


# test_reduction_defaults_match_the_live_definition()
# Verifies the reduction's default parameters.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves R10b's numbers specifically. These are the values that distinguish the
# line-814 definition from the line-628 one, so this test fails loudly if the
# wrong variant is ever wired in -- which is the mistake the investigation found
# and this exists to prevent recurring.
def test_reduction_defaults_match_the_live_definition() -> None:

    signature = inspect.signature(keyframes.reduce_to_keyframes_v3_dirpad)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }

    # The line-814 values. vel_window 2 and speed_gain 0.35 would be line 628.
    assert defaults == {
        "pos_thresh": 0.005,
        "size_thresh": 0.01,
        "vel_window": 3,
        "speed_gain": 0.25,
        "base_pad": 0.00,
        "max_pad": 0.12,
    }


# _straight_line_track()
# Builds a track moving steadily across the frame.
# Inputs: how many frames, and the frame rate to derive times from.
# Output: an ended-track mapping in the shape the reduction reads.
# Use this as the fixture for behavioural assertions below.
def _straight_line_track(frame_count: int = 60, frame_rate: float = 30.0) -> dict:

    return {
        "class_name": "Rockfish",
        "frames": [
            {
                "frame": index,
                "time": index / frame_rate,
                # Centre moves left to right and down; size held constant, so a
                # straight-line track is genuinely reducible to two keyframes.
                "bbox": (0.1 + index * 0.01, 0.1 + index * 0.005, 0.05, 0.05),
                "confidence": 0.9,
            }
            for index in range(frame_count)
        ],
    }


# test_reduction_labels_keyframes_start_middle_end()
# Verifies the keyframe type labels.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves R10's final clause: a finished track is reduced to keyframes labelled
# start/middle/end, which is the shape MARP's review tooling reads.
def test_reduction_labels_keyframes_start_middle_end() -> None:

    reduced = keyframes.reduce_to_keyframes_v3_dirpad(_straight_line_track())

    # At least a start and an end.
    assert len(reduced) >= 2
    assert reduced[0]["type"] == "start"
    assert reduced[-1]["type"] == "end"

    # Anything between them is a middle, and nothing else is.
    for keyframe in reduced[1:-1]:
        assert keyframe["type"] == "middle"

    # Every keyframe carries the species name and a frame number from the track.
    track_frames = {frame["frame"] for frame in _straight_line_track()["frames"]}
    for keyframe in reduced:
        assert keyframe["comname"] == "Rockfish"
        assert keyframe["framenum"] in track_frames


# test_reduction_actually_reduces()
# Verifies that a straight-line track collapses to far fewer keyframes.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves the reduction does its job: sixty frames of steady motion is not sixty
# keyframes. Without this, a reduction that returned its input unchanged would
# satisfy every other assertion in this file.
def test_reduction_actually_reduces() -> None:

    track = _straight_line_track(frame_count=60)
    reduced = keyframes.reduce_to_keyframes_v3_dirpad(track)

    assert len(reduced) < len(track["frames"])


# test_registry_returns_the_named_versioned_reduction()
# Verifies the reduction registry.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves R10b's "named, versioned step": the rule is chosen by name and version
# from the job spec, not hard-wired.
def test_registry_returns_the_named_versioned_reduction() -> None:

    reduce = keyframes.get_reduction("v3_dirpad", "1")
    assert reduce is keyframes.reduce_to_keyframes_v3_dirpad


# test_registry_refuses_an_unknown_reduction()
# Verifies that an unknown reduction is refused rather than substituted.
# Inputs: none.
# Output: pytest pass/fail result.
#
# Proves the other half of R10b, and it is the half that protects the data. If
# an unknown name fell back to v3_dirpad, an observation would be recorded as
# having been produced by a rule that did not produce it -- which is exactly the
# attribution R10b exists to keep.
def test_registry_refuses_an_unknown_reduction() -> None:

    with pytest.raises(KeyError):
        keyframes.get_reduction("v4_experimental", "1")

    # A known name at an unknown version is refused too; a version is part of
    # the identity, not a hint.
    with pytest.raises(KeyError):
        keyframes.get_reduction("v3_dirpad", "2")


# test_available_reductions_lists_what_can_be_applied()
# Verifies the list reported in capabilities.
# Inputs: none.
# Output: pytest pass/fail result.
# Proves the worker can tell a coordinator which rules it has, so a job naming
# an unavailable rule need never be dispatched here at all.
def test_available_reductions_lists_what_can_be_applied() -> None:

    assert keyframes.available_reductions() == [{"name": "v3_dirpad", "version": "1"}]
