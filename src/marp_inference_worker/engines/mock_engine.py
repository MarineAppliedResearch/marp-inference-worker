# mock_engine.py
# Created: 2026-07-07
# Author: Isaac Travers
#
# Mock engine for the MARP Inference Worker.
# This engine implements the whole job contract without loading an ML runtime,
# so the job runner -- poll, lease, child process, heartbeat, cancellation,
# terminal report -- can be exercised on a machine with no GPU and no model.
# That is not a convenience: the runner is the layer most of this worker's
# behaviour lives in, and without this engine there would be no tier below a
# real GPU that could observe it at all.
#
# It must stay honest about the contract it implements. It reports progress,
# checks should_stop() and publishes a real artifact, because a mock that skips
# those would let a runner bug pass.

# json writes the mock result file.
import json

# time paces the mock work so a cancellation test has something to interrupt.
import time

# Path types the artifact path.
from pathlib import Path

# Any and Mapping type the spec and the returned metadata.
from typing import TYPE_CHECKING, Any, Mapping

# The engine contract and the frame-inference capability.
from marp_inference_worker.engines.base_engine import (
    BaseEngine,
    FrameInferenceCapable,
    JobContext,
    JobUnrunnable,
)


# How often, in frames, a durable metric event is emitted.
# The same cadence as the tracking engine's `_PROGRESS_EVERY_FRAMES`, and for the
# same reason: a metric event is a database row, and one per frame is 108,000
# rows for an hour of video at 30fps. This engine emitted one per frame until the
# first live round trip, where 46,000 frames of mock work put 38,000 rows in
# `gpu_job_events` -- which made the mock a misleading stand-in for the real
# engine as well as being wrong on its own terms.
#
# Progress and the stop check stay per frame. Neither is durable: the parent
# keeps only the latest progress, and `should_stop()` is a cached file check, so
# a cancellation still lands within one frame.
_METRICS_EVERY_FRAMES = 30

# ModelSpec types the frame-inference capability's load_model, and is imported
# only for type checking. Importing it at runtime is a cycle: the models package
# re-exports model_manager, which imports engine_registry, which imports this
# module. Importing this module first then failed on a partially initialized
# module -- the same breakage base_engine.py carries a note about.
if TYPE_CHECKING:
    from marp_inference_worker.models.model_spec import ModelSpec


# MockEngine
# A job engine that does no real inference.
# Useful for tests, for checking a new worker's connection to a coordinator
# before any model exists, and for exercising cancellation.
class MockEngine(BaseEngine, FrameInferenceCapable):

    # engine_name
    # Returns the public engine name for the mock engine.
    @property
    def engine_name(self) -> str:

        # Matches ModelSpec.engine and a job spec's `engine` field.
        return "mock"

    # describe()
    # Returns what this engine is.
    # Inputs: none.
    # Output: JSON-safe mapping for the worker's /status.
    def describe(self) -> dict[str, Any]:

        # Says plainly that it needs no GPU, so a coordinator can route to it
        # on a machine that has none.
        return {
            "engine": self.engine_name,
            "pipeline": "none",
            "requires_gpu": False,
            "purpose": "contract and runner testing; performs no inference",
        }

    # preflight()
    # Accepts any job that carries a frame range.
    # Inputs: the job spec mapping.
    # Output: none on success; raises JobUnrunnable otherwise.
    # The range check is real: a runner that hands down a malformed spec should
    # be caught by the mock too, not only by the engine that needs a GPU.
    def preflight(self, spec: Mapping[str, Any]) -> None:

        # Every spec carries a range, even for a whole video (R9).
        frame_range = spec.get("range")
        if not frame_range:
            raise JobUnrunnable("job spec carried no frame range")

    # run()
    # Walks the job's frame range without inferring anything.
    # Inputs: a JobContext and the job spec mapping.
    # Output: a JSON-safe summary, with a published artifact behind it.
    # Use this to test the runner. `frame_delay_s` in params slows it down so a
    # cancellation has a window to land in.
    def run(self, ctx: JobContext, spec: Mapping[str, Any]) -> dict[str, Any]:

        # Read the range the runner handed down.
        frame_range = spec["range"]
        start_frame = int(frame_range["start_frame"])
        end_frame = int(frame_range["end_frame"])
        total = end_frame - start_frame

        # Optional pacing, so a test can cancel a job that is still running.
        frame_delay_s = float(ctx.params.get("frame_delay_s", 0.0))

        # Results go to a real file, so publish_artifact() has real bytes and a
        # real hash to work with.
        results_path: Path = ctx.checkpoint_dir / "observations.jsonl"

        # Count what was actually done, so a stopped run reports honestly.
        frames_processed = 0
        stopped_early = False

        ctx.log(f"mock job over frames {start_frame}..{end_frame}")

        with results_path.open("w", encoding="utf-8") as results_file:
            for frame_index in range(start_frame, end_frame):

                # Stand in for the work.
                if frame_delay_s:
                    time.sleep(frame_delay_s)

                # One row per frame, so the artifact is not empty.
                results_file.write(
                    json.dumps({"frame": frame_index, "detections": []}) + "\n"
                )
                frames_processed += 1

                # Progress every frame: it is overwritten in place rather than
                # accumulated, so it costs one mapping however long the job is.
                ctx.report_progress(frames_processed, total, "frames")

                # Metrics on the real engine's cadence, because each one is a
                # durable row. The first frame is included so a short job still
                # reports something.
                if frames_processed == 1 or frames_processed % _METRICS_EVERY_FRAMES == 0:
                    ctx.report_metrics(
                        step=frame_index, phase="mock", metrics={"frames": frames_processed}
                    )

                # Cancellation has to be observable here or the runner's
                # cancellation path is untested.
                if ctx.should_stop():
                    ctx.log("stop requested", level="warning")
                    stopped_early = True
                    break

        # Hand the file over by hash, as a real engine does.
        results_sha256 = ctx.publish_artifact(results_path, "observations")

        return {
            "engine": self.engine_name,
            "frames_expected": total,
            "frames_processed": frames_processed,
            "stopped_early": stopped_early,
            "results": {
                "kind": "observations",
                "sha256": results_sha256,
                "path": str(results_path),
                "line_count": frames_processed,
            },
        }

    # load_model()
    # Pretends to load a cached model artifact.
    # Inputs: validated ModelSpec and local cached artifact path.
    # Output: dictionary representing a fake loaded model handle.
    def load_model(self, model_spec: "ModelSpec", artifact_path: Path) -> dict[str, object]:

        # Return structured fake loaded-model metadata for tests and status views.
        return {
            "engine": self.engine_name,
            "model_id": model_spec.model_id,
            "artifact_path": str(artifact_path),
            "loaded": True,
        }

    # infer_frame()
    # Returns an empty detection list.
    # Inputs: model ID, image source, confidence, label map, and image flag.
    # Output: an inference result with no detections.
    def infer_frame(
        self,
        model_id: str,
        image_source: str,
        confidence: float,
        class_names_by_id: dict[int, str],
        return_annotated_image: bool = False,
    ) -> dict[str, Any]:

        # Empty rather than fabricated: a mock that invented detections would
        # let a route test pass against a broken response shape.
        return {"detections": []}

    # infer_frame_image()
    # Refuses, because there is no image to annotate.
    # Inputs: model ID, image source, confidence, and label map.
    # Output: never returns; raises.
    def infer_frame_image(
        self,
        model_id: str,
        image_source: str,
        confidence: float,
        class_names_by_id: dict[int, str],
    ) -> bytes:

        # Returning empty bytes would look like a valid but blank image.
        raise NotImplementedError("the mock engine cannot render an annotated image")
