# ultralytics_engine.py
# Created: 2026-07-08
# Author: Isaac Travers
#
# Ultralytics engine adapter for the MARP Inference Worker.
# This file loads Ultralytics-compatible YOLO model artifacts and runs prediction,
# either on one image for the worker's own frame routes or as a generator over a
# stream of decoded frames for job work.
# All Ultralytics-specific API usage belongs here. Video decoding, tracking and
# keyframe reduction do not -- they are separate stages with their own modules.

# Path identifies the cached model artifact supplied by the model manager.
from pathlib import Path

# Any and Iterator type the prediction results and the frame generator.
from typing import TYPE_CHECKING, Any, Iterable, Iterator, NamedTuple

# The frame-inference capability declares this engine can serve the frame routes.
from marp_inference_worker.engines.base_engine import FrameInferenceCapable

# ModelSpec types the frame-inference capability's load_model, and is imported
# only for type checking. Importing it at runtime is a cycle: the models package
# re-exports model_manager, which imports engine_registry, which imports this
# module. Importing this module first then failed on a partially initialized
# module -- the same breakage base_engine.py carries a note about.
if TYPE_CHECKING:
    from marp_inference_worker.models.model_spec import ModelSpec

# DecodedFrame is what the frame-range reader yields into infer_stream().
from marp_inference_worker.media.frame_range_reader import DecodedFrame

# Frame source utilities prepare local paths and URL images for prediction.
from marp_inference_worker.inputs.frame_source import prepare_frame_source

# Detection renderer draws normalized boxes and labels onto frame images.
from marp_inference_worker.rendering.detection_renderer import render_detections_to_image_bytes


# FrameDetections
# The detections found on one frame, with that frame's identity kept alongside.
# Carrying the frame back out means a downstream tracker never has to assume the
# generator and its own counter stayed in step.
class FrameDetections(NamedTuple):

    # The frame these detections came from.
    frame: DecodedFrame

    # Pixel xyxy boxes with class id and confidence, one row per detection.
    detections: list[dict[str, Any]]


# UltralyticsEngine
# Loads Ultralytics YOLO models and runs prediction against them.
# One instance owns its loaded handles, and a job's child process gets its own
# instance, so there is no shared model state to unload or lock across jobs.
class UltralyticsEngine(FrameInferenceCapable):

    # __init__()
    # Initializes the engine-local loaded model handle registry.
    # Inputs: none.
    # Output: initialized UltralyticsEngine instance.
    def __init__(self) -> None:

        # Real YOLO handles stay inside the engine so callers stay generic.
        self._loaded_models_by_id: dict[str, Any] = {}

    # engine_name
    # Returns the public engine name handled by this engine implementation.
    # Inputs: none.
    # Output: engine name string.
    @property
    def engine_name(self) -> str:

        # Match the public engine name accepted by ModelSpec.engine.
        return "ultralytics"

    # ultralytics_version()
    # Returns the installed Ultralytics version, or None when it is absent.
    # Inputs: none.
    # Output: version string or None.
    # Use this in a describe() so /status reports the real pinned version rather
    # than a literal that goes stale the next time the pin moves.
    @staticmethod
    def ultralytics_version() -> str | None:

        # Absence is a legitimate state on a machine that only serves the API.
        try:
            import ultralytics

            return str(ultralytics.__version__)
        except Exception:
            return None

    # load_model()
    # Loads a cached Ultralytics model artifact into memory.
    # Inputs: validated ModelSpec and local cached artifact path.
    # Output: API-safe metadata describing the loaded model.
    def load_model(self, model_spec: "ModelSpec", artifact_path: Path) -> dict[str, Any]:

        # Import Ultralytics lazily so registry tests do not load the ML stack.
        from ultralytics import YOLO

        # Fail clearly if the cached artifact path does not exist.
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Cached model artifact not found: {artifact_path}")

        # Load the YOLO model from the local cached artifact path.
        yolo_model = YOLO(str(artifact_path))

        # Store the real model handle for future inference calls.
        self._loaded_models_by_id[model_spec.model_id] = yolo_model

        # Return only JSON-safe metadata to the API layer.
        return {
            "engine_name": self.engine_name,
            "model_id": model_spec.model_id,
            "model_loaded": True,
            "artifact_path": str(artifact_path),
            "loaded_model_type": type(yolo_model).__name__,
            "loaded_model_count": len(self._loaded_models_by_id),
        }

    # load_weights()
    # Loads a YOLO model from a path without registering it under a model id.
    # Inputs: artifact path and the device string to place the model on.
    # Output: the loaded YOLO handle.
    # Use this from job engines, which own exactly one model for the duration of
    # the job and do not need the id-keyed registry the frame routes use.
    def load_weights(self, artifact_path: Path, device: str) -> Any:

        # Imported lazily for the same reason load_model() does.
        from ultralytics import YOLO

        # Fail clearly if the cached artifact path does not exist.
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Cached model artifact not found: {artifact_path}")

        # Place the model explicitly. The caller has already resolved "auto" to
        # a real device, so nothing here can silently land on CPU.
        return YOLO(str(artifact_path)).to(device)

    # unload_all()
    # Drops every model handle this engine holds.
    # Inputs: none.
    # Output: none.
    # Use this at the end of a job so GPU memory is released before the child
    # process exits, rather than relying on interpreter teardown ordering.
    def unload_all(self) -> None:

        # Drop the Python references first so the CUDA allocator can reclaim.
        self._loaded_models_by_id.clear()

        # Empty the allocator cache when torch is present and CUDA is in use.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # A machine without torch or CUDA has nothing to release.
            pass

    # get_loaded_model()
    # Returns a loaded Ultralytics model handle by MARP model ID.
    # Inputs: model ID string.
    # Output: loaded YOLO model object.
    def get_loaded_model(self, model_id: str) -> Any:

        # Fail clearly if inference is requested before a model is loaded.
        if model_id not in self._loaded_models_by_id:
            raise ValueError(f"Ultralytics model is not loaded: {model_id}")

        # Return the raw model handle for engine-internal inference use.
        return self._loaded_models_by_id[model_id]

    # infer_stream()
    # Runs detection over a stream of decoded frames, one frame at a time.
    # Inputs: a loaded YOLO handle, an iterable of DecodedFrame, and confidence.
    # Output: iterator of FrameDetections.
    # Use this for all job inference. It is a generator on purpose: Ultralytics'
    # own predict() with stream=False accumulates every Results object for the
    # whole input, which for a ten-hour video is the whole video in RAM. Feeding
    # single frames and yielding each result keeps one frame alive at a time (R8).
    def infer_stream(
        self,
        yolo_model: Any,
        frames: Iterable[DecodedFrame],
        confidence: float,
    ) -> Iterator[FrameDetections]:

        # inference_mode is cheaper than no_grad and is what the live script used.
        import torch

        # Walk the frame generator; nothing before the current frame is retained.
        for frame in frames:

            # Predict on the single decoded array. verbose=False keeps
            # Ultralytics from printing a line per frame.
            with torch.inference_mode():
                results = yolo_model.predict(
                    source=frame.image,
                    conf=confidence,
                    verbose=False,
                )

            # An empty result list would be an Ultralytics fault, not an empty
            # frame; an empty frame still returns one Results with no boxes.
            if not results:
                raise RuntimeError(f"Ultralytics returned no result for frame {frame.index}")

            # Normalize this frame's boxes into plain Python values, so nothing
            # downstream holds a tensor that pins GPU memory.
            yield FrameDetections(
                frame=frame,
                detections=self._normalize_boxes(results[0]),
            )

    # _normalize_boxes()
    # Converts one Ultralytics Results object into plain detection dictionaries.
    # Inputs: one Ultralytics Results object.
    # Output: list of detections with pixel xyxy, class id and confidence.
    # Use this so tensors never escape this module.
    @staticmethod
    def _normalize_boxes(result: Any) -> list[dict[str, Any]]:

        # A frame with nothing on it has boxes set to None on some tasks.
        if result.boxes is None:
            return []

        # Pull the three arrays once rather than indexing per box, which is what
        # made the legacy script's inner loop quadratic in detections.
        xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        # Build one plain dictionary per detection.
        detections: list[dict[str, Any]] = []
        for row_index in range(len(xyxy)):
            detections.append(
                {
                    "bbox_xyxy": [float(value) for value in xyxy[row_index]],
                    "confidence": float(confidences[row_index]),
                    "class_id": int(class_ids[row_index]),
                }
            )
        return detections

    # _run_frame_prediction()
    # Runs YOLO prediction and normalizes detections for one visual frame.
    # Inputs: model ID, image source, confidence threshold, and class-name map.
    # Output: dictionary containing detections and source image array.
    # Use this so the JSON and image frame routes share one prediction path.
    def _run_frame_prediction(
        self,
        model_id: str,
        image_source: str,
        confidence: float,
        class_names_by_id: dict[int, str],
    ) -> dict[str, Any]:

        # Get the already-loaded YOLO model handle for this model ID.
        yolo_model = self.get_loaded_model(model_id)

        # Prepare local paths and URL images through the shared frame-source utility.
        prepared_source = prepare_frame_source(image_source)

        try:
            # Run Ultralytics prediction on the prepared image source.
            results = yolo_model.predict(
                source=prepared_source.prediction_source,
                conf=confidence,
                verbose=False,
            )

            # Fail clearly if Ultralytics returned no result object.
            if not results:
                raise ValueError("Ultralytics returned no inference results.")

            # Normalized detections returned to callers.
            detections: list[dict[str, Any]] = []

            # Ultralytics returns one result object per input image.
            for result in results:

                # Reuse the shared normalizer, then add the fields the frame
                # routes publish on top of it.
                for detection in self._normalize_boxes(result):

                    # Prefer labels from the model spec, then fall back to
                    # Ultralytics' own names.
                    class_id = detection["class_id"]
                    class_name = class_names_by_id.get(
                        class_id,
                        yolo_model.names.get(class_id, str(class_id)),
                    )

                    # The frame routes also publish normalized coordinates, so
                    # a caller can draw without knowing the frame size.
                    x1, y1, x2, y2 = detection["bbox_xyxy"]
                    frame_height, frame_width = result.orig_shape[0], result.orig_shape[1]

                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": detection["confidence"],
                            "bbox_xyxy": detection["bbox_xyxy"],
                            "bbox_xyxyn": [
                                x1 / frame_width,
                                y1 / frame_height,
                                x2 / frame_width,
                                y2 / frame_height,
                            ],
                        }
                    )

            # Return normalized detections and the original image for rendering.
            return {
                "detections": detections,
                "source_image": results[0].orig_img,
            }

        finally:
            # Clean up temporary URL downloads after prediction finishes.
            prepared_source.cleanup()

    # infer_frame()
    # Runs object detection on one image source using a loaded YOLO model.
    # Inputs: model ID, image source, confidence, label map, and image flag.
    # Output: normalized detections and optional base64 annotated image.
    def infer_frame(
        self,
        model_id: str,
        image_source: str,
        confidence: float,
        class_names_by_id: dict[int, str],
        return_annotated_image: bool = False,
    ) -> dict[str, Any]:

        # Run prediction through the shared frame helper.
        prediction_result = self._run_frame_prediction(
            model_id=model_id,
            image_source=image_source,
            confidence=confidence,
            class_names_by_id=class_names_by_id,
        )

        # Build the base inference result shared by all frame requests.
        inference_result: dict[str, Any] = {
            "detections": prediction_result["detections"],
        }

        # Optionally add a base64-encoded annotated image to the JSON result.
        if return_annotated_image:

            # Base64 is used so the annotated image can travel inside JSON.
            import base64

            # Render the source image with normalized detections.
            annotated_image_bytes = render_detections_to_image_bytes(
                source_image=prediction_result["source_image"],
                detections=prediction_result["detections"],
                output_format="jpg",
            )

            # Add image metadata and payload to the inference result.
            inference_result["annotated_image_format"] = "jpg"
            inference_result["annotated_image_base64"] = base64.b64encode(
                annotated_image_bytes
            ).decode("utf-8")

        # Return detections and optional annotated image data.
        return inference_result

    # infer_frame_image()
    # Runs inference and returns an annotated image.
    # Inputs: model ID, image source, confidence threshold, and class-name map.
    # Output: encoded image bytes.
    def infer_frame_image(
        self,
        model_id: str,
        image_source: str,
        confidence: float,
        class_names_by_id: dict[int, str],
    ) -> bytes:

        # Run prediction through the shared frame helper.
        prediction_result = self._run_frame_prediction(
            model_id=model_id,
            image_source=image_source,
            confidence=confidence,
            class_names_by_id=class_names_by_id,
        )

        # Render the source image with normalized detections.
        return render_detections_to_image_bytes(
            source_image=prediction_result["source_image"],
            detections=prediction_result["detections"],
            output_format="jpg",
        )
