# ultralytics_engine.py
# Created: 2026-07-08
# Author: Isaac Travers
#
# Ultralytics engine adapter for the MARP Inference Worker.
# This file loads Ultralytics-compatible YOLO model artifacts from the local
# model cache and keeps the loaded model handles available for later inference.
# Engine-specific Ultralytics loading, prediction, and tracking code belongs here.

# Path identifies the cached model artifact supplied by the model manager.
from pathlib import Path

# BaseEngine defines the common interface used by the engine registry.
from marp_inference_worker.engines.base_engine import BaseEngine

# ModelSpec defines the validated model metadata passed into the engine.
from marp_inference_worker.models.model_spec import ModelSpec

# Frame source utilities prepare local paths and URL images for prediction.
from marp_inference_worker.inputs.frame_source import prepare_frame_source

# Detection renderer draws normalized boxes and labels onto frame images.
from marp_inference_worker.rendering.detection_renderer import render_detections_to_image_bytes


# UltralyticsEngine
# Loads and stores Ultralytics-compatible YOLO model handles.
# This engine should handle Ultralytics .pt models and later frame/video calls.
# Keep Ultralytics-specific API usage isolated here.
class UltralyticsEngine(BaseEngine):

    # __init__()
    # Initializes the engine-local loaded model handle registry.
    # Inputs: none.
    # Output: initialized UltralyticsEngine instance.
    # Use this when the engine registry creates the shared engine instance.
    def __init__(self) -> None:

        # Keep real YOLO handles inside the engine so model_manager can stay generic.
        self._loaded_models_by_id: dict[str, object] = {}

    # engine_name
    # Returns the public engine name handled by this engine implementation.
    # Inputs: none.
    # Output: engine name string used in ModelSpec.engine.
    # Use this so the registry can match model specs to this engine.
    @property
    def engine_name(self) -> str:

        # Match the public engine name accepted by ModelSpec.engine.
        return "ultralytics"

    # load_model()
    # Loads a cached Ultralytics model artifact into memory.
    # Inputs: validated ModelSpec and local cached artifact path.
    # Output: API-safe metadata describing the loaded model.
    # Use this when /models/load receives an Ultralytics model spec.
    def load_model(self, model_spec: ModelSpec, artifact_path: Path) -> dict[str, object]:

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

    # get_loaded_model()
    # Returns a loaded Ultralytics model handle by MARP model ID.
    # Inputs: model ID string.
    # Output: loaded YOLO model object.
    # Use this later when inference endpoints need the loaded model.
    def get_loaded_model(self, model_id: str) -> object:

        # Fail clearly if inference is requested before a model is loaded.
        if model_id not in self._loaded_models_by_id:
            raise ValueError(f"Ultralytics model is not loaded: {model_id}")

        # Return the raw model handle for engine-internal inference use.
        return self._loaded_models_by_id[model_id]


    
        # _run_frame_prediction()
    # Runs YOLO prediction and normalizes detections for one visual frame.
    # Inputs: model ID, image source, confidence threshold, and class-name map.
    # Output: dictionary containing detections and source image array.
    # Use this so JSON and image routes share one prediction path.
    def _run_frame_prediction(
        self,
        model_id: str,
        image_source: str,
        confidence: float,
        class_names_by_id: dict[int, str],
    ) -> dict[str, object]:

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

            # Store normalized detections returned to callers.
            detections: list[dict[str, object]] = []

            # Ultralytics returns one result object per input image.
            for result in results:

                # Skip result objects with no detection boxes.
                if result.boxes is None:
                    continue

                # Normalize every detected box into the worker response contract.
                for box in result.boxes:

                    # Convert class ID and confidence into plain Python values.
                    class_id = int(box.cls[0].item())
                    detection_confidence = float(box.conf[0].item())

                    # Convert pixel xyxy tensor coordinates into plain Python floats.
                    bbox_xyxy = [
                        float(value)
                        for value in box.xyxy[0].tolist()
                    ]

                    # Convert normalized xyxy tensor coordinates into plain Python floats.
                    bbox_xyxyn = [
                        float(value)
                        for value in box.xyxyn[0].tolist()
                    ]

                    # Prefer labels from the model spec, then fall back to Ultralytics names.
                    class_name = class_names_by_id.get(
                        class_id,
                        yolo_model.names.get(class_id, str(class_id)),
                    )

                    # Add one normalized detection to the result list.
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": detection_confidence,
                            "bbox_xyxy": bbox_xyxy,
                            "bbox_xyxyn": bbox_xyxyn,
                        }
                    )

            # Return normalized detections and the original image for optional rendering.
            return {
                "detections": detections,
                "source_image": results[0].orig_img,
            }

        finally:
            # Clean up temporary URL downloads after prediction finishes.
            prepared_source.cleanup()


        # infer_frame()
    # Runs object detection on one image source using a loaded YOLO model.
    # Inputs: model ID, image source, confidence threshold, label map, and image flag.
    # Output: normalized detections and optional base64 annotated image.
    # Use this for first-pass frame inference before video or tracking jobs.
    def infer_frame(
        self,
        model_id: str,
        image_source: str,
        confidence: float,
        class_names_by_id: dict[int, str],
        return_annotated_image: bool = False,
    ) -> dict[str, object]:

        # Run prediction through the shared frame helper.
        prediction_result = self._run_frame_prediction(
            model_id=model_id,
            image_source=image_source,
            confidence=confidence,
            class_names_by_id=class_names_by_id,
        )

        # Build the base inference result shared by all frame requests.
        inference_result: dict[str, object] = {
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
    # Use this when an API route should return image bytes directly.
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


     