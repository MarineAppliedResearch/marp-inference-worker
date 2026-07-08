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


    # _prepare_image_source_for_prediction()
    # Converts image source input into a YOLO-readable source path.
    # Inputs: local path or HTTP/HTTPS image URL.
    # Output: prediction source path and optional temp file path for cleanup.
    # Use this so URL images are downloaded once instead of treated as streams.
    def _prepare_image_source_for_prediction(self, image_source: str) -> tuple[str, Path | None]:

        # Local paths can be passed directly to Ultralytics.
        if not image_source.lower().startswith(("http://", "https://")):
            return image_source, None

        # Standard library modules parse URL paths and create temp files.
        import tempfile
        from urllib.parse import urlparse

        # HTTPX downloads URL image sources before inference.
        import httpx

        # Download the image URL once so Ultralytics does not treat it as a stream.
        response = httpx.get(
            image_source,
            follow_redirects=True,
            timeout=30.0,
        )

        # Raise a clear error if the URL could not be downloaded.
        response.raise_for_status()

        # Confirm the server returned image content.
        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            raise ValueError(f"URL did not return image content: {content_type}")

        # Pick a file suffix from the content type or URL path.
        suffix_by_content_type = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
        }
        suffix = suffix_by_content_type.get(content_type.split(";")[0], "")

        # Fall back to the URL path suffix if content type was less specific.
        if suffix == "":
            url_path = Path(urlparse(image_source).path)
            suffix = url_path.suffix if url_path.suffix else ".jpg"

        # Write the downloaded image to a temporary file for YOLO prediction.
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = Path(temp_file.name)

        try:
            temp_file.write(response.content)
            temp_file.close()
        except Exception:
            temp_file.close()
            temp_path.unlink(missing_ok=True)
            raise

        # Return the temp file path and remember it for cleanup after inference.
        return str(temp_path), temp_path
    


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

        # Get the already-loaded YOLO model handle for this model ID.
        yolo_model = self.get_loaded_model(model_id)

        # Convert URLs into temporary local files before prediction.
        prediction_source, temp_image_path = self._prepare_image_source_for_prediction(image_source)

        try:
            # Run Ultralytics prediction on the prepared image source.
            results = yolo_model.predict(
                source=prediction_source,
                conf=confidence,
                verbose=False,
            )

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

            # Build the base inference result shared by all frame requests.
            inference_result: dict[str, object] = {
                "detections": detections,
            }

            # Optionally add a base64-encoded annotated JPG to the JSON result.
            if return_annotated_image:

                # Base64 is used so the annotated image can travel inside JSON.
                import base64

                # Render the first image result with custom labels and boxes.
                annotated_image_bytes = self._render_annotated_image_jpg(
                    results=results,
                    detections=detections,
                )

                # Add image metadata and payload to the inference result.
                inference_result["annotated_image_format"] = "jpg"
                inference_result["annotated_image_base64"] = base64.b64encode(
                    annotated_image_bytes
                ).decode("utf-8")

            # Return detections and optional annotated image data.
            return inference_result

        finally:
            # Remove temporary URL downloads after prediction and rendering finish.
            if temp_image_path is not None:
                temp_image_path.unlink(missing_ok=True)
    
        # infer_frame_image()
    # Runs inference and returns an annotated JPG image.
    # Inputs: model ID, image source, confidence threshold, and class-name map.
    # Output: encoded JPG image bytes.
    # Use this when an API route should return image/jpeg directly.
    def infer_frame_image(
        self,
        model_id: str,
        image_source: str,
        confidence: float,
        class_names_by_id: dict[int, str],
    ) -> bytes:

        # Get the already-loaded YOLO model handle for this model ID.
        yolo_model = self.get_loaded_model(model_id)

        # Convert URLs into temporary local files before prediction.
        prediction_source, temp_image_path = self._prepare_image_source_for_prediction(image_source)

        try:
            # Run Ultralytics prediction on the prepared image source.
            results = yolo_model.predict(
                source=prediction_source,
                conf=confidence,
                verbose=False,
            )

            # Store normalized detections used by the shared rendering helper.
            detections: list[dict[str, object]] = []

            # Ultralytics returns one result object per input image.
            for result in results:

                # Skip result objects with no detection boxes.
                if result.boxes is None:
                    continue

                # Convert every detection into the same normalized shape used by JSON.
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

                    # Add one normalized detection to the render list.
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": detection_confidence,
                            "bbox_xyxy": bbox_xyxy,
                            "bbox_xyxyn": bbox_xyxyn,
                        }
                    )

            # Render the first image result with the shared annotation helper.
            return self._render_annotated_image_jpg(
                results=results,
                detections=detections,
            )

        finally:
            # Remove temporary URL downloads after prediction and rendering finish.
            if temp_image_path is not None:
                temp_image_path.unlink(missing_ok=True)


        # _render_annotated_image_jpg()
    # Draws detection boxes and attached translucent labels onto one image.
    # Inputs: Ultralytics results and normalized detection dictionaries.
    # Output: encoded JPG image bytes.
    # Use this so JSON and direct-image routes share identical rendering.
    def _render_annotated_image_jpg(
        self,
        results: object,
        detections: list[dict[str, object]],
    ) -> bytes:

        # OpenCV is used for drawing and JPG encoding.
        import cv2

        # Fail clearly if Ultralytics returned no result object to render.
        if not results:
            raise ValueError("Ultralytics returned no inference results to annotate.")

        # Copy the original image so drawing does not mutate the Ultralytics result.
        annotated_image = results[0].orig_img.copy()

        # Read image dimensions so labels can be clamped inside image bounds.
        image_height, image_width = annotated_image.shape[:2]

        # Draw each detection box and attached label.
        for detection in detections:

            # Extract and round pixel box coordinates.
            x1, y1, x2, y2 = [
                int(round(value))
                for value in detection["bbox_xyxy"]
            ]

            # Clamp the box coordinates to the image bounds.
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))

            # Calculate box size for label placement and fitting.
            box_width = max(1, x2 - x1)
            box_height = max(1, y2 - y1)

            # Compose a concise label with class name and confidence.
            label_text = (
                f"{detection['class_name']} "
                f"{float(detection['confidence']):.2f}"
            )

            # Set font bounds so small boxes remain readable and large boxes stay reasonable.
            max_font_scale = min(0.85, max(0.45, box_height / 120.0))
            min_font_scale = 0.35
            font_scale = max_font_scale
            text_thickness = 1

            # Reserve horizontal padding inside the label bar.
            horizontal_padding = max(4, min(10, box_width // 12))
            available_text_width = max(1, box_width - (horizontal_padding * 2))

            # Shrink the font until the label fits the box width or reaches the minimum.
            while font_scale > min_font_scale:

                # Measure the current label text at the current font scale.
                text_size, baseline = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_thickness,
                )
                text_width, text_height = text_size

                # Stop shrinking once the label fits inside the box width.
                if text_width <= available_text_width:
                    break

                # Reduce font size gradually to fit longer labels.
                font_scale -= 0.05

            # Clamp to the minimum readable font size.
            font_scale = max(min_font_scale, font_scale)

            # Re-measure after final font scale is selected.
            text_size, baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_thickness,
            )
            text_width, text_height = text_size

            # Truncate the label if it still does not fit at the minimum font size.
            if text_width > available_text_width:

                # Keep shortening until the label plus ellipsis fits.
                truncated_label = label_text
                while len(truncated_label) > 3:

                    # Test the shortened label with an ellipsis.
                    candidate_label = truncated_label[:-1].rstrip() + "..."
                    text_size, baseline = cv2.getTextSize(
                        candidate_label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_thickness,
                    )
                    text_width, text_height = text_size

                    # Use the first shortened label that fits the available width.
                    if text_width <= available_text_width:
                        label_text = candidate_label
                        break

                    # Remove one more character and test again.
                    truncated_label = truncated_label[:-1]

            # Re-measure the final label after any truncation.
            text_size, baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_thickness,
            )
            text_width, text_height = text_size

            # Vertical padding keeps text from touching the label bar edges.
            vertical_padding = max(3, int(round(font_scale * 6)))

            # Make the label bar exactly align with the bounding box x coordinates.
            label_x1 = x1
            label_x2 = x2

            # Prefer placing the label directly below the bounding box.
            label_y1 = y2
            label_y2 = y2 + text_height + baseline + (vertical_padding * 2)

            # If below the box would leave the image, place the label inside the box bottom.
            if label_y2 >= image_height:
                label_y2 = y2
                label_y1 = y2 - text_height - baseline - (vertical_padding * 2)

            # Clamp the label bar inside the image.
            label_y1 = max(0, min(label_y1, image_height - 1))
            label_y2 = max(0, min(label_y2, image_height - 1))

            # Draw a visible bounding box around the detection.
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2,
            )

            # Draw a translucent label background aligned to the box width.
            overlay = annotated_image.copy()
            cv2.rectangle(
                overlay,
                (label_x1, label_y1),
                (label_x2, label_y2),
                (0, 0, 0),
                -1,
            )
            cv2.addWeighted(
                overlay,
                0.55,
                annotated_image,
                0.45,
                0,
                annotated_image,
            )

            # Center the label text horizontally inside the label bar.
            text_x = label_x1 + max(0, ((label_x2 - label_x1) - text_width) // 2)

            # Center the label text vertically inside the label bar.
            label_height = max(1, label_y2 - label_y1)
            text_y = label_y1 + ((label_height + text_height) // 2) - baseline

            # Clamp the text baseline so it stays visible.
            text_y = max(text_height, min(text_y, image_height - baseline - 1))

            # Draw the label text in white over the translucent background.
            cv2.putText(
                annotated_image,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        # Encode the final rendered image as JPG bytes.
        encoded_success, encoded_image = cv2.imencode(".jpg", annotated_image)

        # Fail clearly if OpenCV cannot encode the rendered image.
        if not encoded_success:
            raise ValueError("Could not encode annotated inference image.")

        # Return raw JPG bytes for direct API image responses.
        return encoded_image.tobytes()