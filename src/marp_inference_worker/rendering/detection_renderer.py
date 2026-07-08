# detection_renderer.py
# Created: 2026-07-08
# Author: Isaac Travers
#
# Detection rendering utilities for the MARP Inference Worker.
# This file draws normalized detection results onto visual frame images.
# Shared annotation and image-encoding logic belongs here so multiple engines
# can reuse the same rendering behavior.

# OpenCV is used for drawing boxes, text, translucent overlays, and image encoding.
import cv2


# Minimum readable font scale for detection labels.
LABEL_MIN_FONT_SCALE = 0.55

# Maximum font scale so large boxes do not create oversized labels.
LABEL_MAX_FONT_SCALE = 0.85

# Box-height divisor used to estimate initial label font size.
LABEL_FONT_SCALE_BOX_HEIGHT_DIVISOR = 120.0

# Minimum label bar width used when detection boxes are too narrow for readable text.
LABEL_MIN_WIDTH_PIXELS = 140

# Horizontal padding inside the label bar.
LABEL_HORIZONTAL_PADDING_PIXELS = 8

# Minimum vertical padding inside the label bar.
LABEL_MIN_VERTICAL_PADDING_PIXELS = 4

# Multiplier used to scale vertical label padding with font size.
LABEL_VERTICAL_PADDING_SCALE = 6

# Detection box line thickness in pixels.
DETECTION_BOX_THICKNESS_PIXELS = 2

# Label background opacity. Higher values make the black label background stronger.
LABEL_BACKGROUND_ALPHA = 0.55

# Original image opacity used when blending the translucent label background.
LABEL_IMAGE_ALPHA = 1.0 - LABEL_BACKGROUND_ALPHA

# Font size reduction step used while fitting labels.
LABEL_FONT_SCALE_REDUCTION_STEP = 0.05


# render_detections_to_image_bytes()
# Draws detection boxes and labels onto an image and encodes the result.
# Inputs: source image array, normalized detections, and output format.
# Output: encoded image bytes.
# Use this when an API route needs an annotated image response.
def render_detections_to_image_bytes(
    source_image: object,
    detections: list[dict[str, object]],
    output_format: str = "jpg",
) -> bytes:

    # Copy the source image so rendering does not mutate caller-owned image data.
    annotated_image = source_image.copy()

    # Draw the normalized detections onto the copied image.
    _draw_detections_on_image(
        image=annotated_image,
        detections=detections,
    )

    # Encode the rendered image into the requested output format.
    return encode_image_bytes(
        image=annotated_image,
        output_format=output_format,
    )


# encode_image_bytes()
# Encodes an image array into bytes using OpenCV.
# Inputs: image array and output format string.
# Output: encoded image bytes.
# Use this for browser image responses and base64 JSON image payloads.
def encode_image_bytes(image: object, output_format: str = "jpg") -> bytes:

    # Normalize common format spellings into OpenCV file extensions.
    normalized_format = output_format.lower().strip().lstrip(".")
    extension_by_format = {
        "jpg": ".jpg",
        "jpeg": ".jpg",
        "png": ".png",
        "webp": ".webp",
    }

    # Fall back to JPG when the requested format is not supported yet.
    output_extension = extension_by_format.get(normalized_format, ".jpg")

    # Encode the final rendered image as bytes.
    encoded_success, encoded_image = cv2.imencode(output_extension, image)

    # Fail clearly if OpenCV cannot encode the rendered image.
    if not encoded_success:
        raise ValueError(f"Could not encode annotated inference image as {output_extension}.")

    # Return raw encoded image bytes for API responses.
    return encoded_image.tobytes()


# _draw_detections_on_image()
# Draws detection boxes and attached translucent labels onto an image.
# Inputs: mutable image array and normalized detection dictionaries.
# Output: none.
# Use this internally so image encoding and drawing stay separate.
def _draw_detections_on_image(image: object, detections: list[dict[str, object]]) -> None:

    # Read image dimensions so labels can be clamped inside image bounds.
    image_height, image_width = image.shape[:2]

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

        # Start from a box-scaled font size but clamp it to readable limits.
        font_scale = min(
            LABEL_MAX_FONT_SCALE,
            max(
                LABEL_MIN_FONT_SCALE,
                box_height / LABEL_FONT_SCALE_BOX_HEIGHT_DIVISOR,
            ),
        )
        text_thickness = 1

        # Use a minimum readable label width so small boxes do not create tiny labels.
        label_target_width = min(
            image_width,
            max(LABEL_MIN_WIDTH_PIXELS, box_width),
        )
        available_text_width = max(
            1,
            label_target_width - (LABEL_HORIZONTAL_PADDING_PIXELS * 2),
        )

        # Shrink the font only until the readable minimum is reached.
        while font_scale > LABEL_MIN_FONT_SCALE:

            # Measure the current label text at the current font scale.
            text_size, baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_thickness,
            )
            text_width, text_height = text_size

            # Stop shrinking once the label fits inside the target width.
            if text_width <= available_text_width:
                break

            # Reduce font size gradually, but not below the readable minimum.
            font_scale -= LABEL_FONT_SCALE_REDUCTION_STEP

        # Clamp to the minimum readable font size.
        font_scale = max(LABEL_MIN_FONT_SCALE, font_scale)

        # Re-measure after final font scale is selected.
        text_size, baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        text_width, text_height = text_size

        # If the text still does not fit, widen the label bar beyond the box.
        label_width = max(
            box_width,
            LABEL_MIN_WIDTH_PIXELS,
            text_width + (LABEL_HORIZONTAL_PADDING_PIXELS * 2),
        )

        # Do not allow the label bar to exceed the image width.
        label_width = min(label_width, image_width)

        # If the label still cannot fit, truncate at the readable minimum size.
        available_text_width = max(
            1,
            label_width - (LABEL_HORIZONTAL_PADDING_PIXELS * 2),
        )
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

        # Re-measure the final label after any widening or truncation.
        text_size, baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        text_width, text_height = text_size

        # Vertical padding keeps text from touching the label bar edges.
        vertical_padding = max(
            LABEL_MIN_VERTICAL_PADDING_PIXELS,
            int(round(font_scale * LABEL_VERTICAL_PADDING_SCALE)),
        )

        # Center the label bar horizontally on the detection box.
        box_center_x = x1 + (box_width // 2)
        label_x1 = box_center_x - (label_width // 2)
        label_x2 = label_x1 + label_width

        # Clamp the label bar horizontally inside the image.
        if label_x1 < 0:
            label_x1 = 0
            label_x2 = label_width
        if label_x2 >= image_width:
            label_x2 = image_width - 1
            label_x1 = max(0, label_x2 - label_width)

        # Prefer placing the label directly below the bounding box.
        label_y1 = y2
        label_y2 = y2 + text_height + baseline + (vertical_padding * 2)

        # If below the box would leave the image, place the label inside the box bottom.
        if label_y2 >= image_height:
            label_y2 = y2
            label_y1 = y2 - text_height - baseline - (vertical_padding * 2)

        # Clamp the label bar vertically inside the image.
        label_y1 = max(0, min(label_y1, image_height - 1))
        label_y2 = max(0, min(label_y2, image_height - 1))

        # Draw a visible bounding box around the detection.
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            DETECTION_BOX_THICKNESS_PIXELS,
        )

        # Draw a translucent label background centered on the detection box.
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (label_x1, label_y1),
            (label_x2, label_y2),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(
            overlay,
            LABEL_BACKGROUND_ALPHA,
            image,
            LABEL_IMAGE_ALPHA,
            0,
            image,
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
            image,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )