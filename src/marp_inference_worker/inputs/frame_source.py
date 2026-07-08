# frame_source.py
# Created: 2026-07-08
# Author: Isaac Travers
#
# Frame source preparation utilities for the MARP Inference Worker.
# This file prepares visual frame inputs for image-capable inference engines.
# Local image paths and HTTP/HTTPS image URLs should be normalized here so
# engine implementations do not duplicate source handling logic.

# Temporary files are used when URL images must be downloaded before inference.
import tempfile

# Path is used for local file paths and temporary downloaded images.
from pathlib import Path

# URL parsing is used to inspect image URL path suffixes.
from urllib.parse import urlparse

# HTTPX downloads image URLs before they are passed into model engines.
import httpx


# PreparedFrameSource
# Describes a frame source that has been prepared for engine prediction.
# The source path is always readable by local engine code.
# Temporary paths are tracked so callers can clean them up after inference.
class PreparedFrameSource:

    # __init__()
    # Initializes a prepared frame source record.
    # Inputs: prediction source path and optional temporary file path.
    # Output: initialized PreparedFrameSource instance.
    # Use this to pass source path plus cleanup metadata together.
    def __init__(self, prediction_source: str, temp_path: Path | None = None) -> None:

        # Prediction source is the local path passed into model engine inference.
        self.prediction_source = prediction_source

        # Temporary path is set only when a URL was downloaded to disk.
        self.temp_path = temp_path

    # cleanup()
    # Removes any temporary file created while preparing the frame source.
    # Inputs: none.
    # Output: none.
    # Use this in finally blocks after inference has finished.
    def cleanup(self) -> None:

        # Remove temporary URL downloads and ignore already-removed files.
        if self.temp_path is not None:
            self.temp_path.unlink(missing_ok=True)


# prepare_frame_source()
# Converts a visual frame source into a local engine-readable path.
# Inputs: local image path or HTTP/HTTPS image URL.
# Output: PreparedFrameSource with prediction path and cleanup metadata.
# Use this before running visual-frame inference in any image-capable engine.
def prepare_frame_source(image_source: str) -> PreparedFrameSource:

    # Local paths can be passed directly to image-capable engines.
    if not image_source.lower().startswith(("http://", "https://")):
        return PreparedFrameSource(prediction_source=image_source)

    # Download URL sources once so engines do not treat image URLs as streams.
    response = httpx.get(
        image_source,
        follow_redirects=True,
        timeout=30.0,
    )

    # Raise a clear HTTPX error if the URL could not be downloaded.
    response.raise_for_status()

    # Confirm the server returned image content instead of HTML or video.
    content_type = response.headers.get("content-type", "").lower()
    if not content_type.startswith("image/"):
        raise ValueError(f"URL did not return image content: {content_type}")

    # Choose a temporary file suffix from the response content type.
    suffix = _get_image_suffix(
        content_type=content_type,
        image_source=image_source,
    )

    # Write the downloaded image to a named temporary file for engine prediction.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)

    try:
        # Save the downloaded image bytes for the model engine to read.
        temp_file.write(response.content)
        temp_file.close()
    except Exception:
        # Close and remove the temp file if writing fails.
        temp_file.close()
        temp_path.unlink(missing_ok=True)
        raise

    # Return the local temp path and cleanup metadata.
    return PreparedFrameSource(
        prediction_source=str(temp_path),
        temp_path=temp_path,
    )


# _get_image_suffix()
# Selects a local file suffix for a downloaded image source.
# Inputs: HTTP content type and original image source string.
# Output: file suffix string such as .jpg or .png.
# Use this so temp files keep a useful image extension.
def _get_image_suffix(content_type: str, image_source: str) -> str:

    # Map common image content types to file suffixes.
    suffix_by_content_type = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }

    # Remove optional content-type parameters such as charset.
    normalized_content_type = content_type.split(";")[0].strip()

    # Prefer the content type when it maps cleanly to an image suffix.
    if normalized_content_type in suffix_by_content_type:
        return suffix_by_content_type[normalized_content_type]

    # Fall back to the URL path suffix when the content type is generic.
    url_path = Path(urlparse(image_source).path)
    if url_path.suffix:
        return url_path.suffix

    # Default to JPG when no better suffix is available.
    return ".jpg"