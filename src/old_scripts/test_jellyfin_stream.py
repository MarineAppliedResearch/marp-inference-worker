# -----------------------------------------------------------------------------
# File name: test_jellyfin_stream.py
# Date created: 2026-07-09
# Author: Isaac Travers
#
# Manual Jellyfin stream test for the MARP inference worker.
# This script authenticates to Jellyfin, searches for a known video, builds a
# stream URL, and verifies that OpenCV can read at least one frame.
# -----------------------------------------------------------------------------

# os reads Jellyfin credentials and the test video query from environment variables.
import os

# sys exits with explicit status codes for manual testing.
import sys

# cv2 tests whether OpenCV can open a Jellyfin stream URL.
import cv2

# JellyfinClient provides the reusable worker-side Jellyfin API wrapper.
from marp_inference_worker.media.jellyfin_client import JellyfinClient


# Runs the manual Jellyfin stream smoke test.
# Inputs come from environment variables.
# Outputs are console logs and one saved JPG frame.
# Use this before wiring Jellyfin into dataset building.
def main():

    # The search query should be a filename or distinct part of a known Jellyfin video.
    query = os.getenv("JELLYFIN_TEST_VIDEO", "").strip()

    # Fail early if no test video was supplied.
    if not query:
        print("[ERROR] Set JELLYFIN_TEST_VIDEO to a known video filename or search term.")
        return 1

    # Create the client from JELLYFIN_BASE_URL, JELLYFIN_USERNAME, and JELLYFIN_PASSWORD.
    client = JellyfinClient()

    try:
        # Authenticate before searching or building stream URLs.
        client.authenticate()

        print("[INFO] Authenticated Jellyfin client.")
        print(f"[INFO] Searching Jellyfin for: {query}")

        # Search Jellyfin for matching video items.
        matches = client.search_video_items(query, limit=10)

        # Fail clearly if Jellyfin returned no candidates.
        if not matches:
            print("[ERROR] No Jellyfin video matches found.")
            return 1

        # Use the first match for this initial smoke test.
        item = matches[0]

        print("[INFO] First match:")
        print(f"  Name: {item.name}")
        print(f"  Id:   {item.item_id}")
        print(f"  Path: {item.path}")
        print(f"  Type: {item.item_type}")
        print(f"  MediaType: {item.media_type}")

        # Build the same original/direct stream URL pattern used by the C# client.
        stream_url = client.build_direct_stream_url(item.item_id)

        print("[INFO] Direct stream URL:")
        print(stream_url)

        # Open the Jellyfin stream with OpenCV.
        cap = cv2.VideoCapture(stream_url)

        # If OpenCV cannot open the stream, report that before trying to read frames.
        if not cap.isOpened():
            print("[ERROR] OpenCV could not open the Jellyfin stream URL.")
            return 1

        # Print basic stream properties if OpenCV can read them.
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print("[INFO] OpenCV opened the stream.")
        print(f"  Width:       {width}")
        print(f"  Height:      {height}")
        print(f"  FPS:         {fps}")
        print(f"  Frame count: {frame_count}")

        # Read the first frame.
        ret, frame = cap.read()

        # Release the stream as soon as the smoke test is done.
        cap.release()

        # Fail clearly if decoding did not produce a frame.
        if not ret or frame is None:
            print("[ERROR] OpenCV opened the stream but did not decode a frame.")
            return 1

        # Save one frame so the result can be visually checked.
        output_path = os.path.join("datasets", "jellyfin_stream_test_frame.jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the decoded frame to disk.
        cv2.imwrite(output_path, frame)

        print(f"[INFO] Saved test frame: {output_path}")
        return 0

    finally:
        # Close the HTTP client even if the test fails.
        client.close()


# Standard script entry point.
if __name__ == "__main__":
    sys.exit(main())