# -----------------------------------------------------------------------------
# File name: test_jellyfin_resolver.py
# Date created: 2026-07-09
# Author: Isaac Travers
#
# Manual Jellyfin resolver test for the MARP inference worker.
# This script authenticates to Jellyfin, resolves a database-style video_source
# value, and verifies that OpenCV can read the resolved stream.
# -----------------------------------------------------------------------------

# os reads Jellyfin credentials and test video source from environment variables.
import os

# sys exits with explicit status codes for manual testing.
import sys

# cv2 verifies whether OpenCV can read the resolved video source.
import cv2

# JellyfinClient provides authenticated Jellyfin API access.
from marp_inference_worker.media.jellyfin_client import JellyfinClient

# VideoSourceResolver converts database video_source values into readable sources.
from marp_inference_worker.media.video_source_resolver import VideoSourceResolver


# Runs the manual Jellyfin resolver smoke test.
# Inputs come from environment variables.
# Outputs are console logs and one saved JPG frame.
# Use this before wiring the resolver into model_training_live.py.
def main():
    
    # The test value should match a database video_source value.
    video_source = os.getenv("JELLYFIN_TEST_VIDEO", "").strip()

    # Fail early when the test value is missing.
    if not video_source:
        print("[ERROR] Set JELLYFIN_TEST_VIDEO to a database video_source value.")
        return 1

    # Create the Jellyfin client from environment variables.
    client = JellyfinClient()

    try:
        # Authenticate before resolving Jellyfin streams.
        client.authenticate()

        print("[INFO] Authenticated Jellyfin client.")
        print(f"[INFO] Resolving video source: {video_source}")

        # Create a resolver that falls back to Jellyfin.
        resolver = VideoSourceResolver(jellyfin_client=client, prefer_local=True)

        # Resolve the database video_source value.
        resolution = resolver.resolve(video_source)

        # Fail clearly if no source was found.
        if resolution is None:
            print("[ERROR] Could not resolve video source.")
            return 1

        print("[INFO] Resolution:")
        print(f"  Source type: {resolution.source_type}")
        print(f"  Resolved:    {resolution.resolved_source}")
        print(f"  Match score: {resolution.match_score}")
        print(f"  Search term: {resolution.search_term}")

        # Print Jellyfin metadata when the resolver used Jellyfin.
        if resolution.jellyfin_item is not None:
            print("[INFO] Jellyfin item:")
            print(f"  Name: {resolution.jellyfin_item.name}")
            print(f"  Id:   {resolution.jellyfin_item.item_id}")
            print(f"  Path: {resolution.jellyfin_item.path}")

        # Open the resolved source with OpenCV.
        cap = cv2.VideoCapture(resolution.resolved_source)

        # Fail clearly if OpenCV cannot open the source.
        if not cap.isOpened():
            print("[ERROR] OpenCV could not open the resolved source.")
            return 1

        # Print basic stream properties.
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print("[INFO] OpenCV opened the resolved source.")
        print(f"  Width:       {width}")
        print(f"  Height:      {height}")
        print(f"  FPS:         {fps}")
        print(f"  Frame count: {frame_count}")

        # Read the first frame.
        ret, frame = cap.read()

        # Release the capture handle immediately after the test frame.
        cap.release()

        # Fail clearly if decoding did not produce a frame.
        if not ret or frame is None:
            print("[ERROR] OpenCV opened the source but did not decode a frame.")
            return 1

        # Save one frame for visual confirmation.
        output_path = os.path.join("datasets", "jellyfin_resolver_test_frame.jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the decoded frame.
        cv2.imwrite(output_path, frame)

        print(f"[INFO] Saved resolver test frame: {output_path}")
        return 0

    finally:
        # Close the HTTP client before exiting.
        client.close()


# Standard script entry point.
if __name__ == "__main__":
    sys.exit(main())