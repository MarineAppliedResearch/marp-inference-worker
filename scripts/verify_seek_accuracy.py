# verify_seek_accuracy.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Manual verification step: does seeking to a frame land on that frame?
# This is the entry point for the check `.marp/verification.md` calls manual
# step 3. It resolves a Jellyfin item to a stream, decodes it sequentially as
# ground truth, then seeks to each target on a freshly opened capture and reports
# where each seek actually landed.
#
# Only argument handling and reporting belong here. The analysis is in
# media/seek_verification.py so it can be unit tested without a video server.
#
# Credentials come from JELLYFIN_BASE_URL, JELLYFIN_USERNAME and
# JELLYFIN_PASSWORD in the environment, the same three names MARP_API uses. They
# are never printed, and the stream url is not printed either because it carries
# an api_key.
#
#   python scripts/verify_seek_accuracy.py <jellyfin_item_id> [target ...]

# argparse handles the item id and the optional target list.
import argparse

# sys supplies argv and the exit status.
import sys

# The Jellyfin client resolves an item id to a stream url.
from marp_inference_worker.media.jellyfin_client import JellyfinClient

# The measurement itself, which is where the reasoning lives.
from marp_inference_worker.media.seek_verification import (
    measure_seek,
    sequential_identities,
    summarize,
)


# Targets used when none are given.
# Spread deliberately: a couple inside the first GOP, the [0,300)/[300,600) seam
# the range design rests on, and one far enough in to have crossed many GOPs.
_DEFAULT_TARGETS = (1, 100, 299, 300, 301, 600, 900)


# How many frames after each seek are decoded and located.
# Larger is more discriminating on video with repeated frames and costs one
# decode per frame per target.
_RUN_LENGTH = 40


# main()
# Runs the measurement and prints the table and the verdict.
# Inputs: argv.
# Output: process exit code -- 0 when every seek was accounted for, 1 otherwise.
# Use this from a machine that can reach the video server.
def main(argv: list[str]) -> int:

    # Imported here rather than at module scope so `--help` works without
    # OpenCV installed.
    import cv2

    parser = argparse.ArgumentParser(description="Measure frame-seek accuracy on a stream.")
    parser.add_argument("item_id", help="Jellyfin item id of the video to measure")
    parser.add_argument(
        "targets",
        nargs="*",
        type=int,
        default=list(_DEFAULT_TARGETS),
        help="frame indices to seek to (default: a spread across the video)",
    )
    parser.add_argument(
        "--run-length",
        type=int,
        default=_RUN_LENGTH,
        help="how many frames to decode after each seek",
    )
    args = parser.parse_args(argv[1:])
    targets = sorted(set(args.targets or _DEFAULT_TARGETS))

    # Resolve the item to something OpenCV can open.
    client = JellyfinClient()
    client.authenticate()
    stream_url = client.build_direct_stream_url(args.item_id)
    client.close()

    # A fresh capture per seek, which is what a job does.
    def open_capture():

        # Raise rather than return an unopened capture, so a network failure is
        # not read as a seek failure.
        capture = cv2.VideoCapture(stream_url)
        if not capture.isOpened():
            raise RuntimeError("could not open the stream")
        return capture

    # Ground truth: one sequential decode, never seeked, stopped just past the
    # furthest target plus its run.
    ceiling = max(targets) + args.run_length + 1
    truth_capture = open_capture()
    try:
        print(f"item {args.item_id}")
        print(f"  reported fps   : {truth_capture.get(cv2.CAP_PROP_FPS)}")
        print(f"  reported frames: {truth_capture.get(cv2.CAP_PROP_FRAME_COUNT)}")
        identities = sequential_identities(truth_capture, limit=ceiling)
    finally:
        truth_capture.release()

    print(f"  ground truth   : {len(identities)} frames, {len(set(identities))} distinct")
    if len(set(identities)) < len(identities):
        # Worth saying out loud: this is the condition that makes a single-frame
        # comparison lie, and it is common in ROV video.
        print("  NOTE repeated frames exist in this video; runs are located, not single frames")
    print()

    # One measurement per target.
    print(f"{'target':>8} {'reported':>10} {'verdict':>16}  candidates")
    measurements = []
    for target in targets:
        if target >= len(identities):
            print(f"{target:>8} {'-':>10} {'past end':>16}  ground truth is only {len(identities)}")
            continue
        measurement = measure_seek(open_capture, identities, target, args.run_length)
        measurements.append(measurement)
        print(
            f"{measurement.target:>8} {measurement.reported_position:>10.1f} "
            f"{measurement.verdict:>16}  {measurement.candidates}"
        )

    # The verdict, and the exit code that goes with it.
    summary = summarize(measurements)
    print()
    print("verdicts        :", summary["verdicts"])
    print("distinct offsets:", summary["distinct_offsets"])
    print("PASSED" if summary["passed"] else "FAILED")
    return 0 if summary["passed"] else 1


# Entry point guard, so the module can be imported without running.
if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
