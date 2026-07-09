
# pathlib provides safe project-relative path handling.
from pathlib import Path

# sys allows this old script to temporarily import the local ByteTrack package.
import sys


# PROJECT_ROOT points to the marp-inference-project folder from this script location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# BYTETRACK_ROOT points to the copied ByteTrack folder beside src/.
BYTETRACK_ROOT = PROJECT_ROOT / "ByteTrack"

# Add ByteTrack to Python's import path so yolox.tracker.byte_tracker can be imported.
sys.path.insert(0, str(BYTETRACK_ROOT))

import os
import cv2
import csv
import math
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from tkinter import Tk, filedialog, messagebox
from yolox.tracker.byte_tracker import BYTETracker
from itertools import islice
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import api
import matplotlib.pyplot as plt
from bisect import bisect_left


COMNAME_TO_TAXSERIAL = {
    "Bat star": 157133,
    "California sea cucumber": 158344,
    "Cookie star": 156992,

    "Fish eating star": 157272,
    "Fish-eating anemone": 611869,
    "Leather star": 157139,
    "Red sea star": 157011,
    "Red sea urchin": 157971,
    "Short red gorgonian": 719219,
    "White-plumed anemone" : 611773,

    "Thorny sea star": 157176,
    "UI sea star": 97,
    "White spine sea cucumber": 656044,
    "Vermilion Rockfish": 166729,
    "Olive/Yellowtail Rockfish complex": 30,
    "Canary Rockfish": 166734,
    "Blue/Deacon Rockfish": 166730,
    "Gopher Rockfish": 166767,
    "Copper Rockfish": 166713,
    "Lingcod": 167116,
    "Quillback Rockfish": 166726

    # add more...
}

project_id = None  # start as None until user makes a selection
user_id = None
dive = None
line = None
dataType = None
session = None  # Session info from db
model_file = None
video_path = None
session_ID = None


active_tracks = {}  # track_id -> {"class": str, "frames": []}


# Alias np.float to handle deprecated usage
if not hasattr(np, 'float'):
    np.float = float

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

class Args:
    def __init__(self):
        self.confidence_threshold = 0.15
        self.track_thresh = 0.30
        self.match_thresh = 0.70
        self.track_buffer = 240
        self.mot20 = False

def select_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with Videos")
    root.destroy()
    return folder_path

def batch_videos(video_files, batch_size):
    """Yield successive batches of videos from the list."""
    it = iter(video_files)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch

def parse_timecode(tc: str) -> float:
    """Convert HH:MM:SS.sss string into seconds as float."""
    import datetime
    h, m, s = tc.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


""" def process_video(total_video_path, yolo_model_path, view_videos, start_time=None):
    try:
        print(f"Processing video: {os.path.basename(total_video_path)}")

        # Load a separate YOLO model instance for each thread
        model = YOLO(yolo_model_path).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Create a separate tracker instance for each thread
        tracker = BYTETracker(Args())

        cap = cv2.VideoCapture(total_video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file: {total_video_path}")
            return

        # if a starting time is given, jump to it
        if start_time:
            # OpenCV takes milliseconds
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            print(f"Starting video at {start_time:.2f} seconds")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        model_name = os.path.splitext(os.path.basename(yolo_model_path))[0]  # Strip path and extension

        # Create the output directory based on the model name
        output_folder = os.path.join(os.path.dirname(total_video_path), model_name)
        os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

        # Create output filenames inside the output folder
        base_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract the video filename without extension
        output_video_path = os.path.join(output_folder, f"{base_name}_output.mp4")
        csv_file_path = os.path.join(output_folder, f"{base_name}_output.csv")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

        # Resizing dimensions for live preview
        display_width = frame_width // 2
        display_height = frame_height // 2

        if view_videos:
            # Set the window title with video name and model name
            window_title = f"{os.path.basename(total_video_path)} - YOLOv8 Model"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_title, display_width, display_height)

        with open(csv_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Frame", "Object Name", "Track_ID", "X", "Y", "Width", "Height", "Confidence"])

            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                #print(f"Processing Frame {frame_idx + 1} of {total_frames} for {os.path.basename(video_path)}...")

                # YOLOv8 detection
                results = model(frame)
                detections = results[0]

                detection_boxes = []
                for i, box in enumerate(detections.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = box[:4]
                    confidence = detections.boxes.conf.cpu().numpy()[i]
                    detection_boxes.append([x1, y1, x2, y2, confidence])

                detection_boxes = np.array(detection_boxes) if detection_boxes else np.empty((0, 5))
                img_info = [frame_height, frame_width]
                tracked_objects = tracker.update(detection_boxes, img_info, (frame_height, frame_width))

                track_id_to_class_id = {}

                for track in tracked_objects:
                    x1, y1, x2, y2 = map(int, track.tlbr)
                    track_id = track.track_id

                    if track_id not in track_id_to_class_id:
                        for i, det_box in enumerate(detections.boxes.xyxy.cpu().numpy()):
                            det_x1, det_y1, det_x2, det_y2 = det_box[:4]
                            iou = calculate_iou((x1, y1, x2, y2), (det_x1, det_y1, det_x2, det_y2))
                            if iou > 0.4:
                                track_id_to_class_id[track_id] = (int(detections.boxes.cls.cpu().numpy()[i]), detections.boxes.conf.cpu().numpy()[i])
                                break

                    class_id, confidence = track_id_to_class_id.get(track_id, (-1, 0.0))
                    object_name = model.names[class_id] if class_id != -1 else "Unknown"

                    # Normalize bounding box
                    xCenter = (x1 + x2) / 2 / frame_width
                    yCenter = (y1 + y2) / 2 / frame_height
                    width = (x2 - x1) / frame_width
                    height = (y2 - y1) / frame_height

                    # Store in active_tracks
                    if track_id not in active_tracks:
                        active_tracks[track_id] = {
                            "class_name": object_name,
                            "frames": [],
                            "last_seen": frame_idx
                        }

                    active_tracks[track_id]["frames"].append({
                        "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)), 
                        "time": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                        "bbox": (xCenter, yCenter, width, height),
                        "confidence": confidence
                    })

                    active_tracks[track_id]["last_seen"] = frame_idx  # update every time we see it

                    csv_writer.writerow([frame_idx, object_name, track_id, xCenter, yCenter, width, height, confidence])

                    # Draw bounding box and labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, object_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # Draw the ID and confidence at the bottom of the bounding box
                    label = f"ID: {track_id}, Conf: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # --- after processing all tracks in this frame ---
                # --- after processing all tracks in this frame ---
                # Finalize tracks missing for longer than buffer
                for tid, obs in list(active_tracks.items()):
                    if frame_idx - obs["last_seen"] > tracker.args.track_buffer:
                        ended_obs = active_tracks.pop(tid)

                        num_frames = ended_obs["frames"][-1]["frame"] - ended_obs["frames"][0]["frame"]
                        if num_frames < 30:
                            print(f"[OBS] Track {tid} discarded (only {num_frames} frames)")
                            continue  # skip this one

                        print(f"[OBS] Track {tid} finalized with {len(ended_obs['frames'])} frames")

                         # --- reduce frames -> keyframes ---
                        keyframes = reduce_to_keyframes(ended_obs)
                        print(f"[OBS] Reduced to {len(keyframes)} keyframes")

                        # --- debug plot (optional) ---
                        plot_keyframe_reduction(ended_obs["frames"], keyframes, title=f"Track {tid}")

                         # --- pick observation time ---
                        chosen_kf = pick_observation_time(ended_obs["frames"], dataType)
                        if chosen_kf:
                            print(f"[OBS] Time chosen at frame {chosen_kf['frame']} (dataset={dataType})")
                        else:
                            print(f"[OBS] No valid collision found, using fallback")

                        # TODO: send obs + keyframes to DB
                        chosen_frame = chosen_kf["frame"]

                        observation_payload = {
                            "session_id": session_ID,
                            "comname": keyframes[0]["comname"],       # from YOLO mapping
                            "taxserial": COMNAME_TO_TAXSERIAL.get(keyframes[0]["comname"]),  # if available
                            "count": 1,
                            "tc": frame_to_timecode(chosen_frame, frame_rate),
                            "frame": str(chosen_frame % int(frame_rate)),
                            "video_source": os.path.basename(video_path),
                            "videoLocation": video_path,
                            "mediaPosition": frame_to_media_position(chosen_frame, frame_rate),
                            "actualPosition": frame_to_media_position(chosen_frame, frame_rate),
                            "keyframes": keyframes  # already reduced & labeled with start/middle/end
                        }

                        api.create_observation(observation_payload)

                out.write(frame)

                if view_videos:
                    cv2.imshow(window_title, frame)
                    # Use waitKey(1) ONLY to catch key presses, without adding delay
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        print(f"User terminated video preview for {os.path.basename(total_video_path)}.")
                        break

                frame_idx += 1

        cap.release()
        out.release()
        if view_videos:
            cv2.destroyWindow(window_title)
        print(f"Processing complete for {os.path.basename(total_video_path)}")
        print(f"Results saved to:\nVideo: {output_video_path}\nCSV: {csv_file_path}")

    except Exception as e:
        print(f"Error processing video {total_video_path}: {e}") """

def process_video(total_video_path, yolo_model_path, view_videos, start_time=None):
    try:
        print(f"Processing video: {os.path.basename(total_video_path)}")

        model = YOLO(yolo_model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
        tracker = BYTETracker(Args())

        cap = cv2.VideoCapture(total_video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file: {total_video_path}")
            return

        if start_time:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            print(f"Starting video at {start_time:.2f} seconds")

        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate   = max(1, int(cap.get(cv2.CAP_PROP_FPS)))  # guard against 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        model_name = os.path.splitext(os.path.basename(yolo_model_path))[0]

        output_folder = os.path.join(os.path.dirname(total_video_path), model_name)
        os.makedirs(output_folder, exist_ok=True)

        # ✅ use total_video_path here
        base_name = os.path.splitext(os.path.basename(total_video_path))[0]
        output_video_path = os.path.join(output_folder, f"{base_name}_output.mp4")
        csv_file_path     = os.path.join(output_folder, f"{base_name}_output.csv")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

        display_width  = frame_width // 2
        display_height = frame_height // 2

        if view_videos:
            window_title = f"{os.path.basename(total_video_path)} - YOLOv8 Model"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_title, display_width, display_height)

        with open(csv_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Frame", "Object Name", "Track_ID", "X", "Y", "Width", "Height", "Confidence"])

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Slightly safer inference wrapper
                with torch.inference_mode():
                    results = model(frame)
                detections = results[0]

                detection_boxes = []
                for i, box in enumerate(detections.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = box[:4]
                    confidence = detections.boxes.conf.cpu().numpy()[i]
                    detection_boxes.append([x1, y1, x2, y2, confidence])

                detection_boxes = np.array(detection_boxes) if detection_boxes else np.empty((0, 5))
                img_info = [frame_height, frame_width]
                tracked_objects = tracker.update(detection_boxes, img_info, (frame_height, frame_width))

                track_id_to_class_id = {}

                for track in tracked_objects:
                    x1, y1, x2, y2 = map(int, track.tlbr)
                    track_id = track.track_id

                    if track_id not in track_id_to_class_id:
                        for i, det_box in enumerate(detections.boxes.xyxy.cpu().numpy()):
                            det_x1, det_y1, det_x2, det_y2 = det_box[:4]
                            iou = calculate_iou((x1, y1, x2, y2), (det_x1, det_y1, det_x2, det_y2))
                            if iou > 0.4:
                                track_id_to_class_id[track_id] = (
                                    int(detections.boxes.cls.cpu().numpy()[i]),
                                    detections.boxes.conf.cpu().numpy()[i]
                                )
                                break

                    class_id, confidence = track_id_to_class_id.get(track_id, (-1, 0.0))
                    object_name = model.names[class_id] if class_id != -1 else "Unknown"

                    xCenter = (x1 + x2) / 2 / frame_width
                    yCenter = (y1 + y2) / 2 / frame_height
                    width   = (x2 - x1) / frame_width
                    height  = (y2 - y1) / frame_height

                    if track_id not in active_tracks:
                        active_tracks[track_id] = {"class_name": object_name, "frames": [], "last_seen": frame_idx}

                    active_tracks[track_id]["frames"].append({
                        "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        "time": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                        "bbox": (xCenter, yCenter, width, height),
                        "confidence": confidence
                    })
                    active_tracks[track_id]["last_seen"] = frame_idx

                    csv_writer.writerow([frame_idx, object_name, track_id, xCenter, yCenter, width, height, confidence])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, object_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    label = f"ID: {track_id}, Conf: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # finalize aged-out tracks
                for tid, obs in list(active_tracks.items()):
                    if frame_idx - obs["last_seen"] > tracker.args.track_buffer:
                        ended_obs = active_tracks.pop(tid)
                        num_frames = ended_obs["frames"][-1]["frame"] - ended_obs["frames"][0]["frame"]
                        if num_frames < 30:
                            print(f"[OBS] Track {tid} discarded (only {num_frames} frames)")
                            continue

                        print(f"[OBS] Track {tid} finalized with {len(ended_obs['frames'])} frames")
                        keyframes = reduce_to_keyframes_v3_dirpad(ended_obs)
                        print(f"[OBS] Reduced to {len(keyframes)} keyframes")
                        plot_keyframe_reduction(ended_obs["frames"], keyframes, title=f"Track {tid}")

                        chosen_kf = pick_observation_time(ended_obs["frames"], dataType)
                        if chosen_kf:
                            print(f"[OBS] Time chosen at frame {chosen_kf['frame']} (dataset={dataType})")
                        else:
                            print(f"[OBS] No valid collision found, using fallback")

                        chosen_frame = chosen_kf["frame"] if chosen_kf else ended_obs["frames"][0]["frame"]

                        observation_payload = {
                            "session_id": session_ID,
                            "comname": keyframes[0]["comname"],
                            "taxserial": COMNAME_TO_TAXSERIAL.get(keyframes[0]["comname"]),
                            "count": 1,
                            "tc": frame_to_timecode(chosen_frame, frame_rate),
                            "frame": str(chosen_frame % int(frame_rate)),
                            # ✅ use total_video_path here
                            "video_source": os.path.basename(total_video_path),
                            "videoLocation": total_video_path,
                            "mediaPosition": frame_to_media_position(chosen_frame, frame_rate),
                            "actualPosition": frame_to_media_position(chosen_frame, frame_rate),
                            "keyframes": keyframes
                        }
                        api.create_observation(observation_payload)

                out.write(frame)

                if view_videos:
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        print(f"User terminated video preview for {os.path.basename(total_video_path)}.")
                        break

                frame_idx += 1

        cap.release()
        out.release()
        if view_videos:
            cv2.destroyWindow(window_title)
        print(f"Processing complete for {os.path.basename(total_video_path)}")
        print(f"Results saved to:\nVideo: {output_video_path}\nCSV: {csv_file_path}")

    except Exception as e:
        # during debugging, re-raise so you see the exact line
        print(f"Error processing video {total_video_path}: {e}")
        raise



def reduce_to_keyframes(endedObs, pos_thresh=0.005, size_thresh=0.01):
    """
    Reduce dense frame data to keyframes using a recursive Douglas–Peucker style simplification.
    Output is structured to match DB schema.

    :param frames: list of dicts with keys: framenum, time, bbox=(x,y,w,h)
    :param pos_thresh: max allowed deviation in x/y
    :param size_thresh: max allowed deviation in w/h
    :param subset: keyframe subset ID (string)
    :param comname: common name to tag with each keyframe
    :return: list of DB-ready keyframes
    """
    frames = endedObs["frames"]
    comname = endedObs["class_name"]
    if len(frames) <= 2:
        # Edge case: only mark start/end
        out = []
        for i, f in enumerate(frames):
            out.append({
                "subset": "1",
                "comname": comname,
                "type": "start" if i == 0 else "end",
                "framenum": f["frame"],
                "x": f["bbox"][0],
                "y": f["bbox"][1],
                "width": f["bbox"][2],
                "height": f["bbox"][3]
            })
        return out

    # Recursive Douglas–Peucker simplification
    def recursive_reduce(seq):
        if len(seq) <= 2:
            return [seq[0], seq[-1]]

        start, end = seq[0], seq[-1]
        max_err = 0
        index = 0
        for i in range(1, len(seq) - 1):
            err = position_size_error(seq[i], start, end)
            if err > max_err:
                max_err = err
                index = i

        if max_err > pos_thresh:
            left = recursive_reduce(seq[: index + 1])
            right = recursive_reduce(seq[index:])
            return left[:-1] + right
        else:
            return [start, end]

    reduced = recursive_reduce(frames)

    # Wrap into DB-ready dicts
    out = []
    for i, f in enumerate(reduced):
        out.append({
            "subset": "1",
            "comname": comname,
            "type": "start" if i == 0 else ("end" if i == len(reduced) - 1 else "middle"),
            "framenum": f["frame"],
            "x": f["bbox"][0],
            "y": f["bbox"][1],
            "width": f["bbox"][2],
            "height": f["bbox"][3]
        })
    return out

    # Recursive Douglas–Peucker style
    def recursive_reduce(seq):
        if len(seq) <= 2:
            return [seq[0], seq[-1]]

        start, end = seq[0], seq[-1]
        max_err = 0
        index = 0
        for i in range(1, len(seq) - 1):
            err = position_size_error(seq[i], start, end)
            if err > max_err:
                max_err = err
                index = i

        if max_err > pos_thresh:
            left = recursive_reduce(seq[: index + 1])
            right = recursive_reduce(seq[index:])
            return left[:-1] + right
        else:
            return [start, end]

    reduced = recursive_reduce(frames)

    # Tag keyframe types
    out = []
    for i, f in enumerate(reduced):
        f_out = dict(f)
        if i == 0:
            f_out["key_type"] = "start"
        elif i == len(reduced) - 1:
            f_out["key_type"] = "end"
        else:
            f_out["key_type"] = "middle"
        out.append(f_out)

    return out

def position_size_error(frame, start, end):
    """
    Compute how far 'frame' deviates from the straight-line interpolation
    between start and end, considering both position and size.

    :param frame: dict with time, bbox=(x, y, w, h)
    :param start: first frame dict
    :param end: last frame dict
    :return: max error across position and size
    """
    t = (frame["time"] - start["time"]) / (end["time"] - start["time"] + 1e-8)

    # Interpolated values at time t
    interp_x = start["bbox"][0] + t * (end["bbox"][0] - start["bbox"][0])
    interp_y = start["bbox"][1] + t * (end["bbox"][1] - start["bbox"][1])
    interp_w = start["bbox"][2] + t * (end["bbox"][2] - start["bbox"][2])
    interp_h = start["bbox"][3] + t * (end["bbox"][3] - start["bbox"][3])

    # Actual values
    x, y, w, h = frame["bbox"]

    # Errors (normalized units)
    pos_err = math.hypot(x - interp_x, y - interp_y)  # distance error
    size_err = max(abs(w - interp_w), abs(h - interp_h))  # biggest size deviation

    return max(pos_err, size_err)


def reduce_to_keyframes_v3_dirpad(
    endedObs,
    pos_thresh=0.005,
    size_thresh=0.01,
    # --- directional padding controls ---
    vel_window=2,          # use ±vel_window frames to estimate velocity (central difference)
    speed_gain=0.35,       # scales padding from speed → fraction of box size
    base_pad=0.02,         # optional always-on padding as a fraction of size (0–0.05 typical)
    max_pad=0.15           # clamp total padding fraction (protect against spikes)
):
    """
    Douglas–Peucker keyframe reducer with **direction-aware velocity padding**.

    What it does:
      1) Runs your original recursive simplification unchanged.
      2) For each resulting keyframe, estimates local velocity (vx, vy) of the bbox center
         using nearby raw frames (central difference).
      3) Expands the bbox **only on the side of motion**:
           - if vx > 0 (moving right): widen to the right
           - if vx < 0 (moving left):  widen to the left
           - if vy > 0 (moving down):  extend downward
           - if vy < 0 (moving up):    extend upward
         This prevents cutting off the object on the side it’s heading toward during
         non-linear or turning motion without adding extra keyframes.

    Parameters
    ----------
    endedObs : dict
        {"frames": [ { "frame", "time", "bbox"=(x,y,w,h) }, ... ],
         "class_name": str }
        NOTE: x,y are assumed to be bbox **center** in normalized 0..1 coords.
    pos_thresh, size_thresh : float
        Original Douglas–Peucker thresholds (same meaning as your baseline).
    vel_window : int
        How many frames to look backward/forward when estimating velocity.
        Effective difference uses indices i-vel_window and i+vel_window if available.
    speed_gain : float
        Converts |velocity| into padding fraction of current size. Increase if you still
        see clipping on the motion side; decrease if boxes get too generous.
    base_pad : float
        Optional always-on padding fraction (applied even when speed≈0). Leave at 0.00
        if you want purely velocity-driven padding.
    max_pad : float
        Upper clamp for the total per-axis padding fraction to avoid extreme growth.

    Returns
    -------
    list[dict]
        DB-ready keyframes with directional padding applied:
        {subset, comname, type (start/middle/end), framenum, x, y, width, height}
    """

    frames = endedObs["frames"]
    comname = endedObs["class_name"]

    # --- trivial short sequence: return start/end with directional pad using rough velocity ---
    if len(frames) <= 2:
        out = []
        for i, f in enumerate(frames):
            x, y, w, h = f["bbox"]
            vx, vy = 0.0, 0.0
            if len(frames) == 2:
                # estimate simple velocity from the other frame
                other = frames[1 - i]["bbox"]
                dt = abs(frames[1 - i]["time"] - f["time"]) + 1e-8
                vx = (other[0] - x) / dt
                vy = (other[1] - y) / dt
            x, y, w, h = _apply_directional_pad(x, y, w, h, vx, vy, base_pad, speed_gain, max_pad)
            out.append({
                "subset": "1",
                "comname": comname,
                "type": "start" if i == 0 else "end",
                "framenum": f["frame"],
                "x": x, "y": y, "width": w, "height": h
            })
        return out

    # --- original recursive DP (unchanged logic) ---------------------------------------------
    def position_size_error(frame, start, end):
        t = (frame["time"] - start["time"]) / (end["time"] - start["time"] + 1e-8)
        ix = start["bbox"][0] + t * (end["bbox"][0] - start["bbox"][0])
        iy = start["bbox"][1] + t * (end["bbox"][1] - start["bbox"][1])
        iw = start["bbox"][2] + t * (end["bbox"][2] - start["bbox"][2])
        ih = start["bbox"][3] + t * (end["bbox"][3] - start["bbox"][3])
        x, y, w, h = frame["bbox"]
        pos_err  = math.hypot(x - ix, y - iy)
        size_err = max(abs(w - iw), abs(h - ih))
        return max(pos_err, size_err)

    def recursive_reduce(seq):
        if len(seq) <= 2:
            return [seq[0], seq[-1]]
        start, end = seq[0], seq[-1]
        max_err, idx = 0.0, 0
        for i in range(1, len(seq) - 1):
            err = position_size_error(seq[i], start, end)
            if err > max_err:
                max_err, idx = err, i
        if max_err > pos_thresh:
            left  = recursive_reduce(seq[: idx + 1])
            right = recursive_reduce(seq[idx:])
            return left[:-1] + right
        else:
            return [start, end]

    reduced = recursive_reduce(frames)

    # --- map frame -> index to pull local raw-neighborhood quickly ---------------------------
    # frames are typically ascending by frame number; we build index for O(1) lookup
    frame_nums = [f["frame"] for f in frames]
    def _find_raw_index(frame_num):
        # binary search closest index with exact match fall-through
        j = bisect_left(frame_nums, frame_num)
        if j < len(frame_nums) and frame_nums[j] == frame_num:
            return j
        # fallback: nearest neighbor
        if j == 0: return 0
        if j == len(frame_nums): return len(frame_nums) - 1
        return j if abs(frame_nums[j] - frame_num) < abs(frame_nums[j-1] - frame_num) else j - 1

    # --- package as DB rows with **direction-aware padding** ---------------------------------
    out = []
    for i, kf in enumerate(reduced):
        x, y, w, h = kf["bbox"]

        # estimate local velocity from raw frames around this keyframe (central difference)
        center_idx = _find_raw_index(kf["frame"])
        i0 = max(0, center_idx - vel_window)
        i1 = min(len(frames) - 1, center_idx + vel_window)
        if i1 == i0:
            vx, vy = 0.0, 0.0
        else:
            x0, y0, t0 = frames[i0]["bbox"][0], frames[i0]["bbox"][1], frames[i0]["time"]
            x1, y1, t1 = frames[i1]["bbox"][0], frames[i1]["bbox"][1], frames[i1]["time"]
            dt = (t1 - t0) if (t1 - t0) != 0 else 1e-8
            vx, vy = (x1 - x0) / dt, (y1 - y0) / dt

        # apply directional padding: expand & shift toward motion side on each axis
        x, y, w, h = _apply_directional_pad(x, y, w, h, vx, vy, base_pad, speed_gain, max_pad)

        out.append({
            "subset": "1",
            "comname": comname,
            "type": ("start" if i == 0 else ("end" if i == len(reduced) - 1 else "middle")),
            "framenum": kf["frame"],
            "x": x, "y": y, "width": w, "height": h
        })

    return out


# -------- helper: directional padding applied to a (centered) bbox ---------------------------
def _apply_directional_pad(x, y, w, h, vx, vy, base_pad, speed_gain, max_pad):
    """
    Expand axis-aligned bbox on the **side of motion** for each axis independently.
    x,y are bbox centers in normalized image coordinates; w,h are normalized sizes.

    Padding on X:
        pad_x = clamp(base_pad + speed_gain * |vx|, 0, max_pad)
        if vx > 0 (moving right):  increase width by pad_x*w and shift center +pad_x*w/2
        if vx < 0 (moving left):   increase width by pad_x*w and shift center -pad_x*w/2

    Padding on Y:
        pad_y = clamp(base_pad + speed_gain * |vy|, 0, max_pad)
        if vy > 0 (moving down):   increase height by pad_y*h and shift center +pad_y*h/2
        if vy < 0 (moving up):     increase height by pad_y*h and shift center -pad_y*h/2
    """
    # compute per-axis padding fractions (clamped)
    pad_x = max(0.0, min(max_pad, base_pad + speed_gain * abs(vx)))
    pad_y = max(0.0, min(max_pad, base_pad + speed_gain * abs(vy)))

    # expand and shift on X toward motion side
    if pad_x > 0.0:
        dx = 0.5 * pad_x * w
        w  = w * (1.0 + pad_x)
        x += dx if vx > 0 else (-dx if vx < 0 else 0.0)

    # expand and shift on Y toward motion side
    if pad_y > 0.0:
        dy = 0.5 * pad_y * h
        h  = h * (1.0 + pad_y)
        y += dy if vy > 0 else (-dy if vy < 0 else 0.0)

    return x, y, w, h


def reduce_to_keyframes_v3_dirpad(endedObs,
                                  pos_thresh=0.005,
                                  size_thresh=0.01,
                                  vel_window=3,
                                  speed_gain=0.25,
                                  base_pad=0.00,
                                  max_pad=0.12):
    """
    Douglas–Peucker keyframe reducer with direction-aware velocity padding.

    This keeps your original simplification algorithm intact and simply
    adds a post-process that expands each keyframe’s bounding box toward
    the direction of motion (down, up, left, right).

    Parameters
    ----------
    endedObs : dict
        Observation with "frames" (each frame has frame, time, bbox=(x,y,w,h))
        and "class_name".
    pos_thresh, size_thresh : float
        Original Douglas–Peucker thresholds.
    vel_window : int
        Number of frames before/after the keyframe used to estimate velocity.
    speed_gain : float
        Converts absolute velocity into padding fraction of bbox size.
    base_pad : float
        Constant baseline padding fraction (0–0.05 typical).
    max_pad : float
        Clamp for total padding fraction to avoid excessive box growth.
    """

    frames = endedObs["frames"]
    comname = endedObs["class_name"]

    # ----------------------------------------------------------------------
    # Recursive simplifier (identical structure, just renamed error fn)
    # ----------------------------------------------------------------------
    def position_size_error_dir(frame, start, end):
        """Same math as original, distinct name for clarity."""
        t = (frame["time"] - start["time"]) / (end["time"] - start["time"] + 1e-8)
        ix = start["bbox"][0] + t * (end["bbox"][0] - start["bbox"][0])
        iy = start["bbox"][1] + t * (end["bbox"][1] - start["bbox"][1])
        iw = start["bbox"][2] + t * (end["bbox"][2] - start["bbox"][2])
        ih = start["bbox"][3] + t * (end["bbox"][3] - start["bbox"][3])
        x, y, w, h = frame["bbox"]
        pos_err = math.hypot(x - ix, y - iy)
        size_err = max(abs(w - iw), abs(h - ih))
        return max(pos_err, size_err)

    def recursive_reduce(seq):
        if len(seq) <= 2:
            return [seq[0], seq[-1]]
        start, end = seq[0], seq[-1]
        max_err, idx = 0.0, 0
        for i in range(1, len(seq) - 1):
            err = position_size_error_dir(seq[i], start, end)
            if err > max_err:
                max_err, idx = err, i
        if max_err > pos_thresh:
            left = recursive_reduce(seq[: idx + 1])
            right = recursive_reduce(seq[idx:])
            return left[:-1] + right
        else:
            return [start, end]

    reduced = recursive_reduce(frames)

    # Map frame numbers for quick raw-frame lookup
    frame_nums = [f["frame"] for f in frames]
    def find_raw_index(frame_num):
        j = bisect_left(frame_nums, frame_num)
        if j < len(frame_nums) and frame_nums[j] == frame_num:
            return j
        if j == 0: return 0
        if j == len(frame_nums): return len(frame_nums) - 1
        return j if abs(frame_nums[j] - frame_num) < abs(frame_nums[j-1] - frame_nums[j-1]) else j - 1

    # ----------------------------------------------------------------------
    # Build output list with direction-aware velocity padding
    # ----------------------------------------------------------------------
    out = []
    for i, kf in enumerate(reduced):
        x, y, w, h = kf["bbox"]

        # Estimate local velocity (vx, vy) from raw frames near this keyframe
        center_idx = find_raw_index(kf["frame"])
        i0 = max(0, center_idx - vel_window)
        i1 = min(len(frames) - 1, center_idx + vel_window)
        if i1 == i0:
            vx = vy = 0.0
        else:
            x0, y0, t0 = frames[i0]["bbox"][0], frames[i0]["bbox"][1], frames[i0]["time"]
            x1, y1, t1 = frames[i1]["bbox"][0], frames[i1]["bbox"][1], frames[i1]["time"]
            dt = (t1 - t0) if (t1 - t0) != 0 else 1e-8
            vx, vy = (x1 - x0) / dt, (y1 - y0) / dt

        # Apply directional padding based on motion
        x, y, w, h = _apply_directional_pad(x, y, w, h, vx, vy,
                                            base_pad=base_pad,
                                            speed_gain=speed_gain,
                                            max_pad=max_pad)

        out.append({
            "subset": "1",
            "comname": comname,
            "type": "start" if i == 0 else ("end" if i == len(reduced) - 1 else "middle"),
            "framenum": kf["frame"],
            "x": x, "y": y, "width": w, "height": h
        })

    return out


# --------------------------------------------------------------------------
# Helper: expand bbox toward direction of motion (center-based coords)
# --------------------------------------------------------------------------
def _apply_directional_pad(x, y, w, h, vx, vy, base_pad, speed_gain, max_pad):
    """
    Expand an (x,y,w,h) box on the side of motion using normalized units.
    """
    pad_x = min(max_pad, base_pad + speed_gain * abs(vx))
    pad_y = min(max_pad, base_pad + speed_gain * abs(vy))

    # Shift and scale width based on X velocity
    if pad_x > 0:
        dx = 0.5 * pad_x * w
        w *= (1 + pad_x)
        x += dx if vx > 0 else (-dx if vx < 0 else 0)

    # Shift and scale height based on Y velocity
    if pad_y > 0:
        dy = 0.5 * pad_y * h
        h *= (1 + pad_y)
        y += dy if vy > 0 else (-dy if vy < 0 else 0)

    return x, y, w, h


def reduce_to_keyframes_v2(endedObs, pos_thresh=0.005, size_thresh=0.01, pad_frac=0.05):
    """
    Stable Douglas–Peucker keyframe reducer with downward padding.

    This version fixes the common visual issue where generated keyframes
    slightly cut off the bottom of the object due to perspective and
    underestimation of downward motion.

    Instead of altering the simplification math, it simply expands each
    keyframe's bounding box downward by a small fraction of its own height.
    This approach is stable, requires no motion estimation, and avoids
    overfitting to noise.

    Parameters
    ----------
    endedObs : dict
        Contains:
          • "frames": list of dicts with keys:
                "frame", "time", "bbox" = (x, y, w, h)
          • "class_name": common name for the object.
    pos_thresh : float
        Maximum allowed deviation in x/y between interpolated and actual points.
    size_thresh : float
        Maximum allowed deviation in w/h between interpolated and actual points.
    pad_frac : float
        Fraction of each keyframe’s height used to extend the box downward.
        Typical range: 0.03–0.08 (3–8%).
    """

    frames = endedObs["frames"]
    comname = endedObs["class_name"]

    # ----------------------------------------------------------------------
    # Edge case: only start and end frames → mark both as keyframes
    # ----------------------------------------------------------------------
    if len(frames) <= 2:
        out = []
        for i, f in enumerate(frames):
            x, y, w, h = f["bbox"]

            # Downward padding applied even for short tracks
            y += pad_frac * h
            h *= (1 + pad_frac)

            out.append({
                "subset": "1",
                "comname": comname,
                "type": "start" if i == 0 else "end",
                "framenum": f["frame"],
                "x": x,
                "y": y,
                "width": w,
                "height": h
            })
        return out

    # ----------------------------------------------------------------------
    # Recursive Douglas–Peucker simplification
    # ----------------------------------------------------------------------
    def recursive_reduce(seq):
        """Recursively split sequence if deviation exceeds threshold."""
        if len(seq) <= 2:
            return [seq[0], seq[-1]]

        start, end = seq[0], seq[-1]
        max_err = 0.0
        index = 0

        for i in range(1, len(seq) - 1):
            err = position_size_error(seq[i], start, end)
            if err > max_err:
                max_err = err
                index = i

        if max_err > pos_thresh:
            left = recursive_reduce(seq[: index + 1])
            right = recursive_reduce(seq[index:])
            return left[:-1] + right
        else:
            return [start, end]

    reduced = recursive_reduce(frames)

    # ----------------------------------------------------------------------
    # Wrap into DB-ready keyframe dicts and apply downward padding
    # ----------------------------------------------------------------------
    out = []
    for i, f in enumerate(reduced):
        x, y, w, h = f["bbox"]

        # Apply small downward padding (extends box downward slightly)
        y += pad_frac * h
        h *= (1 + pad_frac)

        out.append({
            "subset": "1",
            "comname": comname,
            "type": (
                "start" if i == 0 else
                ("end" if i == len(reduced) - 1 else "middle")
            ),
            "framenum": f["frame"],
            "x": x,
            "y": y,
            "width": w,
            "height": h
        })

    return out



def reduce_to_keyframes_v5(endedObs, frame_width=1920, frame_height=1080,
                           pos_thresh_px=15, size_thresh_px=30, min_gap=3):
    """
    Curvature-aware Douglas–Peucker keyframe reducer (pixel-space version).

    Uses quadratic regression to model smooth acceleration in object motion
    (especially downward + enlarging movement caused by ROV approach).  This
    prevents premature tail cut-offs where linear interpolation would
    underestimate motion.

    Parameters
    ----------
    endedObs : dict
        Observation with "frames" (each has bbox=(x,y,w,h) in normalized [0–1])
        and "class_name".
    frame_width, frame_height : int
        Pixel dimensions used to convert normalized coordinates to pixels.
    pos_thresh_px, size_thresh_px : float
        Maximum allowed deviation in pixels before a new keyframe is required.
    min_gap : int
        Minimum number of frames between recursive splits.

    Returns
    -------
    list of dict
        Simplified keyframes ready for DB insertion.
    """

    frames = endedObs["frames"]
    comname = endedObs["class_name"]

    if len(frames) <= 2:
        return [
            {
                "subset": "1",
                "comname": comname,
                "type": "start" if i == 0 else "end",
                "framenum": f["frame"],
                "x": f["bbox"][0],
                "y": f["bbox"][1],
                "width": f["bbox"][2],
                "height": f["bbox"][3],
            }
            for i, f in enumerate(frames)
        ]

    # ----------------------------------------------------------------------
    # Helper: quadratic error in pixel space
    # ----------------------------------------------------------------------
    def position_size_error_quad(frame_seq, start_idx, end_idx, test_idx):
        """
        Fit a quadratic curve through (frame vs bbox_coord) for x,y,w,h between
        start_idx and end_idx, then compute pixel error of frame_seq[test_idx]
        from that curve.
        """
        idxs = [f["frame"] for f in frame_seq[start_idx:end_idx + 1]]
        xs = np.array([f["bbox"][0] * frame_width for f in frame_seq[start_idx:end_idx + 1]])
        ys = np.array([f["bbox"][1] * frame_height for f in frame_seq[start_idx:end_idx + 1]])
        ws = np.array([f["bbox"][2] * frame_width for f in frame_seq[start_idx:end_idx + 1]])
        hs = np.array([f["bbox"][3] * frame_height for f in frame_seq[start_idx:end_idx + 1]])

        # Fit quadratic polynomials (2nd order)
        def fit_quad(t, v): return np.poly1d(np.polyfit(t, v, 2))
        fpx, fpy, fpw, fph = fit_quad(idxs, xs), fit_quad(idxs, ys), fit_quad(idxs, ws), fit_quad(idxs, hs)

        test_frame = frame_seq[test_idx]
        t = test_frame["frame"]
        x_est, y_est = fpx(t), fpy(t)
        w_est, h_est = fpw(t), fph(t)

        x_act, y_act = test_frame["bbox"][0] * frame_width, test_frame["bbox"][1] * frame_height
        w_act, h_act = test_frame["bbox"][2] * frame_width, test_frame["bbox"][3] * frame_height

        pos_err = math.hypot(x_act - x_est, y_act - y_est)
        size_err = math.hypot(w_act - w_est, h_act - h_est)

        return max(pos_err / pos_thresh_px, size_err / size_thresh_px)

    # ----------------------------------------------------------------------
    # Recursive simplification
    # ----------------------------------------------------------------------
    def recursive_reduce_quad(seq, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = len(seq) - 1
        if end_idx - start_idx <= 2:
            return [seq[start_idx], seq[end_idx]]

        max_err, worst_idx = 0.0, start_idx
        for i in range(start_idx + 1, end_idx):
            err = position_size_error_quad(seq, start_idx, end_idx, i)
            if err > max_err:
                max_err, worst_idx = err, i

        if max_err > 1.0 and (worst_idx - start_idx) > min_gap and (end_idx - worst_idx) > min_gap:
            left = recursive_reduce_quad(seq, start_idx, worst_idx)
            right = recursive_reduce_quad(seq, worst_idx, end_idx)
            return left[:-1] + right
        else:
            return [seq[start_idx], seq[end_idx]]

    reduced = recursive_reduce_quad(frames)

    # Deduplicate consecutive identical frames
    deduped = []
    for f in reduced:
        if not deduped or f["frame"] != deduped[-1]["frame"]:
            deduped.append(f)

    # ----------------------------------------------------------------------
    # Output formatting
    # ----------------------------------------------------------------------
    out = []
    for i, f in enumerate(deduped):
        out.append({
            "subset": "1",
            "comname": comname,
            "type": (
                "start" if i == 0 else
                ("end" if i == len(deduped) - 1 else "middle")
            ),
            "framenum": f["frame"],
            "x": f["bbox"][0],
            "y": f["bbox"][1],
            "width": f["bbox"][2],
            "height": f["bbox"][3],
        })

    return out



def reduce_to_keyframes_v4(endedObs, frame_width=1920, frame_height=1080,
                           pos_thresh_px=15, size_thresh_px=30, min_gap=3):
    """
    Improved Douglas–Peucker keyframe reducer that works in pixel space.
    This avoids distortion from normalized coordinates and prevents the
    apparent "cut-off bottom" problem seen when objects accelerate
    downward in perspective.

    Parameters
    ----------
    endedObs : dict
        Observation with "frames" (each with bbox=(x,y,w,h) in normalized
        [0–1] coordinates) and "class_name".
    frame_width : int
        Pixel width of video frame (used to de-normalize coords).
    frame_height : int
        Pixel height of video frame (used to de-normalize coords).
    pos_thresh_px : float
        Max allowed deviation in pixels for (x,y) before adding keyframe.
    size_thresh_px : float
        Max allowed deviation in pixels for (w,h) before adding keyframe.
    min_gap : int
        Minimum frame distance between recursive splits.

    Returns
    -------
    list of dict
        Simplified keyframes formatted for DB.
    """

    frames = endedObs["frames"]
    comname = endedObs["class_name"]

    if len(frames) <= 2:
        # trivial case: just start & end
        return [
            {
                "subset": "1",
                "comname": comname,
                "type": "start" if i == 0 else "end",
                "framenum": f["frame"],
                "x": f["bbox"][0],
                "y": f["bbox"][1],
                "width": f["bbox"][2],
                "height": f["bbox"][3],
            }
            for i, f in enumerate(frames)
        ]

    # --- helper: compute error in pixel space ---
    def position_size_error_px(frame, start, end):
        # convert normalized to pixel space
        f_box = [frame["bbox"][0] * frame_width,
                 frame["bbox"][1] * frame_height,
                 frame["bbox"][2] * frame_width,
                 frame["bbox"][3] * frame_height]
        s_box = [start["bbox"][0] * frame_width,
                 start["bbox"][1] * frame_height,
                 start["bbox"][2] * frame_width,
                 start["bbox"][3] * frame_height]
        e_box = [end["bbox"][0] * frame_width,
                 end["bbox"][1] * frame_height,
                 end["bbox"][2] * frame_width,
                 end["bbox"][3] * frame_height]

        t = (frame["frame"] - start["frame"]) / (end["frame"] - start["frame"])
        interp = [s_box[j] + t * (e_box[j] - s_box[j]) for j in range(4)]

        dx = f_box[0] - interp[0]
        dy = f_box[1] - interp[1]
        dw = f_box[2] - interp[2]
        dh = f_box[3] - interp[3]

        pos_err = math.hypot(dx, dy)
        size_err = math.hypot(dw, dh)

        return max(pos_err / pos_thresh_px, size_err / size_thresh_px)

    # --- recursive simplification in pixel space ---
    def recursive_reduce_px(seq):
        if len(seq) <= 2:
            return [seq[0], seq[-1]]

        start, end = seq[0], seq[-1]
        max_err, index = 0.0, 0

        for i in range(1, len(seq) - 1):
            err = position_size_error_px(seq[i], start, end)
            if err > max_err:
                max_err, index = err, i

        if max_err > 1.0 and index > min_gap and len(seq) - index > min_gap:
            left = recursive_reduce_px(seq[: index + 1])
            right = recursive_reduce_px(seq[index:])
            return left[:-1] + right
        else:
            return [start, end]

    reduced = recursive_reduce_px(frames)

    # deduplicate consecutive identical frames
    deduped = []
    for f in reduced:
        if not deduped or f["frame"] != deduped[-1]["frame"]:
            deduped.append(f)

    # --- format back to normalized DB keyframes ---
    out = []
    for i, f in enumerate(deduped):
        out.append({
            "subset": "1",
            "comname": comname,
            "type": (
                "start" if i == 0 else
                ("end" if i == len(deduped) - 1 else "middle")
            ),
            "framenum": f["frame"],
            "x": f["bbox"][0],
            "y": f["bbox"][1],
            "width": f["bbox"][2],
            "height": f["bbox"][3],
        })

    return out


def plot_keyframe_reduction(frames, keyframes, title="Keyframe Reduction Debug"):
    """
    Plot full trajectory (all frames) vs reduced keyframes for visual debugging.

    :param frames: list of frame dicts with keys: framenum, time, bbox=(x,y,w,h)
    :param keyframes: list of DB-ready keyframe dicts with keys: x, y, framenum
    """
    # --- Dense trajectory ---
    all_x = [f["bbox"][0] for f in frames]
    all_y = [f["bbox"][1] for f in frames]
    times = [f["frame"] for f in frames]

    # --- Reduced keyframes ---
    key_x = [kf["x"] for kf in keyframes]
    key_y = [kf["y"] for kf in keyframes]
    key_times = [kf.get("time", kf["framenum"]) for kf in keyframes]  # use time if available

    plt.figure(figsize=(8, 6))

    # Dense lines
    plt.plot(times, all_x, label="X (all)", alpha=0.5, color="blue")
    plt.plot(times, all_y, label="Y (all)", alpha=0.5, color="green")

    # Keyframe points
    plt.scatter(key_times, key_x, label="X (keyframes)", color="red", marker="x")
    plt.scatter(key_times, key_y, label="Y (keyframes)", color="orange", marker="x")

    # Linear interpolation between keyframes
    if len(keyframes) > 1:
        for i in range(1, len(keyframes)):
            t0, t1 = key_times[i-1], key_times[i]
            x0, x1 = key_x[i-1], key_x[i]
            y0, y1 = key_y[i-1], key_y[i]

            plt.plot([t0, t1], [x0, x1], "--", color="red", alpha=0.6, linewidth=1)
            plt.plot([t0, t1], [y0, y1], "--", color="orange", alpha=0.6, linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Normalized position")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    #plt.show()


def plot_keyframe_reduction_old(frames, keyframes, title="Keyframe Reduction Debug"):
    """
    Plot full trajectory (all frames) vs reduced keyframes for visual debugging.
    
    Shows:
      - Dense path (all frames)
      - Keyframe positions
      - Linear interpolation between keyframes
    """
    # --- Extract from all frames ---
    all_x = [f["bbox"][0] for f in frames]
    all_y = [f["bbox"][1] for f in frames]
    times = [f["time"] for f in frames]

    # --- Extract from keyframes ---
    key_x = [f["bbox"][0] for f in keyframes]
    key_y = [f["bbox"][1] for f in keyframes]
    key_times = [f["time"] for f in keyframes]

    plt.figure(figsize=(8, 6))

    # Dense trajectory
    plt.plot(times, all_x, label="X (all)", alpha=0.5, color="blue")
    plt.plot(times, all_y, label="Y (all)", alpha=0.5, color="green")

    # Keyframe points
    plt.scatter(key_times, key_x, label="X (keyframes)", color="red", marker="x")
    plt.scatter(key_times, key_y, label="Y (keyframes)", color="orange", marker="x")

    # Linear interpolation between keyframes
    if len(keyframes) > 1:
        for i in range(1, len(keyframes)):
            t0, t1 = keyframes[i-1]["time"], keyframes[i]["time"]
            x0, x1 = keyframes[i-1]["bbox"][0], keyframes[i]["bbox"][0]
            y0, y1 = keyframes[i-1]["bbox"][1], keyframes[i]["bbox"][1]

            plt.plot([t0, t1], [x0, x1], "--", color="red", alpha=0.6, linewidth=1)
            plt.plot([t0, t1], [y0, y1], "--", color="orange", alpha=0.6, linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Normalized position")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    print("Please select the folder containing videos...")
    video_folder = select_folder()
    if not video_folder:
        print("No folder selected. Exiting...")
        return

    print("Please select your YOLOv8 model file...")
    yolo_model_path = filedialog.askopenfilename(
        title="Select YOLOv8 Model",
        filetypes=[("YOLOv8 Model", "*.pt")]
    )
    if not yolo_model_path:
        print("No YOLO model selected. Exiting...")
        return

    view_videos = messagebox.askyesno("View Videos", "Do you want to view the videos while processing?")

    video_files = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    if not video_files:
        print("No video files found in the selected folder.")
        return

    print(f"Found {len(video_files)} video(s) in the folder. Starting processing in batches of 2...")

    batch_size = 8
    for batch in batch_videos(video_files, batch_size):
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(process_video, video, yolo_model_path, view_videos)
                for video in batch
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in video processing task: {e}")



def frame_to_timecode(frame_idx, fps):
    seconds = frame_idx / fps
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    return f"{hh:02}:{mm:02}:{ss:02}"

def frame_to_media_position(frame_idx, fps):
    seconds = frame_idx / fps
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    ms = (seconds - int(seconds)) * 1000
    return f"{hh:02}:{mm:02}:{ss:02}.{int(ms):03}"


def pick_observation_time(frames, data_type):
    """
    Decide the observation frame/time based on collision rules.

    :param frames: list of dense frames (ended_obs["frames"]), 
                   each frame has {"frame", "time", "bbox": (x,y,w,h)}
    :param data_type: str, e.g. "Fish", "Invert"
    :return: chosen frame dict or None
    """

    if data_type in ("Fish", "GULF_Fish"):
        # Fish: first time center crosses near bottom (y > 0.8 normalized)
        for f in frames:
            xCenter, yCenter, w, h = f["bbox"]
            if yCenter > 0.8:  # bottom 20% of screen
                return f

    elif data_type in ("Invert", "GULF_Inverts"):
        # Inverts: when center enters trapezoid at bottom half
        candidate = None
        best_bottom = -1.0

        for f in frames:
            xCenter, yCenter, h = f["bbox"][0], f["bbox"][1], f["bbox"][3]

            if yCenter < 0.5:  # must be in bottom half
                continue

            # trapezoid shrinks as you go down
            half_width = 0.5 * (1 - yCenter)  
            x_min, x_max = 0.5 - half_width, 0.5 + half_width

            if x_min <= xCenter <= x_max:
                # bottom of bbox
                y_bottom = yCenter + h / 2
                if y_bottom <= 1.0 and y_bottom > best_bottom:
                    best_bottom = y_bottom
                    candidate = f

        if candidate:
            return candidate

    # Fallback: middle frame
    if frames:
        return frames[len(frames) // 2]
    return None

def pick_observation_time_old(keyframes, frame_height, frame_width, data_type):
    """
    Decide the observation 'time' (frame or timestamp) based on collision rules
    that vary by dataset type.

    :param keyframes: list of reduced keyframes (each has bbox + framenum + time)
    :param frame_height: video frame height
    :param frame_width: video frame width
    :param data_type: str, e.g. "fish", "inverts", "gulf fish", "gulf inverts"
    :return: chosen keyframe (dict) or None
    """
    # Example collision rules (adapt as needed)
    if data_type in ("Fish", "GULF_Fish"): 
        # Observation when fish crosses from top → bottom
        for kf in keyframes:
            y_center = kf["bbox"][1]
            if y_center > frame_height * 0.8:  # near bottom
                return kf

    elif data_type in ("Invert", "GULF_Inverts"):
        # Observation when invert leaves bottom bounds
        for kf in keyframes:
            y_center = kf["bbox"][1]
            if y_center < frame_height * 0.2:  # near top or leaving screen
                return kf

    # Fallback: pick the middle keyframe
    if keyframes:
        return keyframes[len(keyframes) // 2]
    return None

# -----------------------------
# Session Table Utilities
# -----------------------------

def add_session_row(row_index, session_id, dive, line, type_):
    """
    Add a single session row to the sessions table.
    Uses the global sessions_frame that was created in the GUI.
    """
    global sessions_frame

    def on_launch():
        global dataType

        print(f"[CALLBACK] Launching session {session_id} (Dive={dive}, Line={line}, Type={type_})")
        dataType = type_
        start_session(session_id)

    ctk.CTkButton(
        sessions_frame, text="Launch", width=70, command=on_launch
    ).grid(row=row_index, column=0, padx=5, pady=5, sticky="w")

    ctk.CTkLabel(sessions_frame, text=session_id).grid(row=row_index, column=1, padx=5, pady=5, sticky="w")
    ctk.CTkLabel(sessions_frame, text=dive).grid(row=row_index, column=2, padx=5, pady=5, sticky="w")
    ctk.CTkLabel(sessions_frame, text=line).grid(row=row_index, column=3, padx=5, pady=5, sticky="w")
    ctk.CTkLabel(sessions_frame, text=type_).grid(row=row_index, column=4, padx=5, pady=5, sticky="w")


def createSessionList(sessions: list):
    """
    Clear the sessions table and repopulate it with the given sessions.
    Uses the global sessions_frame that was created in the GUI.
    """
    global sessions_frame

    # Clear all rows except header
    clear_sessions(sessions_frame)

    # Add each session row
    for i, sess in enumerate(sessions, start=1):
        add_session_row(
            i,
            sess.get("session_id", ""),
            sess.get("dive", ""),
            sess.get("line", ""),
            sess.get("type", "")
        )


def clear_sessions(parent_frame):
    """
    Clear all session rows from the sessions table except the header row.
    Useful when reloading the table from the database.
    """
    for widget in parent_frame.winfo_children():
        info = widget.grid_info()
        if info["row"] != 0:  # preserve header row
            widget.destroy()


# -----------------------------
# Main Project Setup Window
# -----------------------------

def project_selection_window(projects):
    """
    Main GUI window for project setup.
    Allows the user to select:
      - Model file
      - Input video folder
      - Project name
      - Dive ID
      - Line ID
      - Dataset type
    Also displays a sessions table with Launch buttons.
    """

    # Configure global style
    ctk.set_appearance_mode("light")   # "dark" is also available
    ctk.set_default_color_theme("blue")

    # Create root window
    root = ctk.CTk()
    root.title("Project Setup")
    root.geometry("950x500")

    result = {}  # dictionary to store selections

    # -----------------------------
    # Layout: 2 main columns
    # -----------------------------
    root.grid_columnconfigure(0, weight=2)
    root.grid_columnconfigure(1, weight=3)
    root.grid_rowconfigure(0, weight=1)

    # -----------------------------
    # Left column: Form inputs
    # -----------------------------
    form_frame = ctk.CTkFrame(root)
    form_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    form_frame.grid_columnconfigure(0, weight=1)
    form_frame.grid_columnconfigure(1, weight=2)

    bold_font = ("TkDefaultFont", 11, "bold")

    # --- Model file ---
    def on_model_chosen(path):
        global user_id
        global project_id
        global model_path

        """Callback when a model file is selected."""
        print(f"[CALLBACK] Model chosen: {os.path.basename(path)}")

        # Call API to get user by name
        user = api.get_user(f"{os.path.basename(path)}")

        # Check if we got a result
        if user:
            print("User found:", user)
            print("User ID:", user["user_id"])
            print("Created At:", user["createdAt"])
        else:
            print("User not found. Creating... ")

            user = api.create_user_by_name(f"{os.path.basename(path)}")

        user_id = str(user["user_id"])
        print("User ID set to " + user_id)

        if user_id is not None and project_id is not None:
            sessions = api.get_sessions_by_user_and_project(user_id, project_id)
            createSessionList(sessions)

        model_path = path



    model_label = ctk.CTkLabel(form_frame, text="Model File:", font=bold_font)
    model_label.grid(row=0, column=0, sticky="e", padx=10, pady=(20, 5))
    model_var = ctk.StringVar(value="Browse...")

    def browse_model():
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("YOLO Model", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            filename = os.path.basename(file_path)
            model_var.set(filename)
            result["model_path"] = file_path
            on_model_chosen(file_path)

    model_btn = ctk.CTkButton(form_frame, textvariable=model_var, command=browse_model)
    model_btn.grid(row=0, column=1, sticky="ew", padx=10, pady=(20, 5))

    # --- Video folder ---
    def on_video_chosen(videoWithPath):
        global video_path
        """Callback when a video folder is selected."""
        if videoWithPath:
            video_name = os.path.basename(videoWithPath)
            video_var.set(videoWithPath)
            video_path = videoWithPath

        print(f"[CALLBACK] Video folder chosen: {video_path}")

    video_label = ctk.CTkLabel(form_frame, text="Input Video:", font=bold_font)
    video_label.grid(row=1, column=0, sticky="e", padx=10, pady=5)
    video_var = ctk.StringVar(value="Browse...")

    def browse_videos():
        file_path = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        

        on_video_chosen(file_path)

    video_btn = ctk.CTkButton(form_frame, textvariable=video_var, command=browse_videos)
    video_btn.grid(row=1, column=1, sticky="ew", padx=10, pady=5)

    def on_project_changed(choice):
        """Callback when project selection changes."""
        global project_id
        print(f"[CALLBACK] Project selected: {choice}")

        # Find the matching project object and update the global project_id
        for p in projects:
            if p["name"] == choice:
                project_id = p["project_id"]
                print(f"[CALLBACK] Project ID set to {project_id}")
                break

        if user_id is not None and project_id is not None:
            sessions = api.get_sessions_by_user_and_project(user_id, project_id)
            createSessionList(sessions)

        

    project_label = ctk.CTkLabel(form_frame, text="Project:", font=bold_font)
    project_label.grid(row=2, column=0, sticky="e", padx=10, pady=5)

    # Extract project names from existing projects object
    project_names = [p["name"] for p in projects] if projects else []

    # Default selection: leave empty if no projects, otherwise first project name
    default_project = project_names[0] if project_names else ""
    project_var = ctk.StringVar(value=default_project)

    # Dropdown of project names
    project_dropdown = ctk.CTkComboBox(
        form_frame,
        variable=project_var,
        values=project_names,
        command=on_project_changed
    )
    project_dropdown.grid(row=2, column=1, sticky="ew", padx=10, pady=5)

    # --- Dive ---
    def on_dive_changed(*_):
        """Callback when dive entry is modified."""
        print(f"[CALLBACK] Dive entered: {dive_var.get()}")
        global dive

        dive = str(dive_var.get())

    dive_label = ctk.CTkLabel(form_frame, text="Dive:", font=bold_font)
    dive_label.grid(row=3, column=0, sticky="e", padx=10, pady=5)
    dive_var = ctk.StringVar()
    dive_var.trace_add("write", on_dive_changed)
    dive_entry = ctk.CTkEntry(form_frame, textvariable=dive_var)
    dive_entry.grid(row=3, column=1, sticky="ew", padx=10, pady=5)

    # --- Line ---
    def on_line_changed(*_):
        """Callback when line entry is modified."""
        print(f"[CALLBACK] Line entered: {line_var.get()}")
        global line

        line = str(line_var.get())

    line_label = ctk.CTkLabel(form_frame, text="Line:", font=bold_font)
    line_label.grid(row=4, column=0, sticky="e", padx=10, pady=5)
    line_var = ctk.StringVar()
    line_var.trace_add("write", on_line_changed)
    line_entry = ctk.CTkEntry(form_frame, textvariable=line_var)
    line_entry.grid(row=4, column=1, sticky="ew", padx=10, pady=5)

    # --- Dataset type ---
    def on_dataset_changed():
        """Callback when dataset type changes."""
        print(f"[CALLBACK] Dataset type selected: {dataset_var.get()}")

        global dataType

        dataType = str(dataset_var.get())

    dataset_label = ctk.CTkLabel(form_frame, text="Dataset Type:", font=bold_font)
    dataset_label.grid(row=5, column=0, sticky="ne", padx=10, pady=5)
    dataset_var = ctk.StringVar(value="fish")
    dataset_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
    dataset_frame.grid(row=5, column=1, sticky="w", padx=10, pady=5)
    for opt in ["Fish", "Invert", "GULF_Fish", "GULF_Inverts"]:
        ctk.CTkRadioButton(dataset_frame, text=opt, variable=dataset_var,
                           value=opt, command=on_dataset_changed).pack(side="left", padx=10)

    # -----------------------------
    # Right column: Sessions table
    # -----------------------------
    global sessions_frame   # make sessions_frame available globally
    sessions_frame = ctk.CTkFrame(root)
    sessions_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
    sessions_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

    # Header row
    headers = ["Launch", "SessionID", "Dive", "Line", "Type"]
    for col, h in enumerate(headers):
        label = ctk.CTkLabel(sessions_frame, text=h, font=bold_font)
        label.grid(row=0, column=col, padx=5, pady=5, sticky="w")

    # Example: populate fake sessions
    #add_session_row(sessions_frame, 1, "S001", "D01", "L01", "fish")
    #add_session_row(sessions_frame, 2, "S002", "D02", "L02", "inverts")
    #add_session_row(sessions_frame, 3, "S003", "D03", "L01", "gulf fish")




    # -----------------------------
    # Bottom buttons
    # -----------------------------
    def on_start():
        """Callback for Start button."""

        global session 
        global project_id
        global user_id
        global dive
        global line
        global dataType 

        print("[CALLBACK] Start clicked")
        chosen_project = project_var.get().strip()
        if not chosen_project:
            messagebox.showerror("Error", "Please select or enter a project.")
            return
        if model_var.get() == "Browse...":
            messagebox.showerror("Error", "Please select a model file.")
            return
        if video_var.get() == "Browse...":
            messagebox.showerror("Error", "Please select a video folder.")
            return
        result.update({
            "project": chosen_project,
            "dive": dive_var.get().strip(),
            "line": line_var.get().strip(),
            "model_name": model_var.get().strip(),
            "dataset_type": dataset_var.get(),
            "video_folder": result.get("video_folder", "")
        })

        session = api.create_session(
            project_id,
            user_id,
            dive,
            line,
            dataType
        )
        print("[CALLBACK] Cancel clicked")

        start_session(session["session_id"])

    def on_cancel():
        """Callback for Cancel button."""
        print("[CALLBACK] Cancel clicked")
        root.destroy()

    btn_frame = ctk.CTkFrame(root, fg_color="transparent")
    btn_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="e", padx=20)
    cancel_btn = ctk.CTkButton(btn_frame, text="Cancel", command=on_cancel, fg_color="gray")
    cancel_btn.pack(side="right", padx=10)
    start_btn = ctk.CTkButton(btn_frame, text="Start", command=on_start)
    start_btn.pack(side="right")

    root.mainloop()
    return result if result else None


def start_session(session_id: int):
    """
    Start a session:
      1. Fetch last observation info
      2. Use it to resume video processing if available
    """
    global session_ID
    session_ID = session_id
    last_info = api.get_last_video_info(session_id)
    if last_info and last_info[0].get("videoLocation") and last_info[0].get("mediaPosition"):
        # Pull just the filename from DB path
        video_name = os.path.basename(last_info[0]["videoLocation"])
        # Build full path using the existing video_path (user-chosen folder)
        video_file = os.path.join(os.path.dirname(video_path) , video_name)

        start_time = parse_timecode(last_info[0]["mediaPosition"])
        print(f"[SESSION] Resuming from {video_file} at {start_time:.2f}s")
    else:
        video_file = video_path
        start_time = None
        print(f"[SESSION] Starting new at {video_file}")

    view_videos = True
    process_video(video_file, model_path, view_videos, start_time=start_time)


if __name__ == "__main__":
    fake_projects = [{ "project_id": 1, "name": "PROJECTA", "createdAt": "2023-01-11T19:08:21.364Z", "updatedAt": "2023-01-11T19:08:21.364Z"}]
    projects = api.get_projects()
    selections = project_selection_window(projects)
    print("Selections:", selections)
    
    
    
    
    
    
    
    
    #main()
