"""
Database Video Annotations Manager

This module provides tools to manage and process video annotations. It includes
the ability to store annotations, sort them by frame number, retrieve annotations
based on frame context, and reload data from a database-like structure.

Classes:
- KeyframeList: Manages keyframe annotations for a single observation.
- DatabaseVideoAnnotationsRangeFinder: Handles the collection of all keyframe lists
  and provides querying capabilities.
"""

from collections import namedtuple
from sortedcontainers import SortedList
from typing import List, Dict, Tuple, Optional

# Define AnnotationRectangle as a namedtuple for better readability and field access.
AnnotationRectangle = namedtuple(
    "AnnotationRectangle",
    ["x_center", "y_center", "width_norm", "height_norm", "class_name", "type", "observation_id", "subset", "framenum"]
)


class KeyframeList:
    """
    KeyframeList manages a list of keyframe annotations for a single observation.
    Keyframes are sorted by their frame number (framenum).
    """

    def __init__(self):
        """
        Initializes a KeyframeList with a sorted list of frames and an optional end frame.
        """
        # SortedList to store frames, sorted by the `framenum` field.
        self.frames = SortedList(key=lambda x: x.framenum)
        # Optionally store an end frame separately.
        self.end_frame: Optional[AnnotationRectangle] = None

    def get_previous_and_next_annotation(self, frame_num: int) -> Tuple[Optional[AnnotationRectangle], Optional[AnnotationRectangle]]:
        """
        Retrieves the previous and next annotations for a given frame number.

        Args:
        - frame_num (int): The current frame number.

        Returns:
        - Tuple[Optional[AnnotationRectangle], Optional[AnnotationRectangle]]:
        A tuple containing the previous annotation and the next annotation.
        Either may be None if no suitable annotations exist.
        """
        if not self.frames:
            return None, None  # No frames available.

        previous = None
        next_frame = None

        last_loop_frame = None

        # Iterate through sorted frames to find the previous and next annotations.
        for loop_num, frame in enumerate(self.frames):
            if frame.framenum > frame_num:
                # Set the previous annotation if we are beyond the first frame
                previous = last_loop_frame if loop_num > 0 else None
                next_frame = frame
                return previous, next_frame
            last_loop_frame = frame

        # If frame_num exceeds all frames in self.frames
        if self.end_frame and frame_num < self.end_frame.framenum:
            previous = self.frames[-1] if self.frames else None
            next_frame = self.end_frame
        elif self.end_frame and frame_num >= self.end_frame.framenum:
            previous = self.end_frame
            next_frame = None
        else:
            previous = self.frames[-1] if self.frames else None
            next_frame = None

        return previous, next_frame



class DatabaseVideoAnnotationsRangeFinder:
    """
    DatabaseVideoAnnotationsRangeFinder manages all keyframe lists for a video dataset.
    It provides methods to query annotations, retrieve surrounding annotations, and reload data.
    """

    def __init__(self):
        """
        Initializes the DatabaseVideoAnnotationsRangeFinder with an empty list of keyframe lists.
        """
        self.list: Dict[str, KeyframeList] = {}

    def get_all_annotations(self) -> List[AnnotationRectangle]:
        """
        Retrieves all annotations across all keyframe lists.

        Returns:
        - List[AnnotationRectangle]: A list of all annotations.
        """
        result = []

        # Collect annotations from each keyframe list.
        for keyframe_list in self.list.values():
            result.extend(keyframe_list.frames)

            if keyframe_list.end_frame:
                result.append(keyframe_list.end_frame)

        return result

    def get_annotations_by_key(self, key: str) -> List[AnnotationRectangle]:
        """
        Retrieves all annotations for a specific key (observation ID and subset).

        Args:
        - key (str): The key identifying a specific observation.

        Returns:
        - List[AnnotationRectangle]: A list of annotations for the key.
        """
        result = []
        try:
            keyframe_list = self.list[key]
            result.extend(keyframe_list.frames)

            if keyframe_list.end_frame:
                result.append(keyframe_list.end_frame)
        except KeyError as ex:
            print(f"Error: {ex}")  # Key not found, return an empty list.

        return result

    def get_surrounding_annotations(self, curr_frame: int) -> Dict[str, Tuple[Optional[AnnotationRectangle], Optional[AnnotationRectangle]]]:
        """
        Retrieves surrounding annotations (previous and next) for a given frame number.

        Args:
        - curr_frame (int): The current frame number.

        Returns:
        - Dict[str, Tuple[Optional[AnnotationRectangle], Optional[AnnotationRectangle]]]:
          A dictionary where each key is an observation key, and the value is a tuple
          of (previous, next) annotations.
        """
        result = {}

        # For each keyframe list, get the surrounding annotations.
        for key, keyframe_list in self.list.items():
            previous, next_frame = keyframe_list.get_previous_and_next_annotation(curr_frame)
            result[key] = (previous, next_frame)

        return result

    def reload(self, database_video_annotations: Dict[int, List[AnnotationRectangle]]):
        """
        Reloads annotations from a given database structure into keyframe lists.

        Args:
        - database_video_annotations (Dict[int, List[AnnotationRectangle]]):
          A dictionary where the key is the frame number, and the value is a list of annotations.
        """
        self.list = {}  # Reset the list of keyframe lists.

        # Populate the keyframe lists from the database annotations.
        for frame_num, annotations in database_video_annotations.items():
            for rect in annotations:
                # Generate the observation key based on observation ID and subset.
                observation_id, subset, rect_type = rect.observation_id, rect.subset, rect.type
                new_key = f"{observation_id}_{subset}"

                # If the key is not in the list, create a new KeyframeList.
                if new_key not in self.list:
                    self.list[new_key] = KeyframeList()

                keyframe_list = self.list[new_key]

                # Add the annotation to the appropriate place (end frame or regular frame).
                if rect_type == "end":
                    keyframe_list.end_frame = rect
                else:
                    keyframe_list.frames.add(rect)
