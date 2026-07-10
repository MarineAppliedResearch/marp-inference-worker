# -----------------------------------------------------------------------------
# File name: video_source_resolver.py
# Date created: 2026-07-09
# Author: Isaac Travers
#
# Video source resolver utilities for the MARP inference worker.
# This file converts database video_source values into local paths or Jellyfin
# stream URLs that OpenCV and future video jobs can open.
# -----------------------------------------------------------------------------

# dataclasses provides lightweight typed containers for resolver results.
from dataclasses import dataclass

# os provides local path checks and filename parsing.
import os

# re provides conservative filename normalization helpers.
import re

# typing provides explicit optional return types.
from typing import Optional

# JellyfinClient owns authenticated Jellyfin API calls and stream URL construction.
from marp_inference_worker.media.jellyfin_client import JellyfinClient

# JellyfinItem stores normalized Jellyfin item metadata returned by the client.
from marp_inference_worker.media.jellyfin_client import JellyfinItem


# VideoSourceResolution stores the selected source for one database video.
@dataclass
class VideoSourceResolution:
    
    # Original database video_source value used as the resolver input.
    requested_video_source: str

    # Resolved path or URL that OpenCV can attempt to open.
    resolved_source: str

    # Source type: local_path or jellyfin_stream.
    source_type: str

    # Jellyfin item selected by the resolver, when Jellyfin was used.
    jellyfin_item: Optional[JellyfinItem] = None

    # Score used to pick this source from Jellyfin search results.
    match_score: int = 0

    # Search term that produced the selected Jellyfin item.
    search_term: str = ""


# VideoSourceResolver resolves MARP database video names into readable sources.
class VideoSourceResolver:
    
    # Creates a resolver around an optional Jellyfin client.
    # Inputs are a Jellyfin client and whether local paths should be preferred.
    # Output is a resolver instance.
    # Use this from dataset building and future video processing jobs.
    def __init__(
        self,
        jellyfin_client: Optional[JellyfinClient] = None,
        prefer_local: bool = True,
    ):
        
        # The Jellyfin client is optional so local-only workflows still work.
        self.jellyfin_client = jellyfin_client

        # Local preference preserves the old dataset-builder behavior first.
        self.prefer_local = prefer_local

    # Resolves one database video_source to a local path or Jellyfin stream URL.
    # Inputs are the database value and optional local folder.
    # Output is VideoSourceResolution or None if not found.
    # Use this before opening cv2.VideoCapture.
    def resolve(
        self,
        video_source: str,
        local_video_folder: Optional[str] = None,
    ) -> Optional[VideoSourceResolution]:
        
        # Blank database values cannot be resolved.
        if not video_source:
            return None

        # Prefer an existing local file when requested.
        if self.prefer_local:
            local_resolution = self._resolve_local_path(video_source, local_video_folder)

            if local_resolution is not None:
                return local_resolution

        # Fall back to Jellyfin when a client is available.
        if self.jellyfin_client is not None:
            return self.resolve_jellyfin_stream(video_source)

        return None

    # Resolves a database video_source to an existing local file.
    # Inputs are the database value and optional selected video folder.
    # Output is VideoSourceResolution or None.
    # Use this to preserve legacy local-folder dataset building.
    def _resolve_local_path(
        self,
        video_source: str,
        local_video_folder: Optional[str],
    ) -> Optional[VideoSourceResolution]:
        
        # Direct absolute or relative file paths should work as-is.
        if os.path.isfile(video_source):
            return VideoSourceResolution(
                requested_video_source=video_source,
                resolved_source=video_source,
                source_type="local_path",
            )

        # A selected folder is required for legacy folder-relative lookup.
        if not local_video_folder:
            return None

        # Ignore placeholder GUI values such as "No folder selected".
        if not os.path.isdir(local_video_folder):
            return None

        # The old dataset builder assumes video_source is a file inside this folder.
        candidate_path = os.path.join(local_video_folder, video_source)

        # Return the folder-relative path only if it exists.
        if os.path.isfile(candidate_path):
            return VideoSourceResolution(
                requested_video_source=video_source,
                resolved_source=candidate_path,
                source_type="local_path",
            )

        return None

    # Resolves a database video_source to a Jellyfin direct stream URL.
    # Input is the database video_source value.
    # Output is VideoSourceResolution or None.
    # Use this when the local video is missing or Jellyfin streaming is desired.
    def resolve_jellyfin_stream(self, video_source: str) -> Optional[VideoSourceResolution]:
        
        # A Jellyfin client is required for stream resolution.
        if self.jellyfin_client is None:
            return None

        # Authenticated Jellyfin state is required for search and stream URLs.
        if not self.jellyfin_client.is_authenticated:
            raise RuntimeError("Jellyfin client must be authenticated before resolving streams.")

        # Build staged search terms from the database video_source.
        search_terms = build_video_search_terms(video_source)

        # Track the best match found across all search terms.
        best_item = None

        # Track the best match score found across all search terms.
        best_score = 0

        # Track which search term produced the best match.
        best_search_term = ""

        # Evaluate each search term until a strong match is found.
        for search_term in search_terms:
            matches = self.jellyfin_client.search_video_items(search_term, limit=20)

            for item in matches:
                score = score_jellyfin_video_match(video_source, item)

                if score > best_score:
                    best_item = item
                    best_score = score
                    best_search_term = search_term

            # Stop early when the resolver finds a very strong filename/path match.
            if best_score >= 95:
                break

        # Require a meaningful match to avoid opening the wrong Jellyfin video.
        if best_item is None or best_score < 60:
            return None

        # Build the original/direct stream URL first.
        stream_url = self.jellyfin_client.build_direct_stream_url(best_item.item_id)

        return VideoSourceResolution(
            requested_video_source=video_source,
            resolved_source=stream_url,
            source_type="jellyfin_stream",
            jellyfin_item=best_item,
            match_score=best_score,
            search_term=best_search_term,
        )


# Builds staged Jellyfin search terms from a database video_source value.
# Input is a filename, path, or saved database video source.
# Output is a de-duplicated list of search terms.
# Use this before querying Jellyfin SearchTerm.
def build_video_search_terms(video_source: str) -> list[str]:
    
    # Normalize whitespace around the raw source value.
    raw_value = str(video_source or "").strip()

    # Empty input produces no usable search terms.
    if not raw_value:
        return []

    # Use only the filename part if a path was stored.
    basename = os.path.basename(raw_value)

    # Remove the extension because Jellyfin item names often omit it.
    stem, _extension = os.path.splitext(basename)

    # Jellyfin names from the current server may replace spaces with underscores.
    underscore_stem = stem.replace(" ", "_")

    # Some database values may store underscores while Jellyfin shows spaces.
    space_stem = stem.replace("_", " ")

    # Timestamp prefixes are often distinctive in MARE video filenames.
    timestamp_match = re.search(r"\d{8}[_ -]\d{6}", stem)

    # Collect candidates from most specific to broadest.
    candidates = [
        raw_value,
        basename,
        stem,
        underscore_stem,
        space_stem,
    ]

    # Add the timestamp prefix if one was found.
    if timestamp_match:
        candidates.append(timestamp_match.group(0).replace(" ", "_"))
        candidates.append(timestamp_match.group(0).replace("_", " "))

    # Return candidates without duplicates or blanks.
    return _dedupe_nonblank(candidates)


# Scores how well one Jellyfin item matches a database video_source.
# Inputs are the requested database value and one Jellyfin item.
# Output is an integer match score.
# Use this to avoid picking the wrong Jellyfin search result.
def score_jellyfin_video_match(video_source: str, item: JellyfinItem) -> int:
    
    # Normalize the requested filename and stem.
    requested_basename = os.path.basename(str(video_source or ""))
    requested_stem = os.path.splitext(requested_basename)[0]

    # Normalize Jellyfin item name and path basename.
    item_name = item.name or ""
    item_path_basename = os.path.basename(item.path or "")
    item_path_stem = os.path.splitext(item_path_basename)[0]

    # Build normalized comparison keys.
    requested_key = normalize_video_match_key(requested_stem)
    requested_file_key = normalize_video_match_key(requested_basename)
    item_name_key = normalize_video_match_key(item_name)
    item_path_key = normalize_video_match_key(item_path_stem)
    item_path_file_key = normalize_video_match_key(item_path_basename)

    # Exact normalized stem match is the strongest common case.
    if requested_key and requested_key == item_name_key:
        return 100

    # Exact normalized path stem match is also a strong match.
    if requested_key and requested_key == item_path_key:
        return 98

    # Exact normalized full filename match is strong when the path includes extension.
    if requested_file_key and requested_file_key == item_path_file_key:
        return 96

    # Contained stem matches are useful when Jellyfin adds prefixes or suffixes.
    if requested_key and requested_key in item_name_key:
        return 85

    # Path basename containment is slightly weaker but still useful.
    if requested_key and requested_key in item_path_key:
        return 82

    # Reverse containment catches cases where database names include extra suffixes.
    if item_name_key and item_name_key in requested_key:
        return 75

    # Timestamp matches are useful but not enough to beat exact names.
    requested_timestamp = extract_timestamp_key(requested_stem)
    item_timestamp = extract_timestamp_key(item_name + " " + item_path_basename)

    if requested_timestamp and requested_timestamp == item_timestamp:
        return 70

    return 0


# Normalizes video names for matching across database and Jellyfin values.
# Input is a filename or Jellyfin item name.
# Output is a lowercase comparison key.
# Use this before equality or containment checks.
def normalize_video_match_key(value: str) -> str:
    
    # Lowercase and strip surrounding whitespace.
    normalized = str(value or "").strip().lower()

    # Remove common video extensions before comparing names.
    normalized = re.sub(r"\.(mp4|mov|mkv|avi|m4v)$", "", normalized)

    # Treat spaces, underscores, and hyphens as equivalent separators.
    normalized = re.sub(r"[\s_\-]+", "", normalized)

    # Remove remaining characters that are not useful for filename matching.
    normalized = re.sub(r"[^a-z0-9]", "", normalized)

    return normalized


# Extracts a normalized timestamp key from a video name.
# Input is a filename, item name, or path basename.
# Output is yyyymmddhhmmss or blank.
# Use this as a broad fallback match signal.
def extract_timestamp_key(value: str) -> str:
    
    # Search for common MARE video timestamp patterns.
    match = re.search(r"(\d{8})[_ -](\d{6})", str(value or ""))

    # Return blank when no timestamp exists.
    if not match:
        return ""

    return match.group(1) + match.group(2)


# De-duplicates a list while preserving order.
# Input is a list of possible search terms.
# Output is a list without blanks or duplicates.
# Use this before staged Jellyfin searching.
def _dedupe_nonblank(values: list[str]) -> list[str]:
    
    # Track normalized values already emitted.
    seen = set()

    # Store terms in first-seen order.
    result = []

    for value in values:
        # Normalize only whitespace for de-duplication.
        cleaned = str(value or "").strip()

        # Skip blank values.
        if not cleaned:
            continue

        # Use case-insensitive de-duplication.
        key = cleaned.lower()

        # Skip repeated values.
        if key in seen:
            continue

        seen.add(key)
        result.append(cleaned)

    return result