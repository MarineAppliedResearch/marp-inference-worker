# -----------------------------------------------------------------------------
# File name: jellyfin_client.py
# Date created: 2026-07-09
# Author: Isaac Travers
#
# Reusable Jellyfin API client for the MARP inference worker.
# This file owns low-level Jellyfin authentication, item search, playback info,
# and stream URL construction for future video processing jobs.
# -----------------------------------------------------------------------------

# dataclasses provides lightweight typed response containers for Jellyfin data.
from dataclasses import dataclass

# os reads optional Jellyfin connection settings from environment variables.
import os

# typing provides explicit return types for optional and list-based data.
from typing import Any, Optional

# urllib.parse safely encodes Jellyfin URL path and query values.
from urllib.parse import quote

# httpx provides HTTP client support for Jellyfin API requests.
import httpx


# JellyfinMediaStream stores the video stream fields needed for quality decisions.
@dataclass
class JellyfinMediaStream:
    
    # Jellyfin stream type, usually Video, Audio, or Subtitle.
    stream_type: str = ""

    # Codec name reported by Jellyfin.
    codec: str = ""

    # Video width in pixels when this is a video stream.
    width: Optional[int] = None

    # Video height in pixels when this is a video stream.
    height: Optional[int] = None

    # Stream bitrate when Jellyfin reports it.
    bit_rate: Optional[int] = None

    # Average frame rate when Jellyfin reports it.
    average_frame_rate: Optional[float] = None

    # Real frame rate when Jellyfin reports it.
    real_frame_rate: Optional[float] = None


# JellyfinMediaSource stores the playable source information for one item.
@dataclass
class JellyfinMediaSource:
    
    # Jellyfin media source ID used by playback and stream endpoints.
    source_id: str = ""

    # Server-side source path when Jellyfin includes it.
    path: str = ""

    # Container name, such as mp4, mov, or mkv.
    container: str = ""

    # Human-readable media source name.
    name: str = ""

    # Protocol reported by Jellyfin for the source.
    protocol: str = ""

    # Source file size in bytes when available.
    size: Optional[int] = None

    # Source bitrate when available.
    bitrate: Optional[int] = None

    # Runtime in Jellyfin ticks when available.
    runtime_ticks: Optional[int] = None

    # Whether Jellyfin says direct play is possible.
    supports_direct_play: Optional[bool] = None

    # Whether Jellyfin says direct stream is possible.
    supports_direct_stream: Optional[bool] = None

    # Whether Jellyfin says transcoding is possible.
    supports_transcoding: Optional[bool] = None

    # Relative or absolute direct stream URL returned by PlaybackInfo.
    direct_stream_url: str = ""

    # Relative or absolute transcode URL returned by PlaybackInfo.
    transcoding_url: str = ""

    # Jellyfin transcode subprotocol, often hls.
    transcoding_sub_protocol: str = ""

    # Jellyfin transcode container, often ts.
    transcoding_container: str = ""

    # Media streams contained in this source.
    media_streams: list[JellyfinMediaStream] = None


# JellyfinItem stores the Jellyfin item fields needed by browsing and lookup.
@dataclass
class JellyfinItem:
    
    # Human-readable Jellyfin item name.
    name: str = ""

    # Stable Jellyfin item ID.
    item_id: str = ""

    # Jellyfin item type, such as Folder, CollectionFolder, or Movie.
    item_type: str = ""

    # Server-side path when Jellyfin includes it.
    path: str = ""

    # Whether this item is a folder-like container.
    is_folder: Optional[bool] = None

    # Jellyfin media type, such as Video.
    media_type: str = ""

    # Runtime in Jellyfin ticks when available.
    runtime_ticks: Optional[int] = None

    # Child count for folder-like items.
    child_count: Optional[int] = None

    # Media sources included with the item, when requested.
    media_sources: list[JellyfinMediaSource] = None


# JellyfinPlaybackInfo stores the useful subset of PlaybackInfo.
@dataclass
class JellyfinPlaybackInfo:
    
    # Jellyfin play session ID returned by PlaybackInfo.
    play_session_id: str = ""

    # Jellyfin error code if PlaybackInfo reports one.
    error_code: str = ""

    # Media sources returned by PlaybackInfo.
    media_sources: list[JellyfinMediaSource] = None


# JellyfinPlaybackOption represents an app-level requested stream target.
@dataclass
class JellyfinPlaybackOption:
    
    # Human-readable option name.
    display_name: str

    # Option mode: Original, Auto, or Transcode.
    mode: str

    # Maximum stream bitrate requested from Jellyfin.
    max_streaming_bitrate: Optional[int] = None

    # Maximum video width requested from Jellyfin.
    max_width: Optional[int] = None

    # Maximum video height requested from Jellyfin.
    max_height: Optional[int] = None


# JellyfinClient owns Jellyfin session state and low-level API calls.
class JellyfinClient:
    
    # Creates a client using explicit values or environment variables.
    # Inputs are optional server URL and credentials.
    # Output is an unauthenticated client until authenticate is called.
    # Use this as the reusable worker-side Jellyfin API wrapper.
    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ):
        
        # Base URL is normalized before endpoint URLs are built.
        self.base_url = self.normalize_base_url(base_url or os.getenv("JELLYFIN_BASE_URL", ""))

        # Username may be provided directly or through the environment.
        self.username = username or os.getenv("JELLYFIN_USERNAME", "")

        # Password may be provided directly or through the environment.
        self.password = password or os.getenv("JELLYFIN_PASSWORD", "")

        # Access token is populated after authentication.
        self.access_token = ""

        # User ID is populated after authentication.
        self.user_id = ""

        # A persistent HTTP client avoids rebuilding connection state for each request.
        self._client = httpx.Client(timeout=timeout_seconds, follow_redirects=True)

    # Reports whether the client has enough session state for authenticated calls.
    # Inputs are none.
    # Output is True when base URL, token, and user ID are present.
    # Use this before calling authenticated Jellyfin methods.
    @property
    def is_authenticated(self) -> bool:
        
        # All three values are required for authenticated Jellyfin requests.
        return bool(self.base_url and self.access_token and self.user_id)

    # Normalizes a Jellyfin server URL for safe endpoint construction.
    # Input is a server URL string.
    # Output is the same URL without trailing slashes.
    # Use this before appending endpoint paths.
    @staticmethod
    def normalize_base_url(base_url: str) -> str:
        
        # Blank values remain blank so the caller can report a clear config error.
        if not base_url:
            return ""

        # Trailing slashes are removed so endpoint strings are predictable.
        return base_url.strip().rstrip("/")

    # Authenticates against Jellyfin using username and password.
    # Inputs may override constructor credentials.
    # Output is the raw Jellyfin authentication response dictionary.
    # Use this before browsing, searching, or building stream URLs.
    def authenticate(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> dict[str, Any]:
        
        # Optional arguments allow one-off authentication without rebuilding the client.
        if base_url is not None:
            self.base_url = self.normalize_base_url(base_url)

        # Optional username overrides the stored username.
        if username is not None:
            self.username = username

        # Optional password overrides the stored password.
        if password is not None:
            self.password = password

        # Missing connection values should fail before making a malformed request.
        if not self.base_url:
            raise ValueError("Jellyfin base URL is required.")

        # Missing username should fail before asking Jellyfin.
        if not self.username:
            raise ValueError("Jellyfin username is required.")

        # Missing password should fail before asking Jellyfin.
        if not self.password:
            raise ValueError("Jellyfin password is required.")

        # Authentication endpoint matches the existing C# client.
        url = f"{self.base_url}/Users/AuthenticateByName"

        # Jellyfin expects username and password in this JSON shape.
        body = {
            "Username": self.username,
            "Pw": self.password,
        }

        # Jellyfin uses this authorization header to identify the client application.
        headers = {
            "Authorization": self._build_media_browser_authorization_header(),
        }

        # Send the authentication request and let the shared helper validate status.
        response = self._client.post(url, json=body, headers=headers)
        self._raise_for_bad_response(response, "Jellyfin authentication failed")

        # Parse the authentication response.
        data = response.json()

        # Store the token returned by Jellyfin.
        self.access_token = str(data.get("AccessToken") or "")

        # Store the user ID returned by Jellyfin.
        self.user_id = str((data.get("User") or {}).get("Id") or "")

        # Authentication is not usable without both token and user ID.
        if not self.access_token or not self.user_id:
            raise RuntimeError("Jellyfin did not return a usable access token and user ID.")

        return data

    # Loads the top-level libraries available to the authenticated user.
    # Inputs are none.
    # Output is a list of JellyfinItem objects.
    # Use this for future browser roots.
    def get_libraries(self) -> list[JellyfinItem]:
        
        # Authenticated session state is required for user library browsing.
        self._require_authenticated_session()

        # Jellyfin user views endpoint returns top-level libraries.
        url = f"{self.base_url}/Users/{quote(self.user_id)}/Views"

        # Send the authenticated GET and parse the items.
        data = self._get_json(url)

        return self._parse_items_response(data)

    # Loads child items under a Jellyfin parent item.
    # Input is a Jellyfin parent item ID.
    # Output is a list of child JellyfinItem objects.
    # Use this for future folder browsing.
    def get_child_items(self, parent_item_id: str) -> list[JellyfinItem]:
        
        # Authenticated session state is required for item browsing.
        self._require_authenticated_session()

        # Parent item ID is required because this method is non-recursive.
        if not parent_item_id:
            raise ValueError("Parent Jellyfin item ID is required.")

        # Fields match the useful subset from the C# browser client.
        fields = (
            "Path,Overview,MediaSources,MediaStreams,PrimaryImageAspectRatio,"
            "DateCreated,ChildCount,RunTimeTicks,ImageTags,BackdropImageTags"
        )

        # Jellyfin child item endpoint browses one parent folder at a time.
        url = (
            f"{self.base_url}/Users/{quote(self.user_id)}/Items"
            f"?ParentId={quote(parent_item_id)}"
            f"&Recursive=false"
            f"&Fields={quote(fields, safe=',')}"
        )

        # Send the authenticated GET and parse the items.
        data = self._get_json(url)

        return self._parse_items_response(data)

    # Searches Jellyfin video items by text.
    # Input is a filename or title search term.
    # Output is a list of matching Jellyfin video items.
    # Use this to resolve database video_source values.
    def search_video_items(self, query: str, limit: int = 20) -> list[JellyfinItem]:
        
        # Authenticated session state is required for item searching.
        self._require_authenticated_session()

        # Blank searches should not accidentally return large media lists.
        if not query:
            raise ValueError("Jellyfin search query is required.")

        # Fields include path and media source data needed for matching and playback.
        fields = "Path,MediaSources,MediaStreams,RunTimeTicks"

        # Jellyfin item search is recursive and constrained to video items.
        url = (
            f"{self.base_url}/Users/{quote(self.user_id)}/Items"
            f"?Recursive=true"
            f"&IncludeItemTypes=Video"
            f"&SearchTerm={quote(query)}"
            f"&Limit={limit}"
            f"&Fields={quote(fields, safe=',')}"
        )

        # Send the authenticated GET and parse the items.
        data = self._get_json(url)

        return self._parse_items_response(data)

    # Gets PlaybackInfo for one Jellyfin media item.
    # Inputs are item ID and max streaming bitrate.
    # Output is parsed JellyfinPlaybackInfo.
    # Use this before deciding direct stream or transcode behavior.
    def get_playback_info(
        self,
        item_id: str,
        max_streaming_bitrate: int = 8_000_000,
    ) -> JellyfinPlaybackInfo:
        
        # Authenticated session state is required for PlaybackInfo.
        self._require_authenticated_session()

        # Jellyfin item ID is required for PlaybackInfo.
        if not item_id:
            raise ValueError("Jellyfin item ID is required.")

        # PlaybackInfo endpoint matches the existing C# client behavior.
        url = (
            f"{self.base_url}/Items/{quote(item_id)}/PlaybackInfo"
            f"?UserId={quote(self.user_id)}"
        )

        # Request all playback paths so Jellyfin reports source capabilities.
        body = {
            "UserId": self.user_id,
            "MaxStreamingBitrate": max_streaming_bitrate,
            "EnableDirectPlay": True,
            "EnableDirectStream": True,
            "EnableTranscoding": True,
        }

        # Send the authenticated POST and parse the playback info.
        data = self._post_json(url, body)

        return self._parse_playback_info(data)

    # Requests transcode PlaybackInfo for one constrained target.
    # Inputs are item ID and a playback option.
    # Output is parsed JellyfinPlaybackInfo, usually with TranscodingUrl.
    # Use this instead of manually assembling fragile transcode URLs.
    def get_transcode_playback_info(
        self,
        item_id: str,
        option: JellyfinPlaybackOption,
    ) -> JellyfinPlaybackInfo:
        
        # Authenticated session state is required for PlaybackInfo.
        self._require_authenticated_session()

        # Jellyfin item ID is required for transcode PlaybackInfo.
        if not item_id:
            raise ValueError("Jellyfin item ID is required.")

        # Playback option is required so a constrained target can be requested.
        if option is None:
            raise ValueError("A Jellyfin playback option is required.")

        # Bitrate fallback mirrors the C# client default.
        max_streaming_bitrate = option.max_streaming_bitrate or 4_000_000

        # Width fallback mirrors the C# client default.
        max_width = option.max_width or 1280

        # Height fallback mirrors the C# client default.
        max_height = option.max_height or 720

        # PlaybackInfo endpoint returns session-associated playback URLs.
        url = (
            f"{self.base_url}/Items/{quote(item_id)}/PlaybackInfo"
            f"?UserId={quote(self.user_id)}"
        )

        # Device profile asks Jellyfin for an HLS H.264/AAC transcode target.
        body = {
            "UserId": self.user_id,
            "EnableDirectPlay": False,
            "EnableDirectStream": False,
            "EnableTranscoding": True,
            "MaxStreamingBitrate": max_streaming_bitrate,
            "MaxWidth": max_width,
            "MaxHeight": max_height,
            "AllowVideoStreamCopy": False,
            "AllowAudioStreamCopy": True,
            "DeviceProfile": {
                "Name": "MARP Inference Worker",
                "MaxStreamingBitrate": max_streaming_bitrate,
                "TranscodingProfiles": [
                    {
                        "Container": "ts",
                        "Type": "Video",
                        "VideoCodec": "h264",
                        "AudioCodec": "aac",
                        "Protocol": "hls",
                    }
                ],
            },
        }

        # Send the authenticated POST and parse the playback info.
        data = self._post_json(url, body)

        return self._parse_playback_info(data)

    # Builds the original direct Jellyfin stream URL for an item.
    # Input is a Jellyfin item ID.
    # Output is an absolute Jellyfin stream URL.
    # Use this as the first OpenCV stream test target.
    def build_direct_stream_url(self, item_id: str) -> str:
        
        # Authenticated session state is required because the URL includes the token.
        self._require_authenticated_session()

        # Blank item IDs cannot produce a valid stream URL.
        if not item_id:
            return ""

        # Direct stream URL matches the existing C# client behavior.
        return (
            f"{self.base_url}/Videos/{quote(item_id)}/stream"
            f"?static=true&api_key={quote(self.access_token)}"
        )

    # Converts Jellyfin relative URLs into absolute URLs.
    # Input is a relative or absolute Jellyfin URL.
    # Output is an absolute URL.
    # Use this for PlaybackInfo DirectStreamUrl or TranscodingUrl values.
    def build_absolute_jellyfin_url(self, jellyfin_url: str) -> str:
        
        # Authenticated session state is required because base URL must be known.
        self._require_authenticated_session()

        # Blank input remains blank.
        if not jellyfin_url:
            return ""

        # Absolute URLs can be returned unchanged.
        if jellyfin_url.lower().startswith(("http://", "https://")):
            return jellyfin_url

        # Root-relative URLs are appended directly to the base server URL.
        if jellyfin_url.startswith("/"):
            return f"{self.base_url}{jellyfin_url}"

        # Path-relative URLs are appended with a slash.
        return f"{self.base_url}/{jellyfin_url}"

    # Closes the underlying HTTP client.
    # Inputs are none.
    # Output is none.
    # Use this when a script is done with the client.
    def close(self) -> None:
        
        # Close releases HTTP connection resources.
        self._client.close()

    # Validates that authenticated session state exists.
    # Inputs are none.
    # Output is none or an exception.
    # Use this at the top of authenticated methods.
    def _require_authenticated_session(self) -> None:
        
        # Base URL, access token, and user ID are all required.
        if not self.is_authenticated:
            raise RuntimeError("Jellyfin API client is not authenticated.")

    # Builds the Jellyfin MediaBrowser authorization header.
    # Inputs are none.
    # Output is the header value.
    # Use this during username/password authentication.
    def _build_media_browser_authorization_header(self) -> str:
        
        # Client name identifies this worker in Jellyfin sessions.
        client_name = "MARP Inference Worker"

        # Device name is stable enough for development and server logs.
        device_name = os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME") or "marp-worker"

        # Device ID keeps Jellyfin from treating every process as a different app.
        device_id = f"marp-inference-worker-{device_name}".lower()

        # App version mirrors the current project version.
        version = "0.1.0"

        return (
            'MediaBrowser '
            f'Client="{self._escape_header_value(client_name)}", '
            f'Device="{self._escape_header_value(device_name)}", '
            f'DeviceId="{self._escape_header_value(device_id)}", '
            f'Version="{self._escape_header_value(version)}"'
        )

    # Escapes quoted MediaBrowser header values.
    # Input is a header field value.
    # Output is the safely escaped value.
    # Use this before inserting values into the authorization header.
    @staticmethod
    def _escape_header_value(value: str) -> str:
        
        # None should become an empty quoted field value.
        if value is None:
            return ""

        # Backslashes are escaped before quotes to avoid ambiguous header values.
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    # Sends an authenticated Jellyfin GET request and parses JSON.
    # Input is a full URL.
    # Output is the decoded JSON dictionary.
    # Use this for typed Jellyfin GET helpers.
    def _get_json(self, url: str) -> dict[str, Any]:
        
        # Jellyfin accepts X-Emby-Token on authenticated API requests.
        headers = {
            "X-Emby-Token": self.access_token,
        }

        # Send the GET request.
        response = self._client.get(url, headers=headers)
        self._raise_for_bad_response(response, "Jellyfin GET failed")

        return response.json()

    # Sends an authenticated Jellyfin POST request and parses JSON.
    # Inputs are a full URL and JSON-compatible body.
    # Output is the decoded JSON dictionary.
    # Use this for PlaybackInfo requests.
    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        
        # Jellyfin accepts X-Emby-Token on authenticated API requests.
        headers = {
            "X-Emby-Token": self.access_token,
        }

        # Send the POST request.
        response = self._client.post(url, json=body, headers=headers)
        self._raise_for_bad_response(response, "Jellyfin POST failed")

        return response.json()

    # Raises a useful exception for failed Jellyfin HTTP responses.
    # Inputs are an httpx response and context string.
    # Output is none or an exception.
    # Use this to keep API failure messages consistent.
    @staticmethod
    def _raise_for_bad_response(response: httpx.Response, context: str) -> None:
        
        # Successful responses need no action.
        if response.is_success:
            return

        # Response text is included because Jellyfin often explains auth/API failures.
        raise RuntimeError(
            f"{context}. HTTP {response.status_code}: {response.text}"
        )

    # Parses a Jellyfin item-list response.
    # Input is a decoded Jellyfin response dictionary.
    # Output is a list of JellyfinItem objects.
    # Use this after libraries, child items, or search calls.
    def _parse_items_response(self, data: dict[str, Any]) -> list[JellyfinItem]:
        
        # Jellyfin returns item arrays under Items.
        raw_items = data.get("Items") or []

        return [self._parse_item(raw_item) for raw_item in raw_items]

    # Parses one Jellyfin item DTO.
    # Input is a decoded Jellyfin item dictionary.
    # Output is a JellyfinItem object.
    # Use this to normalize Jellyfin item fields.
    def _parse_item(self, raw_item: dict[str, Any]) -> JellyfinItem:
        
        # Raw media sources may be included when Fields requests them.
        raw_media_sources = raw_item.get("MediaSources") or []

        # Parse media sources into typed lightweight objects.
        media_sources = [
            self._parse_media_source(raw_source)
            for raw_source in raw_media_sources
        ]

        return JellyfinItem(
            name=str(raw_item.get("Name") or ""),
            item_id=str(raw_item.get("Id") or ""),
            item_type=str(raw_item.get("Type") or ""),
            path=str(raw_item.get("Path") or ""),
            is_folder=raw_item.get("IsFolder"),
            media_type=str(raw_item.get("MediaType") or ""),
            runtime_ticks=raw_item.get("RunTimeTicks"),
            child_count=raw_item.get("ChildCount"),
            media_sources=media_sources,
        )

    # Parses a Jellyfin media source DTO.
    # Input is a decoded Jellyfin media source dictionary.
    # Output is a JellyfinMediaSource object.
    # Use this after item search or PlaybackInfo responses.
    def _parse_media_source(self, raw_source: dict[str, Any]) -> JellyfinMediaSource:
        
        # Raw media streams may include video, audio, and subtitle streams.
        raw_media_streams = raw_source.get("MediaStreams") or []

        # Parse media stream records into typed lightweight objects.
        media_streams = [
            self._parse_media_stream(raw_stream)
            for raw_stream in raw_media_streams
        ]

        return JellyfinMediaSource(
            source_id=str(raw_source.get("Id") or ""),
            path=str(raw_source.get("Path") or ""),
            container=str(raw_source.get("Container") or ""),
            name=str(raw_source.get("Name") or ""),
            protocol=str(raw_source.get("Protocol") or ""),
            size=raw_source.get("Size"),
            bitrate=raw_source.get("Bitrate"),
            runtime_ticks=raw_source.get("RunTimeTicks"),
            supports_direct_play=raw_source.get("SupportsDirectPlay"),
            supports_direct_stream=raw_source.get("SupportsDirectStream"),
            supports_transcoding=raw_source.get("SupportsTranscoding"),
            direct_stream_url=str(raw_source.get("DirectStreamUrl") or ""),
            transcoding_url=str(raw_source.get("TranscodingUrl") or ""),
            transcoding_sub_protocol=str(raw_source.get("TranscodingSubProtocol") or ""),
            transcoding_container=str(raw_source.get("TranscodingContainer") or ""),
            media_streams=media_streams,
        )

    # Parses a Jellyfin media stream DTO.
    # Input is a decoded Jellyfin media stream dictionary.
    # Output is a JellyfinMediaStream object.
    # Use this to normalize video stream metadata.
    def _parse_media_stream(self, raw_stream: dict[str, Any]) -> JellyfinMediaStream:
        
        return JellyfinMediaStream(
            stream_type=str(raw_stream.get("Type") or ""),
            codec=str(raw_stream.get("Codec") or ""),
            width=raw_stream.get("Width"),
            height=raw_stream.get("Height"),
            bit_rate=raw_stream.get("BitRate"),
            average_frame_rate=raw_stream.get("AverageFrameRate"),
            real_frame_rate=raw_stream.get("RealFrameRate"),
        )

    # Parses a Jellyfin PlaybackInfo response.
    # Input is a decoded PlaybackInfo dictionary.
    # Output is a JellyfinPlaybackInfo object.
    # Use this before selecting direct or transcode stream URLs.
    def _parse_playback_info(self, data: dict[str, Any]) -> JellyfinPlaybackInfo:
        
        # PlaybackInfo returns source records under MediaSources.
        raw_media_sources = data.get("MediaSources") or []

        # Parse media source records.
        media_sources = [
            self._parse_media_source(raw_source)
            for raw_source in raw_media_sources
        ]

        return JellyfinPlaybackInfo(
            play_session_id=str(data.get("PlaySessionId") or ""),
            error_code=str(data.get("ErrorCode") or ""),
            media_sources=media_sources,
        )