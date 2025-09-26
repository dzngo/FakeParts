"""Utility helpers for the annotation Streamlit application.

The module centralises non-UI functionality shared by the Streamlit app and
supporting scripts:

* Google API client initialisation (Sheets, Drive)
* Audio stripping with ffmpeg for Drive-hosted videos
* Downloading and sampling the video catalogue CSV
* Persisting annotations and writing them to Google Sheets
"""

from __future__ import annotations

import atexit
import json
import os
import random
import shutil
import string
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import gspread
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from oauth2client.service_account import ServiceAccountCredentials
from gspread.exceptions import WorksheetNotFound

# ---------------------------------------------------------------------------
# Constants and configuration bootstrap
# ---------------------------------------------------------------------------
GOOGLE_SCOPES: Sequence[str] = (
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
)
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov", ".webm")
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}
DEFAULT_SAMPLE_SIZE: int = 10  # Total videos sampled per session (split evenly fake/real)
SHEET_HEADER: Sequence[str] = (
    "User ID",
    "ID",
    "Video Path",
    "Name",
    "Ground Truth",
    "Method",
    "User Answer",
    "Comment",
    "Timestamp",
)
DRIVE_CACHE_DIR = Path(tempfile.gettempdir()) / "fakeparts_drive_cache"
DRIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
NO_AUDIO_SUFFIX = ".noaudio"


def _cleanup_drive_cache() -> None:
    """Remove cached Drive files on interpreter exit."""

    shutil.rmtree(DRIVE_CACHE_DIR, ignore_errors=True)


atexit.register(_cleanup_drive_cache)


def _get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except (KeyError, AttributeError, RuntimeError):
        return default


def _safe_int(value, fallback: int) -> int:
    try:
        if value is None:
            raise TypeError
        return int(str(value))
    except (TypeError, ValueError):
        return fallback


def _resolve_setting(key: str, default=""):
    """Return the configuration ``key`` preferring env vars then secrets."""

    env_value = os.getenv(key)
    if env_value not in (None, ""):
        return env_value
    return _get_secret(key, default)


def _load_run_settings() -> Tuple[str, str, str, int]:
    sheet_url = _resolve_setting("GOOGLE_SHEET_URL", "") or ""
    catalog_sheet_url = _resolve_setting("VIDEO_CATALOG_SHEET_URL", "") or ""
    catalog_worksheet = _resolve_setting("VIDEO_CATALOG_WORKSHEET", "") or ""
    sample_raw = _resolve_setting("DEFAULT_SAMPLE_SIZE", DEFAULT_SAMPLE_SIZE)
    sample_size = _safe_int(sample_raw, DEFAULT_SAMPLE_SIZE)
    return sheet_url, catalog_sheet_url, catalog_worksheet, sample_size


GOOGLE_SHEET_URL, VIDEO_CATALOG_SHEET_URL, VIDEO_CATALOG_WORKSHEET, DEFAULT_SAMPLE_SIZE = _load_run_settings()


# ---------------------------------------------------------------------------
# Data descriptors
# ---------------------------------------------------------------------------
@dataclass
class DriveVideo:
    """Representation of a video file stored in Google Drive."""

    id: str
    name: str
    label: str  # Fake or Real
    method: str  # Generation method / folder name for fake videos
    mime_type: Optional[str] = None

    @property
    def stem(self):
        return Path(self.name).stem

    def __str__(self) -> str:
        return f"https://drive.google.com/uc?export=download&id={self.id}"


# ---------------------------------------------------------------------------
# Cached Google API clients
# ---------------------------------------------------------------------------
_CREDENTIALS = None
_DRIVE_SERVICE = None
_SHEET_CLIENT = None
_SHEET_HEADER_INITIALIZED = False
_SHEET_WORKSHEET = None
_FFMPEG_BIN = None
_FFMPEG_REENCODE_WARNED = False


def get_credentials():
    """Return shared service-account credentials for Google APIs."""

    global _CREDENTIALS
    if _CREDENTIALS is None:
        creds_source = os.getenv("GCP_CREDENTIALS") or _get_secret("GCP_CREDENTIALS")
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if creds_source:
            if isinstance(creds_source, dict):
                creds_dict = creds_source
            else:
                try:
                    creds_dict = json.loads(str(creds_source))
                except json.JSONDecodeError as err:
                    st.error(f"Invalid JSON in GCP_CREDENTIALS: {err}")
                    st.stop()

            try:
                _CREDENTIALS = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, GOOGLE_SCOPES)
            except Exception as err:  # pylint: disable=broad-except
                st.error(f"Unable to load Google credentials: {err}")
                st.stop()
        elif creds_path:
            path_obj = Path(creds_path)
            if not path_obj.is_file():
                st.error(f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {path_obj}")
                st.stop()
            try:
                _CREDENTIALS = ServiceAccountCredentials.from_json_keyfile_name(str(path_obj), GOOGLE_SCOPES)
            except Exception as err:  # pylint: disable=broad-except
                st.error(f"Unable to load Google credentials from file: {err}")
                st.stop()
        else:
            st.error(
                "Google credentials missing. Provide GCP_CREDENTIALS (JSON) or set "
                "GOOGLE_APPLICATION_CREDENTIALS to the service-account file."
            )
            st.stop()
    return _CREDENTIALS


def get_sheet_client():
    """Return a cached gspread client."""

    global _SHEET_CLIENT
    if _SHEET_CLIENT is None:
        creds = get_credentials()
        try:
            _SHEET_CLIENT = gspread.authorize(creds)
        except Exception as err:  # pylint: disable=broad-except
            st.error(f"Unable to authorize Google Sheets client: {err}")
            st.stop()
    return _SHEET_CLIENT


def get_drive_service():
    """Return a cached Google Drive service client."""

    global _DRIVE_SERVICE
    if _DRIVE_SERVICE is None:
        creds = get_credentials()
        try:
            _DRIVE_SERVICE = build("drive", "v3", credentials=creds)
        except Exception as err:  # pylint: disable=broad-except
            st.error(f"Unable to initialize Google Drive client: {err}")
            st.stop()
    return _DRIVE_SERVICE


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def _get_ffmpeg() -> str:
    global _FFMPEG_BIN
    if _FFMPEG_BIN is None:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            st.error(
                "ffmpeg not found in PATH. Install ffmpeg locally and add it to `packages.txt` "
                "for Streamlit Cloud deployments to remove audio before playback."
            )
            st.stop()
        _FFMPEG_BIN = ffmpeg_path
    return _FFMPEG_BIN


def strip_audio(source: Path, destination: Path) -> bool:
    """Copy ``source`` to ``destination`` without audio using ffmpeg."""

    global _FFMPEG_REENCODE_WARNED
    ffmpeg = _get_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)

    tmp_handle = tempfile.NamedTemporaryFile(
        prefix=destination.stem + "_",
        suffix=destination.suffix,
        dir=destination.parent,
        delete=False,
    )
    temp_output = Path(tmp_handle.name)
    tmp_handle.close()

    def _run(cmd) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )

    common_flags = []
    if destination.suffix.lower() == ".mp4":
        common_flags.extend(["-movflags", "+faststart"])

    primary_cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-c:v",
        "copy",
        "-an",
        *common_flags,
        str(temp_output),
    ]

    try:
        _run(primary_cmd)
    except subprocess.CalledProcessError:
        if not _FFMPEG_REENCODE_WARNED:
            st.warning(
                "Direct stream copy failed while stripping audio. Falling back to video re-encoding; "
                "this may take longer."
            )
            _FFMPEG_REENCODE_WARNED = True

        fallback_cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-an",
            *common_flags,
            str(temp_output),
        ]
        try:
            _run(fallback_cmd)
        except subprocess.CalledProcessError as err:
            message = err.stderr.strip().splitlines()[-1] if err.stderr else str(err)
            st.error(f"ffmpeg failed to strip audio from '{source.name}': {message}")
            temp_output.unlink(missing_ok=True)
            return False

    try:
        temp_output.replace(destination)
    except OSError as err:
        st.error(f"Unable to finalize audio-free copy for '{source.name}': {err}")
        temp_output.unlink(missing_ok=True)
        return False
    return True


# ---------------------------------------------------------------------------
# Google Sheets helper
# ---------------------------------------------------------------------------


def _column_label(index: int) -> str:
    label = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        label = chr(65 + remainder) + label
    return label


def send_to_google_sheet(data_row: Sequence[str]) -> None:
    """Append ``data_row`` to the configured Google Sheet."""

    global _SHEET_WORKSHEET, _SHEET_HEADER_INITIALIZED

    if not GOOGLE_SHEET_URL:
        st.error("GOOGLE_SHEET_URL is not configured.")
        return

    client = get_sheet_client()
    try:
        if _SHEET_WORKSHEET is None:
            sheet = client.open_by_url(GOOGLE_SHEET_URL)
            _SHEET_WORKSHEET = sheet.sheet1
    except Exception as err:  # pylint: disable=broad-except
        st.error(f"Unable to open Google Sheet: {err}")
        return

    if _SHEET_WORKSHEET is None:
        st.error("Google Sheet worksheet not available.")
        return

    try:
        if not _SHEET_HEADER_INITIALIZED:
            first_row = _SHEET_WORKSHEET.row_values(1)
            if list(first_row[: len(SHEET_HEADER)]) != list(SHEET_HEADER):
                _SHEET_WORKSHEET.update("A1", [SHEET_HEADER])
            _SHEET_HEADER_INITIALIZED = True

        end_column = _column_label(len(SHEET_HEADER))
        _SHEET_WORKSHEET.append_rows(
            [list(data_row)],
            value_input_option="USER_ENTERED",
            table_range=f"A:{end_column}",
        )
    except Exception as err:  # pylint: disable=broad-except
        st.error(f"Unable to append row to Google Sheet: {err}")


# ---------------------------------------------------------------------------
# Video catalogue loading & sampling
# ---------------------------------------------------------------------------


def get_video_catalog() -> List[DriveVideo]:
    if not VIDEO_CATALOG_SHEET_URL:
        st.error("VIDEO_CATALOG_SHEET_URL is not configured.")
        st.stop()

    client = get_sheet_client()
    try:
        sheet = client.open_by_url(VIDEO_CATALOG_SHEET_URL)
    except Exception as err:  # pylint: disable=broad-except
        st.error(f"Unable to open video catalogue sheet: {err}")
        st.stop()

    if VIDEO_CATALOG_WORKSHEET:
        try:
            worksheet = sheet.worksheet(VIDEO_CATALOG_WORKSHEET)
        except WorksheetNotFound:
            st.error(f"Worksheet '{VIDEO_CATALOG_WORKSHEET}' not found in the video catalogue sheet.")
            st.stop()
    else:
        worksheet = sheet.sheet1

    try:
        records = worksheet.get_all_records()
    except Exception as err:  # pylint: disable=broad-except
        st.error(f"Unable to read video catalogue: {err}")
        st.stop()

    videos: List[DriveVideo] = []
    for row in records:
        category = (row.get("category") or row.get("label") or "").strip()
        method = (row.get("method") or row.get("folder") or "").strip()
        video_id = (row.get("drive_id") or row.get("id") or "").strip()
        name = (row.get("name") or row.get("file_name") or video_id).strip()
        mime = (row.get("mime_type") or "").strip() or None
        if not (category and method and video_id and name):
            continue
        videos.append(DriveVideo(id=video_id, name=name, label=category, method=method, mime_type=mime))

    return videos


def _sample_fake_videos(fake_videos: List[DriveVideo], quota: int) -> List[DriveVideo]:
    if quota <= 0 or not fake_videos:
        return []

    by_method: dict[str, List[DriveVideo]] = {}
    for video in fake_videos:
        by_method.setdefault(video.method, []).append(video)

    methods = list(by_method.keys())
    random.shuffle(methods)

    allocations: dict[str, int] = {method: 0 for method in methods}
    if quota <= len(methods):
        for method in methods[:quota]:
            allocations[method] = 1
    else:
        base = quota // len(methods)
        remainder = quota % len(methods)
        for method in methods:
            allocations[method] = min(base, len(by_method[method]))
        for method in methods:
            if remainder <= 0:
                break
            if allocations[method] < len(by_method[method]):
                allocations[method] += 1
                remainder -= 1

    selected: List[DriveVideo] = []
    leftovers: List[DriveVideo] = []
    for method in methods:
        bucket = by_method[method][:]
        random.shuffle(bucket)
        want = allocations[method]
        picked = bucket[:want]
        selected.extend(picked)
        leftovers.extend(bucket[want:])

    if len(selected) < quota:
        random.shuffle(leftovers)
        for video in leftovers:
            if len(selected) >= quota:
                break
            selected.append(video)

    random.shuffle(selected)
    return selected[:quota]


def sample_videos(total: int = DEFAULT_SAMPLE_SIZE) -> List[DriveVideo]:
    catalog = get_video_catalog()
    if not catalog:
        return []

    by_category: dict[str, List[DriveVideo]] = {"Fake": [], "Real": []}
    for video in catalog:
        by_category.setdefault(video.label, []).append(video)

    fake_videos = by_category.get("Fake", [])
    real_videos = by_category.get("Real", [])
    if not fake_videos or not real_videos:
        return []

    target_per_class = max(1, total // 2)
    max_possible = min(len(fake_videos), len(real_videos))
    target_per_class = min(target_per_class, max_possible)

    random.shuffle(real_videos)
    real_selection = real_videos[:target_per_class]

    fake_selection = _sample_fake_videos(fake_videos, target_per_class)
    if len(fake_selection) < target_per_class:
        remaining = [v for v in fake_videos if v not in fake_selection]
        random.shuffle(remaining)
        fake_selection.extend(remaining[: target_per_class - len(fake_selection)])
        fake_selection = fake_selection[:target_per_class]

    combined = fake_selection + real_selection
    random.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Playback helpers
# ---------------------------------------------------------------------------


def ensure_local_drive_copy(video: DriveVideo, remove_audio: bool = True) -> Optional[Path]:
    suffix = Path(video.name).suffix or ".mp4"
    original = DRIVE_CACHE_DIR / f"{video.id}{suffix}"

    if not original.exists():
        service = get_drive_service()
        request = service.files().get_media(fileId=video.id)  # pylint: disable=no-member

        try:
            with open(original, "wb") as handle:
                downloader = MediaIoBaseDownload(handle, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
        except HttpError as err:
            st.error(f"Error downloading video (ID: {video.id}): {err}")
            if original.exists():
                original.unlink(missing_ok=True)
            return None
        except Exception as err:  # pylint: disable=broad-except
            st.error(f"Unexpected error downloading video '{video.name}': {err}")
            if original.exists():
                original.unlink(missing_ok=True)
            return None

    if not remove_audio:
        return original

    muted = DRIVE_CACHE_DIR / f"{video.id}{NO_AUDIO_SUFFIX}{suffix}"
    needs_refresh = not muted.exists() or muted.stat().st_mtime < original.stat().st_mtime
    if needs_refresh:
        if not strip_audio(original, muted):
            return None
    return muted


def get_playback_path(video: DriveVideo, remove_audio: bool = True) -> Optional[Path]:
    return ensure_local_drive_copy(video, remove_audio=remove_audio)


def video_mime(video: DriveVideo) -> Optional[str]:
    if video.mime_type:
        return video.mime_type
    return VIDEO_MIME_TYPES.get(Path(video.name).suffix.lower())


def is_drive_video(obj) -> bool:
    return isinstance(obj, DriveVideo)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def save_annotation(ann: dict, video_path: DriveVideo, base: str = "annotations") -> None:
    sub = video_path.label
    outdir = Path(base) / sub
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / (video_path.stem + ".json")
    try:
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(ann, f, indent=4)
    except Exception as err:  # pylint: disable=broad-except
        st.warning(f"Could not save annotation for {outfile.name}: {err}")


def generate_user_id(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


__all__ = [
    "DriveVideo",
    "DEFAULT_SAMPLE_SIZE",
    "VIDEO_EXTENSIONS",
    "generate_user_id",
    "send_to_google_sheet",
    "get_video_catalog",
    "sample_videos",
    "get_playback_path",
    "ensure_local_drive_copy",
    "video_mime",
    "is_drive_video",
    "save_annotation",
    "strip_audio",
    "get_credentials",
    "get_drive_service",
    "get_sheet_client",
]
