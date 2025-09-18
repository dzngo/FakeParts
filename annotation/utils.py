"""Utility helpers for the annotation Streamlit application.

This module centralises all non-UI functionality shared by the app, including
Google API client initialisation, Drive/Sheets helpers, sampling logic and
local fallbacks. Keeping the implementation here keeps ``app.py`` focused on
Streamlit presentation while allowing the utilities to be imported elsewhere
(e.g. for tests or batch scripts).
"""

from __future__ import annotations

import atexit
import json
import os
import random
import shutil
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple, Union

import gspread
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from oauth2client.service_account import ServiceAccountCredentials

# ---------------------------------------------------------------------------
# Constants and configuration bootstrap
# ---------------------------------------------------------------------------
GOOGLE_SCOPES: Sequence[str] = (
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
)
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mkv")
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}
DEFAULT_SAMPLE_SIZE: int = 10
SHEET_HEADER: Sequence[str] = (
    "User ID",
    "ID",
    "Video Path",
    "Name",
    "Ground Truth",
    "User Answer",
    "Comment",
    "Timestamp",
)
DRIVE_CACHE_DIR = Path(tempfile.gettempdir()) / "fakeparts_drive_cache"
DRIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_drive_cache() -> None:
    """Remove cached Drive files on interpreter exit."""

    shutil.rmtree(DRIVE_CACHE_DIR, ignore_errors=True)


atexit.register(_cleanup_drive_cache)


def _get_secret(key: str, default=None):
    """Return a secret value if available, otherwise *default*."""

    try:
        return st.secrets[key]
    except (KeyError, AttributeError, RuntimeError):
        return default


def _safe_int(value, fallback: int) -> int:
    """Return ``value`` as ``int`` or ``fallback`` when conversion fails."""

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
    """Resolve Google Sheet URL, Drive IDs, and sample size for the session."""

    sheet_url = _resolve_setting("GOOGLE_SHEET_URL", "") or ""
    fake_id = _resolve_setting("DRIVE_FAKE_FOLDER_ID", "") or ""
    real_id = _resolve_setting("DRIVE_REAL_FOLDER_ID", "") or ""
    sample_raw = _resolve_setting("DEFAULT_SAMPLE_SIZE", DEFAULT_SAMPLE_SIZE)
    sample_size = _safe_int(sample_raw, DEFAULT_SAMPLE_SIZE)
    return sheet_url, fake_id, real_id, sample_size


GOOGLE_SHEET_URL, DRIVE_FAKE_FOLDER_ID, DRIVE_REAL_FOLDER_ID, DEFAULT_SAMPLE_SIZE = _load_run_settings()


# ---------------------------------------------------------------------------
# Data descriptors
# ---------------------------------------------------------------------------
@dataclass
class DriveFolder:
    """Descriptor for a Google Drive folder containing videos."""

    id: str
    label: str


@dataclass
class DriveVideo:
    """Representation of a video file stored in Google Drive."""

    id: str
    name: str
    label: str
    mime_type: Optional[str] = None

    @property
    def parent(self):
        return SimpleNamespace(name=self.label)

    @property
    def stem(self):
        return Path(self.name).stem

    def __str__(self) -> str:  # pragma: no cover - convenience for logging/UI
        return f"https://drive.google.com/uc?export=download&id={self.id}"


# ---------------------------------------------------------------------------
# Cached Google API clients
# ---------------------------------------------------------------------------
_CREDENTIALS = None
_DRIVE_SERVICE = None
_SHEET_CLIENT = None
_SHEET_HEADER_INITIALIZED = False
_SHEET_WORKSHEET = None


def get_credentials():
    """Return shared service-account credentials for Google APIs."""

    global _CREDENTIALS
    if _CREDENTIALS is None:
        creds_source = os.getenv("GCP_CREDENTIALS") or _get_secret("GCP_CREDENTIALS")
        if not creds_source:
            st.error("Google credentials missing. Set GCP_CREDENTIALS as an env var or Streamlit secret.")
            st.stop()

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
# Google Sheets helpers
# ---------------------------------------------------------------------------


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
            if not _SHEET_WORKSHEET.get_all_values():
                _SHEET_WORKSHEET.append_row(SHEET_HEADER, value_input_option="USER_ENTERED")
            _SHEET_HEADER_INITIALIZED = True

        _SHEET_WORKSHEET.append_row(list(data_row), value_input_option="USER_ENTERED")
    except Exception as err:  # pylint: disable=broad-except
        st.error(f"Unable to append row to Google Sheet: {err}")


# ---------------------------------------------------------------------------
# Video loading utilities
# ---------------------------------------------------------------------------


def list_drive_entries(folder: DriveFolder, exts: Sequence[str] = VIDEO_EXTENSIONS) -> List[DriveVideo]:
    """Return metadata entries for Drive videos without downloading the media."""

    service = get_drive_service()
    videos: List[DriveVideo] = []
    page_token = None

    try:
        while True:
            response = (
                service.files()  # pylint: disable=no-member
                .list(
                    q=f"'{folder.id}' in parents and trashed=false",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                    pageSize=100,
                )
                .execute()
            )
            for item in response.get("files", []):
                name = item.get("name")
                if not name or not name.lower().endswith(tuple(exts)):
                    continue
                videos.append(
                    DriveVideo(
                        id=item["id"],
                        name=name,
                        label=folder.label,
                        mime_type=item.get("mimeType"),
                    )
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                break
    except HttpError as err:
        st.error(f"Error loading Google Drive folder '{folder.label}': {err}")
        return []

    return sorted(videos, key=lambda vid: vid.name.lower())


def list_video_entries(folder: Union[DriveFolder, Path, str], exts: Sequence[str] = VIDEO_EXTENSIONS):
    """Return iterable of video references for Drive folders or local directories."""

    if isinstance(folder, DriveFolder):
        if not folder.id:
            st.error(f"No Google Drive folder ID configured for '{folder.label}'.")
            return []
        return list_drive_entries(folder, exts)

    folder_path = Path(folder)
    try:
        if not folder_path.is_dir():
            st.error(f"Directory '{folder_path}' does not exist.")
            return []
        return sorted(
            [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in tuple(exts)],
            key=lambda path: path.name.lower(),
        )
    except Exception as err:  # pylint: disable=broad-except
        st.error(f"Error reading {folder_path}: {err}")
        return []


def _folder_name(folder: Union[DriveFolder, Path, str]) -> str:
    if isinstance(folder, DriveFolder):
        return folder.label
    return Path(folder).name


def is_drive_video(obj) -> bool:
    """Return ``True`` when ``obj`` behaves like :class:`DriveVideo`.

    Streamlit reruns re-create class objects, so instances stored in
    ``st.session_state`` may not pass ``isinstance`` against the re-imported
    class. Instead of relying on identity we look for the attributes the app
    uses (``id`` and ``name``), allowing the check to stay resilient across
    reruns.
    """

    return hasattr(obj, "id") and hasattr(obj, "name")


def ensure_local_drive_copy(video: DriveVideo) -> Optional[Path]:
    """Create a cached local copy of ``video`` and return its filesystem path."""

    suffix = Path(video.name).suffix or ".mp4"
    target = DRIVE_CACHE_DIR / f"{video.id}{suffix}"
    if target.exists():
        return target

    service = get_drive_service()
    request = service.files().get_media(fileId=video.id)  # pylint: disable=no-member

    try:
        with open(target, "wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
    except HttpError as err:
        st.error(f"Error downloading video (ID: {video.id}): {err}")
        if target.exists():
            target.unlink(missing_ok=True)
        return None
    except Exception as err:  # pylint: disable=broad-except
        st.error(f"Unexpected error downloading video '{video.name}': {err}")
        if target.exists():
            target.unlink(missing_ok=True)
        return None

    return target


def video_mime(video: DriveVideo) -> Optional[str]:
    """Return the MIME type for a :class:`DriveVideo` instance."""

    if video.mime_type:
        return video.mime_type
    return VIDEO_MIME_TYPES.get(Path(video.name).suffix.lower())


def sample_and_mix(fake_folder, real_folder, n_each: int = DEFAULT_SAMPLE_SIZE):
    """Sample videos from ``fake_folder`` and ``real_folder`` then shuffle them."""

    fake = list_video_entries(fake_folder)
    real = list_video_entries(real_folder)

    if not fake:
        st.error(f"No videos found in '{_folder_name(fake_folder)}'.")
    if not real:
        st.error(f"No videos found in '{_folder_name(real_folder)}'.")
    if not fake or not real:
        return []

    pick_f = random.sample(fake, min(n_each, len(fake)))
    pick_r = random.sample(real, min(n_each, len(real)))
    paths = pick_f + pick_r
    random.shuffle(paths)
    return paths


def resolve_video_sources():
    """Return Drive or local sources for fake and real videos.

    When ``DRIVE_FAKE_FOLDER_ID``/``DRIVE_REAL_FOLDER_ID`` are configured, the
    function returns :class:`DriveFolder` descriptors so the rest of the app can
    fetch videos directly from Google Drive. If the IDs are missing it falls
    back to the local ``random/Fake`` and ``random/Real`` directories next to
    ``app.py``. This keeps the sampling logic agnostic to the underlying storage
    location.
    """

    if DRIVE_FAKE_FOLDER_ID and DRIVE_REAL_FOLDER_ID:
        return (
            DriveFolder(DRIVE_FAKE_FOLDER_ID, "Fake"),
            DriveFolder(DRIVE_REAL_FOLDER_ID, "Real"),
        )

    base_dir = Path(__file__).parent / "random"
    fake_dir = base_dir / "Fake"
    real_dir = base_dir / "Real"
    if fake_dir.is_dir() and real_dir.is_dir():
        return fake_dir, real_dir

    st.error(
        "Configure DRIVE_FAKE_FOLDER_ID and DRIVE_REAL_FOLDER_ID via environment variables or Streamlit secrets, "
        "or provide a local 'random' directory."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def save_annotation(ann: dict, video_path, base: str = "annotations") -> None:
    """Persist a single annotation JSON file next to the source label."""

    sub = video_path.parent.name
    outdir = Path(base) / sub
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / (video_path.stem + ".json")
    try:
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(ann, f, indent=4)
    except Exception as err:  # pylint: disable=broad-except
        st.warning(f"Could not save annotation for {outfile.name}: {err}")


def generate_user_id(length: int = 6) -> str:
    """Return a random alphanumeric identifier for a user session."""

    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


__all__ = [
    "DriveFolder",
    "DriveVideo",
    "DEFAULT_SAMPLE_SIZE",
    "VIDEO_EXTENSIONS",
    "generate_user_id",
    "send_to_google_sheet",
    "list_drive_entries",
    "list_video_entries",
    "sample_and_mix",
    "resolve_video_sources",
    "save_annotation",
    "is_drive_video",
    "ensure_local_drive_copy",
    "video_mime",
    "get_credentials",
    "get_drive_service",
    "get_sheet_client",
]
