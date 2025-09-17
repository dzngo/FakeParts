"""
Streamlit App for Human Evaluation of AI-Generated Videos
========================================================

Author
-------
Gaëtan Brison

Description
-------
This application presents users with a series of videos to classify as either
"AI-generated" or "Real". It stores the annotations locally and sends results to
a Google Sheet.

Modules
-------
- os, json, random, string: Standard Python libraries
- pathlib.Path: Path management
- datetime: Timestamps
- streamlit: UI rendering
- gspread, oauth2client: Google Sheets API access

Functions
---------
- send_to_google_sheet: Uploads annotations to Google Sheet.
- generate_user_id: Generates a unique user session ID.
- load_videos: Loads videos from a folder.
- sample_and_mix: Samples and shuffles real/fake videos.
- save_annotation: Saves JSON annotation to disk.

"""

import os
import atexit
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional
import tempfile
import shutil

import streamlit as st
import gspread
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.http import MediaIoBaseDownload


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
GOOGLE_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
DRIVE_CACHE_DIR = Path(tempfile.gettempdir()) / "fakeparts_drive_cache"
GOOGLE_SHEET_URL = ""
DRIVE_FAKE_FOLDER_ID = ""
DRIVE_REAL_FOLDER_ID = ""
DEFAULT_SAMPLE_SIZE = 10
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv")
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}
SHEET_HEADER = [
    "User ID",
    "ID",
    "Video Path",
    "Name",
    "Ground Truth",
    "User Answer",
    "Comment",
    "Timestamp",
]


def _get_secret(key, default=None):
    """Helper to safely read from streamlit secrets."""

    try:
        return st.secrets[key]
    except (KeyError, AttributeError, RuntimeError):
        return default


def _coerce_int(value, default):
    try:
        if value is None:
            raise TypeError
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _load_run_settings():
    global GOOGLE_SHEET_URL, DRIVE_FAKE_FOLDER_ID, DRIVE_REAL_FOLDER_ID, DEFAULT_SAMPLE_SIZE

    sheet_secret = _get_secret("GOOGLE_SHEET_URL", "")
    drive_fake_secret = _get_secret("DRIVE_FAKE_FOLDER_ID", "")
    drive_real_secret = _get_secret("DRIVE_REAL_FOLDER_ID", "")
    sample_secret = _get_secret("DEFAULT_SAMPLE_SIZE", DEFAULT_SAMPLE_SIZE)

    GOOGLE_SHEET_URL = os.getenv("GOOGLE_SHEET_URL") or sheet_secret or GOOGLE_SHEET_URL
    DRIVE_FAKE_FOLDER_ID = os.getenv("DRIVE_FAKE_FOLDER_ID") or drive_fake_secret or DRIVE_FAKE_FOLDER_ID
    DRIVE_REAL_FOLDER_ID = os.getenv("DRIVE_REAL_FOLDER_ID") or drive_real_secret or DRIVE_REAL_FOLDER_ID
    DEFAULT_SAMPLE_SIZE = _coerce_int(os.getenv("DEFAULT_SAMPLE_SIZE") or sample_secret, DEFAULT_SAMPLE_SIZE)


_load_run_settings()


# ---------------------------------------------------------------------------
# Cached Google API clients
# ---------------------------------------------------------------------------
_CREDENTIALS = None
_DRIVE_SERVICE = None
_SHEET_CLIENT = None
_SHEET_HEADER_INITIALIZED = False
_SHEET_WORKSHEET = None
DRIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_drive_cache():
    shutil.rmtree(DRIVE_CACHE_DIR, ignore_errors=True)


atexit.register(_cleanup_drive_cache)


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

    def __str__(self):
        return f"https://drive.google.com/uc?export=download&id={self.id}"


def get_credentials():
    """Return shared service-account credentials for Google APIs."""

    global _CREDENTIALS
    if _CREDENTIALS is None:
        creds_dict = None
        creds_path = None

        secret_creds = _get_secret("GCP_CREDENTIALS")
        if secret_creds:
            if isinstance(secret_creds, str):
                try:
                    creds_dict = json.loads(secret_creds)
                except json.JSONDecodeError as err:
                    st.error(f"Invalid JSON in GCP_CREDENTIALS secret: {err}")
                    st.stop()
            elif isinstance(secret_creds, dict):
                creds_dict = secret_creds
            else:
                try:
                    creds_dict = dict(secret_creds)
                except TypeError:
                    pass

        if creds_dict is None:
            env_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            if env_json:
                try:
                    creds_dict = json.loads(env_json)
                except json.JSONDecodeError as err:
                    st.error(f"Invalid JSON in GOOGLE_SERVICE_ACCOUNT_JSON env var: {err}")
                    st.stop()

        if creds_dict is None:
            env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GCP_CREDENTIALS_FILE")
            if env_path:
                creds_path = Path(env_path)

        if creds_dict is not None:
            try:
                _CREDENTIALS = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, GOOGLE_SCOPES)
            except Exception as err:
                st.error(f"Unable to load Google credentials from JSON: {err}")
                st.stop()
        elif creds_path is not None and creds_path.exists():
            try:
                _CREDENTIALS = ServiceAccountCredentials.from_json_keyfile_name(str(creds_path), GOOGLE_SCOPES)
            except Exception as err:
                st.error(f"Unable to load Google credentials from '{creds_path}': {err}")
                st.stop()
        else:
            st.error(
                "Google API credentials not configured. Provide them via Streamlit secrets (GCP_CREDENTIALS) or environment variables."
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
        except Exception as err:
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
        except Exception as err:
            st.error(f"Unable to initialize Google Drive client: {err}")
            st.stop()
    return _DRIVE_SERVICE


def send_to_google_sheet(data_row):
    """Append a data row to a Google Sheet.

    Parameters
    ----------
    data_row : list
        A list of data values to append as a row.
    """
    global _SHEET_WORKSHEET, _SHEET_HEADER_INITIALIZED

    if not GOOGLE_SHEET_URL:
        st.error("GOOGLE_SHEET_URL is not configured.")
        return

    client = get_sheet_client()
    try:
        if _SHEET_WORKSHEET is None:
            sheet = client.open_by_url(GOOGLE_SHEET_URL)
            _SHEET_WORKSHEET = sheet.sheet1
    except Exception as err:
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

        _SHEET_WORKSHEET.append_row(data_row, value_input_option="USER_ENTERED")
    except Exception as err:
        st.error(f"Unable to append row to Google Sheet: {err}")


def generate_user_id(length=6):
    """Generate a random user ID.

    Parameters
    ----------
    length : int, optional
        Length of the ID string.

    Returns
    -------
    str
        Randomly generated user ID.
    """
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


# Session State Initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = generate_user_id()
if "video_index" not in st.session_state:
    st.session_state.video_index = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = []
if "video_list" not in st.session_state:
    st.session_state.video_list = []
if "awaiting_explanation" not in st.session_state:
    st.session_state.awaiting_explanation = False
if "current_ann" not in st.session_state:
    st.session_state.current_ann = {}


def load_drive_videos(folder: DriveFolder, exts=VIDEO_EXTENSIONS):
    """Load video metadata from a Google Drive folder."""

    service = get_drive_service()
    videos = []
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
                if not name or not name.lower().endswith(exts):
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


def load_videos(folder, exts=VIDEO_EXTENSIONS):
    """Load video references from a local directory or Google Drive folder."""

    if isinstance(folder, DriveFolder):
        if not folder.id:
            st.error(f"No Google Drive folder ID configured for '{folder.label}'.")
            return []
        return load_drive_videos(folder, exts)

    folder_path = Path(folder)
    try:
        if not folder_path.is_dir():
            st.error(f"Directory '{folder_path}' does not exist.")
            return []
        return sorted(
            [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in exts],
            key=lambda path: path.name.lower(),
        )
    except Exception as err:
        st.error(f"Error reading {folder_path}: {err}")
        return []


def _folder_name(folder):
    if isinstance(folder, DriveFolder):
        return folder.label
    return Path(folder).name


def _is_drive_video(obj) -> bool:
    """Best-effort detection for DriveVideo instances across Streamlit reruns.

    Streamlit executes the script top-to-bottom on every interaction and
    redefines classes each time. Objects kept in ``st.session_state`` still
    reference the old class definition, so a direct ``isinstance`` check
    against the newly-defined class returns ``False`` even though the object
    still represents a Drive video. To stay resilient across reruns we simply
    look for the attributes we rely on (``id`` and ``name``).
    """

    return hasattr(obj, "id") and hasattr(obj, "name")


def _ensure_local_drive_copy(video: DriveVideo):
    """Ensure a local cached copy of a Drive video exists and return its path."""

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


def _video_mime(video: DriveVideo):
    if video.mime_type:
        return video.mime_type
    return VIDEO_MIME_TYPES.get(Path(video.name).suffix.lower())


def sample_and_mix(fake_folder, real_folder, n_each=DEFAULT_SAMPLE_SIZE):
    """Sample and mix fake and real videos.

    Parameters
    ----------
    fake_folder : Path or DriveFolder
        Source containing fake videos.
    real_folder : Path or DriveFolder
        Source containing real videos.
    n_each : int, optional
        Number of videos to sample from each folder.

    Returns
    -------
    list
        Shuffled list of sampled video references.
    """
    fake = load_videos(fake_folder)
    real = load_videos(real_folder)

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
    """Return tuple of (fake_source, real_source) for video sampling."""

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


def save_annotation(ann, video_path, base="annotations"):
    """Save a single annotation to a JSON file.

    Parameters
    ----------
    ann : dict
        Annotation data.
    video_path : Path or DriveVideo
        Reference to the video file being annotated.
    base : str, optional
        Base folder to save annotations.
    """
    sub = video_path.parent.name
    outdir = Path(base) / sub
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / (video_path.stem + ".json")
    try:
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(ann, f, indent=4)
    except Exception as e:
        st.warning(f"Could not save annotation for {outfile.name}: {e}")


# ---------------- UI START ----------------
st.title("AI-Generated Video Detection")
st.markdown(f"**🆔 Your session ID:** `{st.session_state.user_id}`")

if not st.session_state.video_list:
    sources = resolve_video_sources()
    videos = sample_and_mix(*sources, n_each=DEFAULT_SAMPLE_SIZE)
    if not videos:
        st.stop()
    st.session_state.video_list = videos

total = len(st.session_state.video_list)

if st.session_state.video_index >= total:
    st.write("### 🎉 All videos done. Thank you!")
    master = Path("annotations") / "all_annotations.json"
    master.parent.mkdir(exist_ok=True)
    with open(master, "w", encoding="utf-8") as f:
        json.dump(st.session_state.annotations, f, indent=4)

    correct = sum(1 for ann in st.session_state.annotations if ann["ai_generated"] == (ann["ground_truth"] == "Fake"))
    st.markdown(f"## You classified **{correct}/{total}** videos correctly!")
    st.balloons()
    st.stop()

idx = st.session_state.video_index
video_path = st.session_state.video_list[idx]

st.markdown(
    f"""
    <div style='padding: 0.75em 1.5em; background-color: #f0f2f6;
        border-left: 6px solid #4A90E2; border-radius: 8px;
        margin-bottom: 1em; font-size: 1.2em; font-weight: bold;'>
        🎬 Video <span style='color:#4A90E2'>{idx+1}</span> of <span style='color:#333'>{total}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if _is_drive_video(video_path):
    local_path = _ensure_local_drive_copy(video_path)
    if local_path and local_path.exists():
        st.video(str(local_path), format=_video_mime(video_path), start_time=0)
    else:
        st.error("Unable to load video from Google Drive. Please try again later.")
        st.stop()
else:
    st.video(str(video_path))

if st.session_state.awaiting_explanation:
    st.subheader("Tell us why you chose that:")
    reason = st.text_area("Your explanation:", key=f"reason_{idx}")
    if st.button("Next Video", key=f"next_{idx}"):
        ann = st.session_state.current_ann
        ann["explanation"] = reason
        ann["timestamp"] = datetime.now(timezone.utc).isoformat()

        save_annotation(ann, video_path)
        st.session_state.annotations.append(ann)

        row = [
            st.session_state.user_id,
            idx + 1,
            str(video_path),
            video_path.name,
            video_path.parent.name,
            "Yes" if ann["ai_generated"] else "No",
            reason,
            ann["timestamp"],
        ]
        try:
            send_to_google_sheet(row)
        except Exception as e:
            st.error(f"❌ Error writing to Google Sheet: {e}")

        st.session_state.awaiting_explanation = False
        st.session_state.current_ann = {}
        st.session_state.video_index += 1
        st.rerun()
    st.stop()

st.subheader("Is this video AI-generated?")
col1, col2 = st.columns(2)

if col1.button("Yes", key=f"yes_{idx}"):
    st.session_state.current_ann = {
        "video": video_path.name,
        "ground_truth": video_path.parent.name,
        "ai_generated": True,
        "video_url": str(video_path),
        "video_source": "drive" if _is_drive_video(video_path) else "local",
        "drive_id": getattr(video_path, "id", ""),
    }
    st.session_state.awaiting_explanation = True
    st.rerun()

if col2.button("No", key=f"no_{idx}"):
    st.session_state.current_ann = {
        "video": video_path.name,
        "ground_truth": video_path.parent.name,
        "ai_generated": False,
        "video_url": str(video_path),
        "video_source": "drive" if _is_drive_video(video_path) else "local",
        "drive_id": getattr(video_path, "id", ""),
    }
    st.session_state.awaiting_explanation = True
    st.rerun()
