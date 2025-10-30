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

Secrets & Configuration
------------------------
The utilities expect credentials and Drive/Sheet IDs via Streamlit secrets or
environment variables. For local runs, create .streamlit/secrets.toml in the
project root, e.g.:

    GOOGLE_SHEET_URL = "https://docs.google.com/..."
    DRIVE_FAKE_FOLDER_ID = "..."
    DRIVE_REAL_FOLDER_ID = "..."
    DEFAULT_SAMPLE_SIZE = 10
    GCP_CREDENTIALS = '''{... service-account JSON ...}'''

On Streamlit Cloud, paste the same keys under App → Settings → Secrets.
"""

# import json
# from pathlib import Path
from datetime import datetime, timezone
import threading

import streamlit as st

from utils import (
    DEFAULT_SAMPLE_SIZE,
    FAKE_METHODS_BALANCED,
    build_video_pools,
    fetch_replacement,
    generate_user_id,
    get_playback_path,
    is_drive_video,
    sample_initial_videos,
    # save_annotation,
    send_to_google_sheet,
    video_mime,
    shuffle_video_pools,
)


@st.cache_resource(show_spinner=False)
def get_video_loading_lock():
    return threading.Lock()


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
if "video_pools" not in st.session_state:
    st.session_state.video_pools = {}


# ---------------- UI START ----------------
st.title("AI-Generated Video Detection")
st.markdown(f"**🆔 Your session ID:** `{st.session_state.user_id}`")

if not st.session_state.video_pools:
    with st.spinner("Loading video catalogue..."):
        st.session_state.video_pools = build_video_pools()

if not st.session_state.video_list:
    with st.spinner("Preparing videos, please wait..."):
        # video_pools needs to be shuffled here to make sure the randomness
        # because build_video_pools has st.cache
        shuffle_video_pools(st.session_state.video_pools)
        videos = sample_initial_videos(
            st.session_state.video_pools,
            DEFAULT_SAMPLE_SIZE,
            fake_method_balanced=FAKE_METHODS_BALANCED,
        )
    if not videos:
        st.error("No videos available to annotate right now.")
        st.stop()
    st.session_state.video_list = videos


def ensure_video_playback(video_idx: int):
    with get_video_loading_lock():
        attempts = 0
        while attempts < 10:
            video = st.session_state.video_list[video_idx]
            playback_pth = get_playback_path(video, remove_audio=False)
            if playback_pth:
                return playback_pth, video
            # If there is a problem of video loading, find a replacement
            replacement = fetch_replacement(
                st.session_state.video_pools,
                video.label,
                getattr(video, "method", ""),
                exclude=video,
            )

            st.session_state.video_list[video_idx] = replacement
            attempts += 1
        return None, st.session_state.video_list[video_idx]


def record_annotation(annotation: dict, user_answer: str, comment: str, video_path, idx: int):
    """Persist the user's response locally and in Google Sheets."""

    ann_record = dict(annotation)
    ann_record["explanation"] = comment
    ann_record["timestamp"] = datetime.now(timezone.utc).isoformat()
    ann_record["user_answer"] = user_answer

    with st.spinner("Saving your answer..."):
        st.session_state.annotations.append(ann_record)

        row = [
            st.session_state.user_id,
            idx + 1,
            str(video_path),
            video_path.name,
            video_path.label,
            getattr(video_path, "method", ""),
            user_answer,
            comment,
            ann_record["timestamp"],
        ]
        try:
            send_to_google_sheet(row)
        except Exception as e:  # pylint: disable=broad-except
            st.error(f"❌ Error writing to Google Sheet: {e}")


total = len(st.session_state.video_list)

if st.session_state.video_index >= total:
    st.write("### 🎉 All videos done. Thank you!")
    # master = Path("annotations") / "all_annotations.json"
    # master.parent.mkdir(exist_ok=True)
    # with open(master, "w", encoding="utf-8") as f:
    #     json.dump(st.session_state.annotations, f, indent=4)

    reported = sum(1 for ann in st.session_state.annotations if ann.get("reported"))
    answered_annotations = [ann for ann in st.session_state.annotations if not ann.get("reported")]
    answered_total = len(answered_annotations)
    if answered_total:
        correct = sum(1 for ann in answered_annotations if ann.get("ai_generated") == (ann["ground_truth"] == "Fake"))
        st.markdown(f"## You classified **{correct}/{answered_total}** answered videos correctly!")
    else:
        st.markdown("## All videos were reported. Thank you for your vigilance!")
    if reported:
        st.info(f"You reported {reported} video(s) for violent content.")
    st.balloons()
    st.stop()

idx = st.session_state.video_index

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
with st.spinner("Retrieving video..."):
    playback_path, video_path = ensure_video_playback(idx)

if not playback_path:
    st.session_state.video_index += 1
    st.rerun()

if is_drive_video(video_path):
    st.video(str(playback_path), format=video_mime(video_path), start_time=0)
else:
    st.video(str(playback_path))

if st.session_state.awaiting_explanation:
    st.subheader("Tell us why you chose that:")
    reason = st.text_area("Your explanation:", key=f"reason_{idx}")
    col_next, col_back = st.columns([2, 1])
    if col_next.button("Next Video", key=f"next_{idx}"):
        ann = st.session_state.current_ann
        user_answer = "Yes" if ann["ai_generated"] else "No"
        record_annotation(ann, user_answer, reason, video_path, idx)
        st.session_state.awaiting_explanation = False
        st.session_state.current_ann = {}
        st.session_state.video_index += 1
        st.rerun()
    if col_back.button("Change Answer", key=f"back_{idx}"):
        st.session_state.awaiting_explanation = False
        st.session_state.current_ann = {}
        st.rerun()
    st.stop()

st.subheader("Is this video AI-generated?")
col1, col2, col3 = st.columns(3)
st.info(
    "We do our best to filter violent content. If this video feels violent or distressing, "
    "please click **Report, violent content** to skip it."
)


if col1.button("Yes", key=f"yes_{idx}"):
    st.session_state.current_ann = {
        "video": video_path.name,
        "ground_truth": video_path.label,
        "method": getattr(video_path, "method", ""),
        "ai_generated": True,
        "video_url": str(video_path),
        "video_source": "drive" if is_drive_video(video_path) else "local",
        "drive_id": getattr(video_path, "id", ""),
    }
    st.session_state.awaiting_explanation = True
    st.rerun()

if col2.button("No", key=f"no_{idx}"):
    st.session_state.current_ann = {
        "video": video_path.name,
        "ground_truth": video_path.label,
        "method": getattr(video_path, "method", ""),
        "ai_generated": False,
        "video_url": str(video_path),
        "video_source": "drive" if is_drive_video(video_path) else "local",
        "drive_id": getattr(video_path, "id", ""),
    }
    st.session_state.awaiting_explanation = True
    st.rerun()

if col3.button("Report, violent content", key=f"report_{idx}"):
    report_annotation = {
        "video": video_path.name,
        "ground_truth": video_path.label,
        "method": getattr(video_path, "method", ""),
        "ai_generated": None,
        "video_url": str(video_path),
        "video_source": "drive" if is_drive_video(video_path) else "local",
        "drive_id": getattr(video_path, "id", ""),
        "reported": True,
    }
    record_annotation(report_annotation, "Reported", "Flagged as violent content", video_path, idx)
    st.session_state.awaiting_explanation = False
    st.session_state.current_ann = {}
    st.session_state.video_index += 1
    st.rerun()
