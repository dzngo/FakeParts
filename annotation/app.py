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

import json
from pathlib import Path
from datetime import datetime, timezone
import threading

import streamlit as st

from utils import (
    DEFAULT_SAMPLE_SIZE,
    build_video_pools,
    fetch_replacement,
    generate_user_id,
    get_playback_path,
    is_drive_video,
    sample_initial_videos,
    save_annotation,
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
# if "annotations" not in st.session_state:
#     st.session_state.annotations = []
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
        videos = sample_initial_videos(st.session_state.video_pools, DEFAULT_SAMPLE_SIZE)
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
            replacement = fetch_replacement(
                st.session_state.video_pools,
                video.label,
                getattr(video, "method", ""),
            )
            st.session_state.video_list[video_idx] = replacement
            attempts += 1
        return None, st.session_state.video_list[video_idx]


total = len(st.session_state.video_list)

if st.session_state.video_index >= total:
    st.write("### 🎉 All videos done. Thank you!")
    # master = Path("annotations") / "all_annotations.json"
    # master.parent.mkdir(exist_ok=True)
    # with open(master, "w", encoding="utf-8") as f:
    #     json.dump(st.session_state.annotations, f, indent=4)

    # correct = sum(1 for ann in st.session_state.annotations if ann["ai_generated"] == (ann["ground_truth"] == "Fake"))
    # st.markdown(f"## You classified **{correct}/{total}** videos correctly!")
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
        ann["explanation"] = reason
        ann["timestamp"] = datetime.now(timezone.utc).isoformat()
        with st.spinner("Saving your answer..."):
            # save_annotation(ann, video_path)
            # st.session_state.annotations.append(ann)

            row = [
                st.session_state.user_id,
                idx + 1,
                str(video_path),
                video_path.name,
                video_path.label,
                getattr(video_path, "method", ""),
                "Yes" if ann["ai_generated"] else "No",
                reason,
                ann["timestamp"],
            ]
            try:
                send_to_google_sheet(row)
            except Exception as e:  # pylint: disable=broad-except
                st.error(f"❌ Error writing to Google Sheet: {e}")

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
col1, col2 = st.columns(2)

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
