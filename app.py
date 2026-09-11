"""
AI Lip Reading App — Streamlit version

Predicts spoken sentences from silent video using visual speech recognition
(no audio needed). Built on mpc001/Visual_Speech_Recognition_for_Multiple_Languages
(LRS3 visual-only model), with RetinaFace for face/mouth detection.

Deployment notes (Streamlit Community Cloud):
- No GPU on the free tier — this runs on CPU, which is slower than the
  Colab GPU version. A short (few-second) clip is recommended.
- All heavy setup (cloning dependencies, downloading ~1GB of model weights)
  happens automatically on first load, handled by model_setup.py, and is
  cached so it only runs once per container instance. First load will be
  slow (a few minutes); later interactions are fast.
"""

import os
import tempfile

import numpy as np
import cv2
import torch
import torchvision
import streamlit as st

import model_setup

st.set_page_config(page_title="AI Lip Reading App", page_icon="👄")
st.title("AI Lip Reading App")
st.write(
    "Upload a short video of someone talking. The model predicts the "
    "spoken sentence purely from lip movement — no audio required."
)

# ---------------------------------------------------------------------------
# One-time setup: clone dependencies, download model weights, wire sys.path.
# Cached — only runs once per running app instance.
# ---------------------------------------------------------------------------
with st.spinner("Setting up the model (first load only, can take a few minutes)..."):
    paths = model_setup.ensure_ready()

# ---------------------------------------------------------------------------
# Patch torchvision.io.read_video — newer torchvision dropped the legacy
# video backend this repo's detector code calls directly. Replaced with an
# OpenCV-based equivalent returning the same (T, H, W, C) uint8 format.
# ---------------------------------------------------------------------------
def _read_video_cv2(filename, pts_unit="sec"):
    cap = cv2.VideoCapture(filename)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be read from {filename}")
    video_tensor = torch.from_numpy(np.stack(frames))
    return video_tensor, torch.empty(0), {}


torchvision.io.read_video = _read_video_cv2

from pipelines.pipeline import InferencePipeline  # noqa: E402  (import after setup)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return InferencePipeline(
        paths["config_path"],
        device=device,
        face_track=True,
        detector="retinaface",
    ), device


with st.spinner("Loading inference pipeline..."):
    pipeline, device = load_pipeline()

st.caption(f"Running on: {device.upper()}" + (" (no GPU available — inference will be slower)" if device == "cpu" else ""))

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Transcribe"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Running inference..."):
                transcript = pipeline(tmp_path)
            st.success("Predicted sentence:")
            st.write(transcript)
        except Exception as exc:
            st.error(f"Error during inference: {exc}")
        finally:
            os.remove(tmp_path)
