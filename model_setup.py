"""
One-time setup logic for Streamlit Cloud deployment.

Streamlit Cloud only runs `pip install -r requirements.txt` (and apt packages
from packages.txt) automatically. It does NOT run setup.sh or any other
script. So instead, this module does the equivalent setup work itself,
in Python, the first time the app loads — and is wrapped in
st.cache_resource so it only runs once per running container instead of
on every user interaction (Streamlit reruns the whole script on every
button click).
"""

import os
import subprocess
import sys

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VSR_DIR = os.path.join(BASE_DIR, "vsr")
THIRD_PARTY_DIR = os.path.join(VSR_DIR, "third_party")


def _run(cmd, cwd=None):
    """Run a shell command, raising with full output if it fails."""
    result = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {cmd}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    return result


def _clone_base_repo():
    if os.path.exists(os.path.join(VSR_DIR, "pipelines")):
        return
    _run(
        "git clone --depth 1 "
        "https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages.git vsr",
        cwd=BASE_DIR,
    )


def _clone_face_detection_alignment():
    os.makedirs(THIRD_PARTY_DIR, exist_ok=True)

    face_det_dir = os.path.join(THIRD_PARTY_DIR, "face_detection")
    if not os.path.exists(face_det_dir):
        _run("git lfs install", cwd=THIRD_PARTY_DIR)
        _run("git clone https://github.com/hhj1897/face_detection.git", cwd=THIRD_PARTY_DIR)
        _run("git lfs pull", cwd=face_det_dir)

    face_align_dir = os.path.join(THIRD_PARTY_DIR, "face_alignment")
    if not os.path.exists(face_align_dir):
        _run("git clone https://github.com/hhj1897/face_alignment.git", cwd=THIRD_PARTY_DIR)


def _download_model_weights():
    models_dir = os.path.join(VSR_DIR, "benchmarks", "LRS3", "models", "LRS3_V_WER19.1")
    lm_dir = os.path.join(VSR_DIR, "benchmarks", "LRS3", "language_models", "lm_en_subword")

    if not os.path.exists(os.path.join(models_dir, "model.pth")):
        os.makedirs(os.path.dirname(models_dir), exist_ok=True)
        zip_path = os.path.join(VSR_DIR, "visual_model.zip")
        _run(f"gdown 1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6 -O {zip_path}")
        _run(f"unzip -o {zip_path} -d {os.path.dirname(models_dir)}")
        os.remove(zip_path)

    if not os.path.exists(os.path.join(lm_dir, "model.pth")):
        os.makedirs(os.path.dirname(lm_dir), exist_ok=True)
        zip_path = os.path.join(VSR_DIR, "lm.zip")
        _run(f"gdown 1g31HGxJnnOwYl17b70ObFQZ1TSnPvRQv -O {zip_path}")
        _run(f"unzip -o {zip_path} -d {os.path.dirname(lm_dir)}")
        os.remove(zip_path)


@st.cache_resource(show_spinner=False)
def ensure_ready():
    """
    Runs the full one-time setup (clone repos, download weights) and returns
    the paths needed to load the inference pipeline. Cached so this heavy
    work only happens once per running app container, not on every rerun.
    """
    os.makedirs(VSR_DIR, exist_ok=True)

    _clone_base_repo()
    _clone_face_detection_alignment()
    _download_model_weights()

    # Make the ibug packages importable. Their setup.py doesn't package
    # subfolders correctly under normal pip install, so we add the cloned
    # folders to sys.path directly instead (same fix used in the Colab version).
    for path in (
        os.path.join(THIRD_PARTY_DIR, "face_detection"),
        os.path.join(THIRD_PARTY_DIR, "face_alignment"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)

    if VSR_DIR not in sys.path:
        sys.path.insert(0, VSR_DIR)

    return {
        "vsr_dir": VSR_DIR,
        "config_path": os.path.join(VSR_DIR, "configs", "LRS3_V_WER19.1.ini"),
    }
