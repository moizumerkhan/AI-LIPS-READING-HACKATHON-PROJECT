# AI Lip Reading App

Predicts spoken sentences from silent video, purely from lip movement (no audio).
Built on top of [mpc001/Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
(the LRS3 visual-only model, `LRS3_V_WER19.1`), with a Streamlit web interface.

**Live demo:** _add your Streamlit Cloud URL here after deploying_

## What's in this repo

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app — UI, video upload, and inference |
| `model_setup.py` | One-time setup logic: clones dependencies and downloads model weights automatically on first app load |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System-level (apt) dependencies Streamlit Cloud needs: ffmpeg, git-lfs, opencv system libs |
| `.gitignore` | Keeps large cloned repos and model weights out of git |

This repo does **not** include the base VSR repo, RetinaFace weights, or
pretrained model checkpoints (~1GB combined) — `model_setup.py` downloads
these automatically the first time the app runs, and caches them so it only
happens once per running instance.

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (as-is — don't manually add the `vsr/` folder or model files)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repo
3. Set the main file path to `app.py`
4. Deploy

**First load will be slow** (a few minutes) — it's cloning the base repo,
RetinaFace, and downloading ~1GB of model weights in the background. This
only happens once; after that, the app responds normally until the
container restarts (Streamlit Cloud free tier sleeps/restarts apps after
inactivity, which will trigger this setup again on next wake-up).

### Important limitation: no GPU

Streamlit Community Cloud's free tier is **CPU-only**. This model was
designed for GPU inference, so:
- Predictions will be noticeably slower than the Colab/GPU version
- Use short video clips (a few seconds) to keep inference time reasonable
- If the app runs out of memory, it's likely hitting the free tier's RAM
  limit — reducing video resolution/length before upload is the main lever
  available without upgrading the hosting tier

## Running locally

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
streamlit run app.py
```

`model_setup.py` runs automatically on first launch, same as on Streamlit
Cloud — no separate setup script needed.

## Known quirks (and why the code looks the way it does)

This project sits on top of a research codebase from 2023 that hasn't been
updated for newer library versions. A few workarounds are baked into the code:

- **`mediapipe` face detector doesn't work** on current mediapipe releases —
  a known upstream bug removed `mp.solutions` in mediapipe 0.10.30+, and older
  mediapipe versions aren't published for recent Python versions. This app
  uses **RetinaFace** instead (`detector="retinaface"`), which works fine.
- **`ibug.face_detection` / `ibug.face_alignment` can't be `pip install`-ed
  normally** — their `setup.py` doesn't correctly package subfolders on
  modern pip/setuptools (both editable and non-editable installs silently
  drop the actual model code). `model_setup.py` clones them directly and
  adds their folders to `sys.path` instead of relying on pip.
- **`torchvision.io.read_video` was removed** in newer torchvision releases.
  `app.py` monkey-patches it with an OpenCV-based equivalent.
- **Streamlit Cloud doesn't run arbitrary setup scripts** — only
  `requirements.txt` and `packages.txt` are installed automatically. All the
  cloning/downloading that would normally live in a `setup.sh` is instead
  done in Python inside `model_setup.py`, triggered on first app load and
  cached with `st.cache_resource`.

## Model performance

The `LRS3_V_WER19.1` model has a ~19% word error rate on the LRS3 benchmark.
Predictions on your own video may be less accurate than the benchmark,
especially with poor lighting, side angles, or fast speech. A clear,
front-facing, well-lit view of the mouth gives the best results.

## Credits

- Model & base inference code: [Pingchuan Ma et al., Imperial College London](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
- Face detection/alignment: [ibug-group](https://github.com/hhj1897/face_detection)
