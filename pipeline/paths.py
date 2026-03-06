"""Path constants and directory management for the VISTA prototype."""

from __future__ import annotations

import os


_THIS_DIR = os.path.dirname(__file__)
# Anchor paths to the repo root (not the runtime CWD) so the web app can be
# launched from anywhere (e.g. `web/`) without breaking model/data directories.
ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
VISTA_DIR = os.path.join(ROOT_DIR, "vista-prototype")
VIDEOS_DIR = os.path.join(VISTA_DIR, "videos")
FRAMES_DIR = os.path.join(VISTA_DIR, "frames")
DETECTIONS_DIR = os.path.join(VISTA_DIR, "detections")
RESULTS_DIR = os.path.join(VISTA_DIR, "results")

# Training Data Manager: raw uploads for face and monument training
TRAINING_DATA_DIR = os.path.join(VISTA_DIR, "training_data")
TRAINING_FACES_DIR = os.path.join(TRAINING_DATA_DIR, "faces")
TRAINING_MONUMENTS_DIR = os.path.join(TRAINING_DATA_DIR, "monuments")
# Kaggle / provided dataset (e.g. Indian monuments) - folder per class
TRAINING_DATASET_DIR = os.path.join(TRAINING_DATA_DIR, "dataset")
# Downloaded/unorganized datasets (e.g. Kaggle): run organize --from-datasets to copy into faces/ and monuments/
DATASETS_DIR = os.path.join(TRAINING_DATA_DIR, "datasets")
# Inbox: put unorganized images here in subfolders (subfolder name = person or monument name), then run organize script
INBOX_FACES_DIR = os.path.join(TRAINING_DATA_DIR, "inbox_faces")
INBOX_MONUMENTS_DIR = os.path.join(TRAINING_DATA_DIR, "inbox_monuments")
# Trained monument classifier and index
MONUMENT_MODEL_DIR = os.path.join(VISTA_DIR, "monument_model")


def ensure_directories() -> None:
    """Create required directories if missing."""
    for path in [
        VISTA_DIR,
        VIDEOS_DIR,
        DETECTIONS_DIR,
        RESULTS_DIR,
        TRAINING_DATA_DIR,
        TRAINING_FACES_DIR,
        TRAINING_MONUMENTS_DIR,
        INBOX_FACES_DIR,
        INBOX_MONUMENTS_DIR,
        MONUMENT_MODEL_DIR,
    ]:
        os.makedirs(path, exist_ok=True)
    # FRAMES_DIR created on demand per video; TRAINING_DATASET_DIR is user-provided
    for path in [FRAMES_DIR]:
        os.makedirs(path, exist_ok=True)


def get_video_results_paths(video_id: str):
    """Return dictionary of paths for a given video_id under RESULTS_DIR."""
    base = os.path.join(RESULTS_DIR, video_id)
    detection_json = os.path.join(base, "detection_results.json")
    processed_frames = os.path.join(base, "processed_frames")
    metadata_txt = os.path.join(base, "metadata.txt")
    return {
        "base": base,
        "detection_json": detection_json,
        "processed_frames": processed_frames,
        "metadata_txt": metadata_txt,
    }


def ensure_video_results_dirs(video_id: str) -> bool:
    """Create per-video results directories, returning success status.

    Returns False if creation fails.
    """
    try:
        paths = get_video_results_paths(video_id)
        os.makedirs(paths["base"], exist_ok=True)
        os.makedirs(paths["processed_frames"], exist_ok=True)
        return True
    except Exception:
        return False