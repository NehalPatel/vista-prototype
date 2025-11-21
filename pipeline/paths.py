"""Path constants and directory management for the VISTA prototype."""

from __future__ import annotations

import os


ROOT_DIR = os.path.abspath(os.getcwd())
VISTA_DIR = os.path.join(ROOT_DIR, "vista-prototype")
VIDEOS_DIR = os.path.join(VISTA_DIR, "videos")
FRAMES_DIR = os.path.join(VISTA_DIR, "frames")
DETECTIONS_DIR = os.path.join(VISTA_DIR, "detections")
RESULTS_DIR = os.path.join(VISTA_DIR, "results")


def ensure_directories() -> None:
    """Create required directories if missing."""
    for path in [VISTA_DIR, VIDEOS_DIR, FRAMES_DIR, DETECTIONS_DIR, RESULTS_DIR]:
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