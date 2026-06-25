from __future__ import annotations

import platform
import subprocess
import sys

from flask import Blueprint, jsonify

from pipeline.mongodb_store import FRAMES_COLLECTION, VIDEOS_COLLECTION, get_db

try:
    import yt_dlp  # type: ignore
except Exception:
    yt_dlp = None


system_bp = Blueprint("system", __name__)


def get_system_info():
    """Return Python version, CPU name, and GPU name (if available)."""
    python_version = platform.python_version()
    cpu_name = platform.processor() or "—"
    if sys.platform == "win32" and (not cpu_name or cpu_name.strip() == ""):
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"],
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0,
                timeout=5,
            )
            lines = out.decode("utf-8", errors="replace").strip().splitlines()
            if len(lines) >= 2:
                cpu_name = lines[1].strip() or cpu_name
        except Exception:
            pass
    gpu_name = "—"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0) or "CUDA GPU"
    except Exception:
        pass
    mongo_status = "disabled"
    mongo_db_name: str | None = None
    mongo_videos: int | None = None
    mongo_frames: int | None = None
    try:
        db = get_db()
        if db is not None:
            mongo_status = "connected"
            mongo_db_name = str(db.name)
            try:
                mongo_videos = db[VIDEOS_COLLECTION].estimated_document_count()
                mongo_frames = db[FRAMES_COLLECTION].estimated_document_count()
            except Exception:
                pass
    except Exception:
        mongo_status = "error"
    return {
        "python_version": python_version,
        "cpu": cpu_name,
        "gpu": gpu_name,
        "mongo_status": mongo_status,
        "mongo_db_name": mongo_db_name,
        "mongo_videos": mongo_videos,
        "mongo_frames": mongo_frames,
    }


def get_video_metadata(url: str):
    if not yt_dlp:
        return {}
    try:
        ydl_opts = {"quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title")
            duration = info.get("duration")  # seconds
            thumbs = info.get("thumbnails") or []
            thumb_url = None
            if thumbs:
                thumbs_sorted = sorted(
                    thumbs, key=lambda t: t.get("width", 0), reverse=True
                )
                thumb_url = (thumbs_sorted[0] or {}).get("url")
            return {"title": title, "duration": duration, "thumbnail": thumb_url}
    except Exception:
        return {}


@system_bp.get("/api/system-info")
def api_system_info():
    return jsonify(get_system_info())

