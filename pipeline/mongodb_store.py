"""MongoDB storage for VISTA detection results.

Persists video metadata and per-frame detections (objects, faces, monuments)
so a separate search engine project can query by face label, object label, etc.

Set MONGODB_URI (or MONGO_URI) in .env or environment to enable; optional VISTA_DB_NAME (default: vista_search).
If MONGODB_URI is not set, indexing is skipped.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env from repo root so MONGODB_URI is set when running scripts or one-liners from repo root
def _load_dotenv() -> None:
    if os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI"):
        return
    try:
        from dotenv import load_dotenv
        repo_root = Path(__file__).resolve().parents[1]
        load_dotenv(repo_root / ".env")
    except Exception:
        # Safe to ignore; MongoDB remains optional
        pass

_load_dotenv()

logger = logging.getLogger(__name__)

# Default database name for VISTA search index
DEFAULT_DB_NAME = "vista_search"

# Collection names
VIDEOS_COLLECTION = "videos"
FRAMES_COLLECTION = "frames"
FACE_BUILD_STATE_COLLECTION = "face_build_state"

_client: Any = None
_db: Any = None


def _get_uri() -> Optional[str]:
    return os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")


def get_client():
    """Return pymongo MongoClient or None if MongoDB is not configured."""
    global _client
    uri = _get_uri()
    if not uri:
        return None
    if _client is None:
        try:
            from pymongo import MongoClient
            _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            _client.admin.command("ping")
        except Exception as e:
            logger.warning("MongoDB connection failed: %s", e)
            return None
    return _client


def get_db():
    """Return database instance or None if MongoDB is not configured."""
    global _db
    client = get_client()
    if client is None:
        return None
    if _db is None:
        db_name = os.environ.get("VISTA_DB_NAME", DEFAULT_DB_NAME)
        _db = client[db_name]
    return _db


def ensure_indexes() -> bool:
    """Create indexes on videos and frames. Call once at startup or before first write."""
    db = get_db()
    if db is None:
        return False
    try:
        videos = db[VIDEOS_COLLECTION]
        videos.create_index("video_id", unique=True)
        frames = db[FRAMES_COLLECTION]
        frames.create_index([("video_id", 1), ("frame_filename", 1)], unique=True)
        frames.create_index("video_id")
        frames.create_index("faces.label")
        frames.create_index("objects.class")
        frames.create_index("objects.color")
        frames.create_index("objects.label")
        frames.create_index("monument.label")
        face_build = db[FACE_BUILD_STATE_COLLECTION]
        face_build.create_index([("person", 1), ("image_path_rel", 1)], unique=True)
        face_build.create_index("person")
        return True
    except Exception as e:
        logger.warning("MongoDB ensure_indexes failed: %s", e)
        return False


def load_face_build_state() -> Dict[str, Dict[str, str]]:
    """Load face build state from MongoDB: { person -> { rel_path -> embedding_file } }.
    Returns empty dict if MongoDB is not configured or on error.
    """
    db = get_db()
    if db is None:
        return {}
    try:
        coll = db[FACE_BUILD_STATE_COLLECTION]
        state: Dict[str, Dict[str, str]] = {}
        for doc in coll.find({}, {"_id": 0, "person": 1, "image_path_rel": 1, "embedding_file": 1}):
            person = doc.get("person", "")
            rel = doc.get("image_path_rel", "")
            emb = doc.get("embedding_file", "")
            if person and rel:
                state.setdefault(person, {})[rel] = emb
        return state
    except Exception as e:
        logger.warning("MongoDB load_face_build_state failed: %s", e)
        return {}


def save_person_face_state(person: str, state_person: Dict[str, str]) -> bool:
    """Replace all state entries for this person with state_person (rel_path -> embedding_file).
    Returns True if write succeeded, False if MongoDB not configured or on error.
    """
    db = get_db()
    if db is None:
        return False
    try:
        coll = db[FACE_BUILD_STATE_COLLECTION]
        coll.delete_many({"person": person})
        if state_person:
            docs = [
                {"person": person, "image_path_rel": rel, "embedding_file": emb}
                for rel, emb in state_person.items()
            ]
            if docs:
                coll.insert_many(docs)
        return True
    except Exception as e:
        logger.warning("MongoDB save_person_face_state failed: %s", e)
        return False


def remove_person_face_state(person: str) -> bool:
    """Remove all state entries for this person. Returns True if succeeded."""
    db = get_db()
    if db is None:
        return False
    try:
        db[FACE_BUILD_STATE_COLLECTION].delete_many({"person": person})
        return True
    except Exception as e:
        logger.warning("MongoDB remove_person_face_state failed: %s", e)
        return False


def clear_face_build_state() -> bool:
    """Remove all documents in face_build_state collection. Used for --full rebuild."""
    db = get_db()
    if db is None:
        return False
    try:
        db[FACE_BUILD_STATE_COLLECTION].delete_many({})
        return True
    except Exception as e:
        logger.warning("MongoDB clear_face_build_state failed: %s", e)
        return False


def upsert_video(video_doc: Dict[str, Any]) -> bool:
    """Insert or replace one video document (by video_id)."""
    db = get_db()
    if db is None:
        return False
    try:
        coll = db[VIDEOS_COLLECTION]
        video_id = video_doc.get("video_id")
        if not video_id:
            return False
        coll.replace_one(
            {"video_id": video_id},
            video_doc,
            upsert=True,
        )
        return True
    except Exception as e:
        logger.warning("MongoDB upsert_video failed: %s", e)
        return False


def replace_frames_for_video(video_id: str, frames_docs: List[Dict[str, Any]]) -> bool:
    """Delete all frames for this video_id and insert the new list."""
    db = get_db()
    if db is None:
        return False
    try:
        coll = db[FRAMES_COLLECTION]
        coll.delete_many({"video_id": video_id})
        if frames_docs:
            coll.insert_many(frames_docs)
        return True
    except Exception as e:
        logger.warning("MongoDB replace_frames_for_video failed: %s", e)
        return False


def _frame_index_from_filename(frame_filename: str) -> int:
    """Extract 1-based frame index from filename, e.g. frame_0001.jpg -> 1."""
    m = re.search(r"(\d+)", frame_filename)
    if m:
        return int(m.group(1), 10)
    return 0


def index_detection_results_to_mongodb(
    video_id: str,
    source_url: str,
    meta: Dict[str, Any],
    run_stats: Dict[str, Any],
    results_by_frame: Dict[str, List[Dict]],
    faces_by_frame: Dict[str, List[Dict]],
    monuments_by_frame: Dict[str, Dict],
    by_class: Dict[str, int],
    confidence_threshold: float,
    object_model: str,
    face_model: str,
    fps: float = 1.0,
) -> bool:
    """
    Build video and frame documents from pipeline results and write to MongoDB.

    Returns True if write succeeded, False if MongoDB not configured or on error.
    """
    db = get_db()
    if db is None:
        return False

    fbf = faces_by_frame or {}
    mbf = monuments_by_frame or {}

    # Unique labels for search
    face_labels_set = set()
    object_labels_set = set()
    monument_labels_set = set()

    frames_docs: List[Dict[str, Any]] = []
    for frame_filename, dets in sorted(results_by_frame.items()):
        frame_index = _frame_index_from_filename(frame_filename)
        time_sec = (frame_index - 1) / fps if fps > 0 else 0.0

        # Normalize objects for storage
        objects = []
        for d in dets:
            obj = {
                "class": d.get("class", "unknown"),
                "color": d.get("color", ""),
                "label": d.get("label", "").strip() or (d.get("color", "") + " " + d.get("class", "unknown")).strip(),
                "conf": round(float(d.get("conf", 0)), 4),
            }
            if "bbox" in d:
                obj["bbox"] = d["bbox"]
            objects.append(obj)
            if obj["label"]:
                object_labels_set.add(obj["label"])

        faces = fbf.get(frame_filename, [])
        face_list = []
        for f in faces:
            label = (f.get("label") or "Unknown").strip()
            if label.startswith("Maybe:"):
                label = label[6:].strip() or "Unknown"
            face_list.append({
                "label": label,
                "confidence": round(float(f.get("confidence", 0)), 4),
                "recognition_confidence": round(float(f.get("recognition_confidence", 0)), 4),
                "bbox": f.get("bbox", []),
            })
            if label and label != "Unknown":
                face_labels_set.add(label)

        mon = mbf.get(frame_filename, {})
        monument = {}
        if isinstance(mon, dict) and mon:
            monument = {
                "label": (mon.get("label") or "Unknown").strip(),
                "confidence": round(float(mon.get("confidence", 0)), 4),
                "bbox": mon.get("bbox", []),
            }
            if monument["label"] and monument["label"] != "Unknown":
                monument_labels_set.add(monument["label"])

        frames_docs.append({
            "video_id": video_id,
            "frame_filename": frame_filename,
            "frame_index": frame_index,
            "time_sec": round(time_sec, 2),
            "objects": objects,
            "faces": face_list,
            "monument": monument,
        })

    total_detections = sum(len(d) for d in results_by_frame.values())
    total_face_detections = sum(len(fbf.get(f, [])) for f in results_by_frame)
    total_monument_detections = sum(1 for f in results_by_frame if mbf.get(f) and (mbf.get(f) or {}).get("label") not in (None, "", "Unknown"))

    video_doc = {
        "video_id": video_id,
        "source_url": source_url,
        "title": (meta.get("title") or "").strip() or None,
        "duration_sec": meta.get("duration") if isinstance(meta.get("duration"), (int, float)) else None,
        "thumbnail": (meta.get("thumbnail") or "").strip() or None,
        "processed_at": datetime.now(timezone.utc),
        "confidence_threshold": confidence_threshold,
        "object_model": object_model,
        "face_model": face_model,
        "run_stats": run_stats,
        "summary": {
            "total_frames": len(results_by_frame),
            "total_detections": total_detections,
            "total_face_detections": total_face_detections,
            "total_monument_detections": total_monument_detections,
        },
        "face_labels": sorted(face_labels_set),
        "object_labels": sorted(object_labels_set),
        "monument_labels": sorted(monument_labels_set),
    }

    if not ensure_indexes():
        return False
    if not upsert_video(video_doc):
        return False
    if not replace_frames_for_video(video_id, frames_docs):
        return False
    logger.info("Indexed video %s to MongoDB (%d frames)", video_id, len(frames_docs))
    return True
