"""Single-file face identity store: mean embedding + sample count per person."""

from __future__ import annotations

import os

import numpy as np

FACE_DB_FILENAME = "face_database.npy"


def load_face_database(known_faces_dir: str) -> dict:
    """Load face_database.npy. Returns {person_key: {"mean": vec, "count": n}}.

    Accepts legacy entries (person_key -> plain vector) and normalizes to mean+count.
    """
    db_path = os.path.join(known_faces_dir, FACE_DB_FILENAME)
    if not os.path.isfile(db_path):
        return {}
    try:
        data = np.load(db_path, allow_pickle=True).item()
        if not isinstance(data, dict):
            return {}
        out: dict = {}
        for person, val in data.items():
            try:
                if isinstance(val, dict) and "mean" in val:
                    out[str(person)] = {
                        "mean": np.asarray(val["mean"], dtype=np.float32),
                        "count": int(val.get("count", 1)),
                    }
                else:
                    out[str(person)] = {
                        "mean": np.asarray(val, dtype=np.float32),
                        "count": 1,
                    }
            except Exception:
                continue
        return out
    except Exception:
        return {}


def save_face_database(known_faces_dir: str, db: dict) -> None:
    """Persist db to known_faces_dir/face_database.npy."""
    os.makedirs(known_faces_dir, exist_ok=True)
    db_path = os.path.join(known_faces_dir, FACE_DB_FILENAME)
    np.save(db_path, db, allow_pickle=True)


def merge_embeddings_for_person(
    db: dict,
    person_display_key: str,
    new_embeddings: list[np.ndarray],
) -> int:
    """Update db in-place with new face embeddings for one person (incremental mean).

    person_display_key should match keys used at recognition time (e.g. canonical_display_name).
    Returns number of new embeddings merged.
    """
    if not new_embeddings:
        return 0
    stack = np.stack([np.asarray(e, dtype=np.float32) for e in new_embeddings], axis=0)
    n_new = stack.shape[0]
    new_mean = np.mean(stack, axis=0).astype(np.float32)

    existing = db.get(person_display_key)
    if existing is None:
        db[person_display_key] = {"mean": new_mean, "count": n_new}
        return n_new

    n_old = int(existing["count"])
    mean_old = np.asarray(existing["mean"], dtype=np.float32)
    merged = (mean_old * n_old + np.sum(stack, axis=0)) / (n_old + n_new)
    db[person_display_key] = {"mean": merged.astype(np.float32), "count": n_old + n_new}
    return n_new


__all__ = [
    "FACE_DB_FILENAME",
    "load_face_database",
    "save_face_database",
    "merge_embeddings_for_person",
]
