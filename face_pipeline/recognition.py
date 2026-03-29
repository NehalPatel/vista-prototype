from typing import List, Tuple, Dict, Any
import os

import numpy as np
from pipeline.utils import canonical_display_name

from .face_database import FACE_DB_FILENAME


def load_known_embeddings(known_dir: str) -> List[Tuple[np.ndarray, str]]:
    """Load known embeddings from face_database.npy only (mean vector per person)."""
    db_path = os.path.join(known_dir, FACE_DB_FILENAME)
    if not os.path.isfile(db_path):
        return []
    try:
        data = np.load(db_path, allow_pickle=True).item()
        if not isinstance(data, dict):
            return []
        known: List[Tuple[np.ndarray, str]] = []
        for person, val in data.items():
            try:
                if isinstance(val, dict) and "mean" in val:
                    arr = np.asarray(val["mean"], dtype=np.float32)
                else:
                    arr = np.asarray(val, dtype=np.float32)
            except Exception:
                continue
            known.append((arr, canonical_display_name(str(person))))
        return known
    except Exception:
        return []


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance = 1 - cosine similarity."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    sim = float(np.dot(a, b) / denom)
    return float(1.0 - sim)


def match(embedding: np.ndarray, known: List[Tuple[np.ndarray, str]], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Match embedding against known set using cosine distance thresholds.

    thresholds: {"same": 0.6, "maybe": 0.8}
    Returns: {label, distance, confidence}
    """
    if not known:
        return {"label": "Unknown", "distance": 1.0, "confidence": 0.0}

    best_label = "Unknown"
    best_dist = 1.0
    for vec, label in known:
        d = cosine_distance(embedding, vec)
        if d < best_dist:
            best_dist = d
            best_label = label

    same_t = thresholds.get("same", 0.6)
    maybe_t = thresholds.get("maybe", 0.8)
    if best_dist < same_t:
        final_label = best_label
    elif best_dist < maybe_t:
        final_label = f"Maybe:{best_label}"
    else:
        final_label = "Unknown"

    confidence = float(max(0.0, 1.0 - min(best_dist, 1.0)))
    return {"label": final_label, "distance": best_dist, "confidence": confidence}
