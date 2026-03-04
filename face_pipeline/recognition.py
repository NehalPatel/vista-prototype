from typing import List, Tuple, Dict, Any
import os
import json

import numpy as np


def load_known_embeddings(known_dir: str) -> List[Tuple[np.ndarray, str]]:
    """Load known embeddings and optional labels mapping.

    Expects embeddings under `known_dir/embeddings/*.npy` and optional `labels.json` mapping filename → label.
    """
    emb_dir = os.path.join(known_dir, "embeddings")
    labels_path = os.path.join(known_dir, "labels.json")
    labels: Dict[str, str] = {}
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            labels = {}

    known: List[Tuple[np.ndarray, str]] = []
    if os.path.isdir(emb_dir):
        for fname in os.listdir(emb_dir):
            if not fname.lower().endswith(".npy"):
                continue
            fpath = os.path.join(emb_dir, fname)
            try:
                vec = np.load(fpath).astype(np.float32)
                label = labels.get(fname, os.path.splitext(fname)[0])
                known.append((vec, label))
            except Exception:
                continue
    return known


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