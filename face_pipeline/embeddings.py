from typing import Optional
import os

import numpy as np


def get_embedding(face_obj) -> Optional[np.ndarray]:
    """Return 512-d embedding from insightface face object if available.

    Uses normed_embedding when present; falls back to embedding.
    """
    emb = getattr(face_obj, "normed_embedding", None)
    if emb is None:
        emb = getattr(face_obj, "embedding", None)
    if emb is None:
        return None
    arr = np.asarray(emb, dtype=np.float32)
    return arr


def save_embedding(path: str, embedding: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embedding)