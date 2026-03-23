"""Option B: Get face embedding from small/cropped face images (e.g. ~4KB thumbnails).

We try, in order: (1) run the detector on the image at its original size (no resize);
(2) if no face, upscale and run detection with normal then lower confidence;
(3) if still nothing, run the recognition model on a 112x112 resize (ArcFace requires
    fixed 112x112 input by architecture, so that step cannot avoid resizing).
Used only by build_models_cropped_faces.py for testing.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from .detection import detect_faces
from .embeddings import get_embedding

# Images with max dimension below this are treated as "small crop".
CROPPED_FACE_MAX_SIZE = 256

# Target size for upscaling so the detector gets a larger "face" in frame.
DETECT_INPUT_SIZE = 640

# ArcFace/InsightFace recognition model expects 112x112 input.
RECOG_INPUT_SIZE = 112


def _embedding_from_recognition_model(detector: Any, bgr_image: np.ndarray) -> Optional[np.ndarray]:
    """Run the recognition model directly on a 112x112 crop (no detection). Returns 512-d embedding or None."""
    try:
        models = getattr(detector, "models", None)
        if models is None:
            return None
        recognizer = None
        if isinstance(models, dict):
            recognizer = models.get("recognition") or models.get("rec")
        elif isinstance(models, (list, tuple)) and len(models) >= 2:
            recognizer = models[1]
        if recognizer is None or not hasattr(recognizer, "get_feat"):
            return None
        # ArcFaceONNX.get_feat(imgs) expects a list of BGR images (HWC); it resizes and normalizes internally.
        small = cv2.resize(bgr_image, (RECOG_INPUT_SIZE, RECOG_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        feats = recognizer.get_feat(small)
        if feats is None or (hasattr(feats, "size") and feats.size == 0):
            return None
        emb = np.asarray(feats, dtype=np.float32).flatten()
        if emb.size >= 512:
            return emb[:512].copy()
        return None
    except Exception:
        return None


def get_embedding_for_small_crop(
    detector: Any,
    bgr_image: np.ndarray,
    conf_thresh: float = 0.8,
) -> Optional[np.ndarray]:
    """Get embedding for a small image assumed to be a single face crop.

    1) Run the detector on the image at its original size (no resize); if a face is found, return its embedding.
    2) If not, upscale to DETECT_INPUT_SIZE and run detection with conf_thresh then 0.4.
    3) If still no face, run the recognition model on a 112x112 resize (ArcFace has fixed 112x112 input).
    """
    h, w = bgr_image.shape[:2]
    if max(h, w) > CROPPED_FACE_MAX_SIZE:
        return None
    if h <= 0 or w <= 0:
        return None

    # 1) Try detector on original small size (no resize)
    for thresh in (conf_thresh, 0.4):
        dets = detect_faces(detector, bgr_image, conf_thresh=thresh)
        if dets:
            dets.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
            emb = get_embedding(dets[0].get("face_obj"))
            if emb is not None:
                return emb

    # 2) Try upscaled image so detector sees a larger "face"
    resized = cv2.resize(
        bgr_image,
        (DETECT_INPUT_SIZE, DETECT_INPUT_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )
    for thresh in (conf_thresh, 0.4):
        dets = detect_faces(detector, resized, conf_thresh=thresh)
        if dets:
            dets.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
            emb = get_embedding(dets[0].get("face_obj"))
            if emb is not None:
                return emb

    # 3) Recognition model only (must resize to 112x112 - ArcFace has fixed input size)
    return _embedding_from_recognition_model(detector, bgr_image)
