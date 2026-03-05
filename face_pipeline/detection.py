from __future__ import annotations

from typing import List, Dict, Any, Tuple, Union
import os

import numpy as np
import cv2


def _get_onnx_providers() -> List[str]:
    """Return ONNX execution providers based on actual availability.

    Mirrors the logic from the working vista-face-recognition project:
    - If CUDAExecutionProvider is available, use CUDA + CPU.
    - Otherwise fall back to CPU only.
    """
    try:
        import onnxruntime as ort  # type: ignore

        if "CUDAExecutionProvider" in getattr(ort, "get_available_providers", lambda: [])():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


# InsightFace model packs: buffalo_l (best accuracy), buffalo_s (smaller), buffalo_sc (smallest, no alignment/attrs)
FACE_MODEL_CHOICES = ("buffalo_l", "buffalo_s", "buffalo_sc")


def load_detector(
    device: str = "cuda",
    det_size: Tuple[int, int] = (640, 640),
    model_name: str = "buffalo_l",
) -> Any:
    """Initialize insightface FaceAnalysis detector+recognition.

    model_name: one of buffalo_l, buffalo_s, buffalo_sc.
    Returns an app object with .get(image) -> list of faces (bbox, landmarks, embeddings).
    """
    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        raise RuntimeError(
            "insightface is required for the fallback detector. Install with: pip install insightface onnxruntime[-gpu]"
        ) from e

    name = model_name if model_name in FACE_MODEL_CHOICES else "buffalo_l"
    providers = _get_onnx_providers()
    app = FaceAnalysis(name=name, providers=providers)
    # Match vista-face-recognition: use ctx_id=0 with det_size=(640, 640)
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def _get_face_attr(face: Any, key: str, default: Any = None) -> Any:
    """Get attribute from InsightFace Face object (supports both object and dict-like access)."""
    v = getattr(face, key, None)
    if v is not None:
        return v
    try:
        return face[key]
    except (KeyError, TypeError):
        return default


def detect_faces(
    detector: Any,
    frame_bgr_or_path: Union[np.ndarray, str],
    conf_thresh: float = 0.5,
) -> List[Dict[str, Any]]:
    """Run face detection and extract bbox, confidence, and landmarks when available.

    frame_bgr_or_path: BGR image (numpy array) or path to image file; path matches vista-face-recognition.
    Returns list of dicts: {bbox, confidence, landmarks(optional), face_obj}
    InsightFace Face objects may be object- or dict-like; we support both.
    """
    if isinstance(frame_bgr_or_path, str):
        frame_bgr = cv2.imread(frame_bgr_or_path)
        if frame_bgr is None:
            return []
    else:
        frame_bgr = frame_bgr_or_path

    faces = detector.get(frame_bgr)
    results: List[Dict[str, Any]] = []
    h, w = frame_bgr.shape[:2]
    for f in faces:
        score = _get_face_attr(f, "det_score", 1.0)
        if score is None:
            score = 1.0
        score = float(score)
        if score < conf_thresh:
            continue
        bbox = _get_face_attr(f, "bbox", None)
        if bbox is None:
            continue
        # Handle numpy array or list
        bbox = getattr(bbox, "tolist", lambda: bbox)() if hasattr(bbox, "tolist") else list(bbox)
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(max(0, min(v, (w if i % 2 == 0 else h)))) for i, v in enumerate(bbox[:4])]
        record: Dict[str, Any] = {
            "bbox": [x1, y1, x2, y2],
            "confidence": score,
            "face_obj": f,
        }
        # Try 5-point landmarks (landmark_5 or kps)
        lm5 = _get_face_attr(f, "landmark_5", None) or _get_face_attr(f, "kps", None)
        if lm5 is not None and hasattr(lm5, "__len__") and len(lm5) >= 5:
            to_list = lambda p: p.tolist() if hasattr(p, "tolist") else list(p)
            record["landmarks"] = {
                "left_eye": to_list(lm5[0]),
                "right_eye": to_list(lm5[1]),
                "nose": to_list(lm5[2]),
                "mouth_left": to_list(lm5[3]),
                "mouth_right": to_list(lm5[4]),
            }
        results.append(record)
    return results


def crop_face(frame_bgr: np.ndarray, bbox: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return frame_bgr[y1:y2, x1:x2]


def save_image(img_bgr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr)