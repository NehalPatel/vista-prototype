from typing import List, Dict, Any, Tuple
import os

import numpy as np
import cv2


def _get_providers(device: str) -> List[str]:
    # Default to ONNX Runtime providers expected by insightface
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_detector(device: str = "cuda", det_size: Tuple[int, int] = (640, 640)):
    """Initialize insightface FaceAnalysis detector+recognition as a practical fallback.

    Returns an app object with .get(image) -> list of faces that include bbox, landmarks, and embeddings.
    """
    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        raise RuntimeError(
            "insightface is required for the fallback detector. Install with: pip install insightface onnxruntime[-gpu]"
        ) from e

    providers = _get_providers(device)
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    # ctx_id: 0 for GPU, -1 for CPU
    ctx_id = 0 if device == "cuda" else -1
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def detect_faces(detector, frame_bgr: np.ndarray, conf_thresh: float = 0.8) -> List[Dict[str, Any]]:
    """Run face detection and extract bbox, confidence, and landmarks when available.

    Returns list of dicts: {bbox, confidence, landmarks(optional), face_obj}
    """
    faces = detector.get(frame_bgr)
    results: List[Dict[str, Any]] = []
    h, w = frame_bgr.shape[:2]
    for f in faces:
        score = float(getattr(f, "det_score", 1.0))
        if score < conf_thresh:
            continue
        bbox = getattr(f, "bbox", None)
        if bbox is None:
            # Some versions expose bbox via .bbox as np.ndarray [x1, y1, x2, y2]
            continue
        x1, y1, x2, y2 = [int(max(0, min(v, (w if i % 2 == 0 else h)))) for i, v in enumerate(bbox)]
        record: Dict[str, Any] = {
            "bbox": [x1, y1, x2, y2],
            "confidence": score,
            "face_obj": f,
        }
        # Try 5-point landmarks; else skip
        lm5 = getattr(f, "landmark_5", None)
        if lm5 is not None:
            record["landmarks"] = {
                "left_eye": lm5[0].tolist(),
                "right_eye": lm5[1].tolist(),
                "nose": lm5[2].tolist(),
                "mouth_left": lm5[3].tolist(),
                "mouth_right": lm5[4].tolist(),
            }
        results.append(record)
    return results


def crop_face(frame_bgr: np.ndarray, bbox: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return frame_bgr[y1:y2, x1:x2]


def save_image(img_bgr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr)