from __future__ import annotations

from contextlib import contextmanager
from typing import List, Dict, Any, Tuple, Union
import json
import os
import sys
import time
import warnings

import numpy as np
import cv2

# #region agent log
_DEBUG_LOG = os.path.join(os.path.dirname(__file__), "..", "debug-837bfb.log")
def _dbg(hid: str, loc: str, msg: str, data: dict) -> None:
    try:
        path = os.path.normpath(_DEBUG_LOG)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "837bfb", "hypothesisId": hid, "location": loc, "message": msg, "data": data, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass
# #endregion


def _get_onnx_providers(device: str = "cuda") -> List[str]:
    """Return ONNX execution providers for the requested device.

    - If device is 'cpu', returns CPU only.
    - If device is 'cuda', uses CUDA + CPU when CUDAExecutionProvider is available
      (requires onnxruntime-gpu); otherwise falls back to CPU.
    """
    if device == "cpu":
        # #region agent log
        _dbg("H2", "detection.py:_get_onnx_providers", "device_cpu", {"device": device, "returned": ["CPUExecutionProvider"]})
        # #endregion
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort  # type: ignore
        providers = getattr(ort, "get_available_providers", lambda: [])()
        if "CUDAExecutionProvider" in providers:
            out = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            # #region agent log
            _dbg("H2", "detection.py:_get_onnx_providers", "cuda_chosen", {"device": device, "available": providers, "returned": out})
            # #endregion
            return out
    except Exception:
        pass
    # #region agent log
    _dbg("H2", "detection.py:_get_onnx_providers", "cuda_fallback_cpu", {"device": device, "reason": "CUDAExecutionProvider not in available or exception"})
    # #endregion
    import warnings
    warnings.warn(
        "Face model: GPU requested but CUDAExecutionProvider not available. Using CPU. Install onnxruntime-gpu: pip uninstall onnxruntime; pip install onnxruntime-gpu",
        UserWarning,
        stacklevel=2,
    )
    return ["CPUExecutionProvider"]


# InsightFace model packs: buffalo_l (best accuracy), buffalo_s (smaller), buffalo_sc (smallest, no alignment/attrs)
FACE_MODEL_CHOICES = ("buffalo_l", "buffalo_s", "buffalo_sc")


@contextmanager
def _suppress_stdout_stderr():
    """Temporarily redirect stdout/stderr to devnull (e.g. to hide InsightFace/ONNX verbose prints)."""
    save_stdout, save_stderr = sys.stdout, sys.stderr
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout = sys.stderr = devnull
            yield
    finally:
        sys.stdout, sys.stderr = save_stdout, save_stderr


def load_detector(
    device: str = "cuda",
    det_size: Tuple[int, int] = (640, 640),
    model_name: str = "buffalo_l",
    silent: bool = False,
) -> Any:
    """Initialize insightface FaceAnalysis detector+recognition.

    model_name: one of buffalo_l, buffalo_s, buffalo_sc.
    silent: if True, suppress InsightFace/ONNX verbose output (Applied providers, find model, etc.).
    Returns an app object with .get(image) -> list of faces (bbox, landmarks, embeddings).
    """
    # Preload PyTorch CUDA DLLs when using GPU so onnxruntime-gpu (CUDA 11.8 build) can find them
    if device == "cuda":
        try:
            import torch  # noqa: F401
        except ImportError:
            pass

    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        raise RuntimeError(
            "insightface is required for the fallback detector. Install with: pip install insightface onnxruntime[-gpu]"
        ) from e

    name = model_name if model_name in FACE_MODEL_CHOICES else "buffalo_l"
    providers = _get_onnx_providers(device)
    # ctx_id: 0 = GPU 0, -1 = CPU (InsightFace convention); must match actual providers
    ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
    # #region agent log
    _dbg("H2", "detection.py:load_detector", "load_detector_called", {"device": device, "providers": providers, "ctx_id": ctx_id})
    # #endregion

    def _create_app() -> Any:
        app = FaceAnalysis(name=name, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        return app

    def _run() -> Any:
        nonlocal providers, ctx_id
        try:
            return _create_app()
        except Exception as e:  # e.g. cublasLt64_12.dll missing when CUDA provider loads
            err_msg = str(e).lower()
            if "cuda" in err_msg and (
                "cublaslt" in err_msg or "cublas_lt" in err_msg
                or "onnxruntime_providers_cuda" in err_msg
                or "error 126" in err_msg
                or "could not be found" in err_msg
                or "specified module" in err_msg
            ):
                warnings.warn(
                    "Face model: CUDA provider failed to load (e.g. cublasLt64_12.dll missing). Using CPU. "
                    "To use GPU: install onnxruntime-gpu for CUDA 11.8 (see docs/GPU.md Option A) or install CUDA 12 Toolkit (Option B).",
                    UserWarning,
                    stacklevel=2,
                )
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
                return _create_app()
            raise

    if silent:
        with _suppress_stdout_stderr():
            return _run()
    return _run()


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