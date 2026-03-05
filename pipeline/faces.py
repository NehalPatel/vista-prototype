"""Face detection overlay on annotated frames.

Runs InsightFace face detection on each image in a directory (e.g. YOLO-annotated
frames), draws face boxes, and returns per-frame face counts and bboxes for the API.
If insightface is not installed, returns empty results and skips drawing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def run_face_detection(
    annotated_frames_dir: str,
    face_model: str = "buffalo_l",
    device: str = "cuda",
    face_conf_threshold: float = 0.5,
    source_frames_dir: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run face detection and draw face boxes on annotated frames.

    - If source_frames_dir is set, runs InsightFace on those (clean) frames for better detection,
      then draws cyan face boxes on the corresponding images in annotated_frames_dir.
    - Otherwise runs detection on images in annotated_frames_dir and overwrites them.
    - Returns faces_by_frame: { frame_filename: [ {"bbox": [x1,y1,x2,y2], "confidence": float}, ... ] }
    - If insightface is not available, returns {} and does not modify images.
    """
    print("[trace] run_face_detection() entered")
    try:
        from face_pipeline.detection import load_detector, detect_faces
        print("[trace] face_pipeline.detection import OK")
    except Exception as e:
        print(f"[trace] Face detection skipped: insightface not available: {e}")
        logger.warning("Face detection skipped: insightface not available: %s", e)
        return {}

    faces_by_frame: Dict[str, List[Dict[str, Any]]] = {}
    try:
        # Match working vista-face-recognition project: det_size=(640, 640)
        detector = load_detector(device=device, model_name=face_model, det_size=(640, 640))
        print("[trace] load_detector() OK")
    except Exception as e:
        print(f"[trace] Face detection skipped: failed to load detector: {e}")
        logger.warning("Face detection skipped: failed to load detector: %s", e)
        return {}

    import cv2

    list_dir = source_frames_dir if source_frames_dir and os.path.isdir(source_frames_dir) else annotated_frames_dir
    frame_files = [f for f in sorted(os.listdir(list_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[trace] list_dir={list_dir!r} frame_count={len(frame_files)} first={frame_files[0] if frame_files else None!r}")
    for fname in frame_files:
        path_for_detection = os.path.join(source_frames_dir, fname) if source_frames_dir else os.path.join(annotated_frames_dir, fname)
        path_annotated = os.path.join(annotated_frames_dir, fname)
        if not os.path.isfile(path_for_detection):
            continue
        # Pass path (like vista-face-recognition) so detector reads image the same way
        dets = detect_faces(detector, path_for_detection, conf_thresh=face_conf_threshold)
        records = [
            {"bbox": d["bbox"], "confidence": round(float(d["confidence"]), 4)}
            for d in dets
        ]
        faces_by_frame[fname] = records
        if not dets and len(faces_by_frame) == 1:
            logger.info("Face detection ran but found no faces in first frame (threshold=%.2f). Check video content or lower face_conf_threshold.", face_conf_threshold)
        # Draw face boxes on the annotated image (so output video has both YOLO and face boxes)
        img_annotated = cv2.imread(path_annotated)
        if img_annotated is not None:
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                conf = d["confidence"]
                cv2.putText(
                    img_annotated, f"face {conf:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                )
            cv2.imwrite(path_annotated, img_annotated)

    return faces_by_frame
