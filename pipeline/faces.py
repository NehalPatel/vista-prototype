"""Face detection overlay on annotated frames.

Runs InsightFace face detection on each image in a directory (e.g. YOLO-annotated
frames), draws face boxes, and returns per-frame face counts and bboxes for the API.
If insightface is not installed, returns empty results and skips drawing.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def run_face_detection(
    annotated_frames_dir: str,
    face_model: str = "buffalo_l",
    device: str = "cuda",
    face_conf_threshold: float = 0.3,
    source_frames_dir: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run face detection and draw face boxes on annotated frames.

    - If source_frames_dir is set, runs InsightFace on those (clean) frames for better detection,
      then draws cyan face boxes on the corresponding images in annotated_frames_dir.
    - Otherwise runs detection on images in annotated_frames_dir and overwrites them.
    - Returns faces_by_frame: { frame_filename: [ {"bbox": [x1,y1,x2,y2], "confidence": float}, ... ] }
    - If insightface is not available, returns {} and does not modify images.
    """
    try:
        from face_pipeline.detection import load_detector, detect_faces
    except Exception:
        return {}

    faces_by_frame: Dict[str, List[Dict[str, Any]]] = {}
    try:
        # Larger det_size (896) helps detect smaller/distant faces in multi-person videos
        detector = load_detector(device=device, model_name=face_model, det_size=(896, 896))
    except Exception:
        return {}

    import cv2

    list_dir = source_frames_dir if source_frames_dir and os.path.isdir(source_frames_dir) else annotated_frames_dir
    for fname in sorted(os.listdir(list_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path_for_detection = os.path.join(source_frames_dir, fname) if source_frames_dir else os.path.join(annotated_frames_dir, fname)
        path_annotated = os.path.join(annotated_frames_dir, fname)
        img_for_detection = cv2.imread(path_for_detection)
        if img_for_detection is None:
            continue
        dets = detect_faces(detector, img_for_detection, conf_thresh=face_conf_threshold)
        records = [
            {"bbox": d["bbox"], "confidence": round(float(d["confidence"]), 4)}
            for d in dets
        ]
        faces_by_frame[fname] = records
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
