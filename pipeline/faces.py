"""Face detection overlay on annotated frames.

Runs InsightFace face detection on each image in a directory (e.g. YOLO-annotated
frames), draws face boxes, and returns per-frame face counts and bboxes for the API.
If known_faces embeddings exist, runs recognition and draws celebrity names.
If insightface is not installed, returns empty results and skips drawing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Recognition thresholds: same <= 0.6, maybe <= 0.8
RECOGNITION_THRESHOLDS = {"same": 0.6, "maybe": 0.8}


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
    - If known_faces/embeddings exist, runs recognition and draws celebrity names on boxes.
    - Returns faces_by_frame: { frame_filename: [ {"bbox", "confidence", "label" (if recognition)}, ... ] }
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

    # Optional: load known faces for recognition (Training Data Manager datasets)
    known_faces: List[tuple] = []
    try:
        from face_pipeline.paths import KNOWN_FACES_DIR
        from face_pipeline.recognition import load_known_embeddings, match
        from face_pipeline.embeddings import get_embedding
        known_dir = str(KNOWN_FACES_DIR)
        if os.path.isdir(os.path.join(known_dir, "embeddings")):
            known_faces = load_known_embeddings(known_dir)
            if known_faces:
                print("[trace] face recognition enabled:", len(known_faces), "known embeddings")
            else:
                logger.info(
                    "Face recognition: no known face embeddings in %s. All faces will show as 'Unknown'. "
                    "Add a face dataset in Training Data Manager and click 'Train faces' to recognize people.",
                    known_dir,
                )
        else:
            logger.info(
                "Face recognition: known_faces/embeddings not found. Add a face dataset and click 'Train faces' to recognize people."
            )
    except Exception as e:
        logger.debug("Face recognition skipped (no known_faces or import error): %s", e)

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
        records: List[Dict[str, Any]] = []
        for d in dets:
            rec = {"bbox": d["bbox"], "confidence": round(float(d["confidence"]), 4)}
            if known_faces and "face_obj" in d:
                emb = get_embedding(d["face_obj"]) if known_faces else None
                if emb is not None:
                    m = match(emb, known_faces, RECOGNITION_THRESHOLDS)
                    rec["label"] = m.get("label", "Unknown")
                    rec["recognition_confidence"] = round(float(m.get("confidence", 0)), 4)
                else:
                    rec["label"] = "Unknown"
            else:
                rec["label"] = "Unknown"
            records.append(rec)
        faces_by_frame[fname] = records
        if not dets and len(faces_by_frame) == 1:
            logger.info("Face detection ran but found no faces in first frame (threshold=%.2f). Check video content or lower face_conf_threshold.", face_conf_threshold)
        # Draw face boxes on the annotated image (so output video has both YOLO and face boxes)
        img_annotated = cv2.imread(path_annotated)
        if img_annotated is not None:
            for d, rec in zip(dets, records):
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = rec.get("label", "Unknown")
                conf = rec.get("confidence", 0)
                text = f"{label} {conf:.2f}" if label != "Unknown" else f"face {conf:.2f}"
                cv2.putText(
                    img_annotated, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                )
            cv2.imwrite(path_annotated, img_annotated)

    return faces_by_frame
