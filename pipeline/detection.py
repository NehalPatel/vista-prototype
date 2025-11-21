"""Object detection using YOLOv8 and helpers to annotate frames.

This module provides:
- run_yolo: runs detection over frames and writes annotated images
- generate_summary: returns counts
- save_detection_results: writes a single JSON with all detections
- write_metadata: writes a text metadata file
"""

from typing import Dict, List, Tuple, Any
import os
import json

from ultralytics import YOLO  # type: ignore


def run_yolo(
    frames_dir: str,
    detections_dir: str,
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.7,
) -> Dict[str, List[Dict]]:
    """Run YOLOv8 on frames, save annotated images, and return filtered detections.

    Only detections with confidence >= conf_threshold are included.
    """
    os.makedirs(detections_dir, exist_ok=True)
    model = YOLO(model_path)
    results_by_frame: Dict[str, List[Dict]] = {}

    import cv2

    for fname in sorted(os.listdir(frames_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        frame_path = os.path.join(frames_dir, fname)
        result = model(frame_path)
        detections: List[Dict] = []
        boxes = result[0].boxes
        names = result[0].names
        for i in range(len(boxes)):
            b = boxes[i]
            xyxy = b.xyxy[0].tolist()
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            if conf >= conf_threshold:
                detections.append({
                    "bbox": xyxy,
                    "class": names.get(cls, str(cls)),
                    "conf": conf,
                })
        results_by_frame[fname] = detections

        # Save annotated image (BGR numpy array)
        annotated = result[0].plot()
        out_path = os.path.join(detections_dir, fname)
        cv2.imwrite(out_path, annotated)

    return results_by_frame


def generate_summary(results_by_frame: Dict[str, List[Dict]]) -> Tuple[int, Dict[str, int]]:
    """Return total detections and counts per class."""
    total = 0
    by_class: Dict[str, int] = {}
    for _, dets in results_by_frame.items():
        total += len(dets)
        for d in dets:
            c = d.get("class", "unknown")
            by_class[c] = by_class.get(c, 0) + 1
    return total, by_class


def save_detection_results(
    results_by_frame: Dict[str, List[Dict]],
    output_json_path: str,
    video_id: str,
    conf_threshold: float,
) -> None:
    """Write a single JSON file containing all detections for the video."""
    payload: Dict[str, Any] = {
        "video_id": video_id,
        "confidence_threshold": conf_threshold,
        "frames": [
            {"frame": frame, "detections": dets}
            for frame, dets in sorted(results_by_frame.items())
        ],
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def write_metadata(
    metadata_path: str,
    video_id: str,
    source: str,
    total_frames: int,
    total_detections: int,
    by_class: Dict[str, int],
    model_name: str,
    device: str,
    conf_threshold: float,
) -> None:
    lines = [
        f"video_id: {video_id}",
        f"source: {source}",
        f"confidence_threshold: {conf_threshold}",
        f"total_frames: {total_frames}",
        f"total_detections: {total_detections}",
        f"model: {model_name}",
        f"device: {device}",
        "class_counts:",
    ]
    for k in sorted(by_class.keys()):
        lines.append(f"  {k}: {by_class[k]}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")