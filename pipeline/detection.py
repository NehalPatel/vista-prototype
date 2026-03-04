"""Object detection using YOLOv8 and helpers to annotate frames.

This module provides:
- run_yolo: runs detection over frames and writes annotated images
- generate_summary: returns counts (including color+class labels)
- save_detection_results: writes a single JSON with all detections (with color attribute)
- write_metadata: writes a text metadata file
"""

from typing import Dict, List, Tuple, Any
import os
import json

import numpy as np
from ultralytics import YOLO  # type: ignore


def _get_dominant_color_name(crop_bgr: np.ndarray) -> str:
    """Infer dominant color name from a BGR crop (e.g. object region). Uses HSV."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    import cv2
    h, w = crop_bgr.shape[:2]
    if h < 2 or w < 2:
        return "unknown"
    # Use center 60% to avoid edges/shadows
    margin_h, margin_w = int(0.2 * h), int(0.2 * w)
    center = crop_bgr[margin_h : h - margin_h, margin_w : w - margin_w]
    if center.size == 0:
        center = crop_bgr
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    h_med = int(np.median(hsv[:, :, 0]))
    s_med = int(np.median(hsv[:, :, 1]))
    v_med = int(np.median(hsv[:, :, 2]))
    # Map to color name (OpenCV H 0-180, S/V 0-255)
    if v_med < 50:
        return "black"
    if s_med < 40:
        if v_med > 200:
            return "white"
        return "gray" if v_med > 90 else "dark gray"
    if v_med > 220 and s_med < 60:
        return "silver"
    if h_med <= 10 or h_med >= 170:
        return "red"
    if 11 <= h_med <= 25:
        return "orange"
    if 26 <= h_med <= 35:
        return "yellow"
    if 36 <= h_med <= 85:
        return "green"
    if 86 <= h_med <= 100:
        return "cyan"
    if 101 <= h_med <= 130:
        return "blue"
    if 131 <= h_med <= 150:
        return "purple"
    if 151 <= h_med <= 169:
        return "pink"
    return "other"


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
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            continue
        result = model(frame_path)
        detections: List[Dict] = []
        boxes = result[0].boxes
        names = result[0].names or {}
        if boxes is not None:
            h_img, w_img = frame_bgr.shape[:2]
            for i in range(len(boxes)):
                b = boxes[i]
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(round(z)) for z in xyxy]
                x1 = max(0, min(x1, w_img - 1))
                y1 = max(0, min(y1, h_img - 1))
                x2 = max(0, min(x2, w_img))
                y2 = max(0, min(y2, h_img))
                cls = int(b.cls.item())
                conf = float(b.conf.item())
                if conf >= conf_threshold:
                    class_name = names.get(cls, str(cls))
                    crop = frame_bgr[y1:y2, x1:x2]
                    color = _get_dominant_color_name(crop)
                    label = f"{color} {class_name}".strip()
                    detections.append({
                        "bbox": xyxy,
                        "class": class_name,
                        "color": color,
                        "label": label,
                        "conf": conf,
                    })
        results_by_frame[fname] = detections

        # Save annotated image (BGR numpy array)
        annotated = result[0].plot()
        out_path = os.path.join(detections_dir, fname)
        cv2.imwrite(out_path, annotated)

    return results_by_frame


def generate_summary(results_by_frame: Dict[str, List[Dict]]) -> Tuple[int, Dict[str, int]]:
    """Return total detections and counts per label (e.g. 'red car', 'blue bus')."""
    total = 0
    by_class: Dict[str, int] = {}
    for _, dets in results_by_frame.items():
        total += len(dets)
        for d in dets:
            label = d.get("label") or (d.get("color", "") + " " + d.get("class", "unknown")).strip() or "unknown"
            by_class[label] = by_class.get(label, 0) + 1
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