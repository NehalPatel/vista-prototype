import argparse
import os
import sys
import time
import json
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import project modules
from pipeline.paths import (
    ensure_directories,
    FRAMES_DIR as OBJ_FRAMES_DIR,
    VIDEOS_DIR,
    get_video_results_paths,
    ensure_video_results_dirs,
)
from pipeline.utils import (
    extract_video_id_from_url,
    sanitize_id,
    validate_video_id,
    HAS_TORCH,
)
from pipeline.video import download_video, extract_frames
from pipeline.render import make_video_from_images

from face_pipeline.detection import load_detector as load_face_detector, detect_faces
from ultralytics import YOLO  # type: ignore


def _yolo_init(model_path: str = "yolov8n.pt"):
    model = YOLO(model_path)
    try:
        if HAS_TORCH:
            import torch  # type: ignore
            if torch.cuda.is_available():
                model.to("cuda")
    except Exception:
        pass
    return model


def _run_yolo_on_image(model, img_bgr: np.ndarray, conf_threshold: float) -> List[Dict[str, Any]]:
    # Ultralytics YOLO accepts numpy arrays (BGR)
    result = model(img_bgr)
    boxes = result[0].boxes
    names = result[0].names
    detections: List[Dict[str, Any]] = []
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
    return detections


def _draw_overlay(img_bgr: np.ndarray, faces: List[Dict[str, Any]], objects: List[Dict[str, Any]]) -> np.ndarray:
    out = img_bgr.copy()
    # Draw objects (red)
    for o in objects:
        x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
        label = f"{o['class']} {o['conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # Draw faces (cyan)
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
        label = f"face {f['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return out


def _list_frames(frames_dir: str) -> List[str]:
    names = [n for n in sorted(os.listdir(frames_dir)) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    return [os.path.join(frames_dir, n) for n in names]


def run_parallel_pipeline(
    frames_dir: str,
    output_base: str,
    yolo_conf: float = 0.7,
    face_conf: float = 0.8,
    face_device: str = "cuda",
    fps: int = 1,
    video_id: str = "",
) -> Dict[str, Any]:
    os.makedirs(output_base, exist_ok=True)
    combined_frames_dir = os.path.join(output_base, "combined_frames")
    os.makedirs(combined_frames_dir, exist_ok=True)

    # Init models
    yolo_model = _yolo_init("yolov8n.pt")
    face_detector = load_face_detector(device=face_device)

    frames = _list_frames(frames_dir)
    if not frames:
        raise RuntimeError(f"No frames found in {frames_dir}")

    combined_json = {"video_id": video_id, "yolo_conf": yolo_conf, "face_conf": face_conf, "frames": []}
    metrics = {"per_frame": [], "summary": {"avg_yolo_ms": 0.0, "avg_face_ms": 0.0, "avg_total_ms": 0.0}}

    # Use a thread pool to process frames concurrently per detector
    with ThreadPoolExecutor(max_workers=4) as executor:
        for frame_path in frames:
            img = cv2.imread(frame_path)
            if img is None:
                combined_json["frames"].append({
                    "frame": os.path.basename(frame_path),
                    "error": "failed_to_load_frame"
                })
                continue

            # Schedule both detectors
            t0 = time.time()
            yolo_future = executor.submit(_run_yolo_on_image, yolo_model, img, yolo_conf)
            face_future = executor.submit(detect_faces, face_detector, img, face_conf)

            # Collect results
            yolo_start = time.time()
            try:
                yolo_objects = yolo_future.result()
            except Exception as e:
                yolo_objects = []
                # mark an error but continue
                # we still compute timing as now
            yolo_ms = (time.time() - yolo_start) * 1000.0

            face_start = time.time()
            try:
                faces = face_future.result()
            except Exception as e:
                faces = []
            face_ms = (time.time() - face_start) * 1000.0

            total_ms = (time.time() - t0) * 1000.0

            # Render overlay
            overlay = _draw_overlay(img, faces, yolo_objects)
            out_img_path = os.path.join(combined_frames_dir, os.path.basename(frame_path))
            cv2.imwrite(out_img_path, overlay)

            # Record JSON
            combined_json["frames"].append({
                "frame": os.path.basename(frame_path),
                "faces": [{"bbox": f["bbox"], "confidence": f["confidence"]} for f in faces],
                "objects": yolo_objects,
                "timings_ms": {"yolo": yolo_ms, "face": face_ms, "total": total_ms},
            })
            metrics["per_frame"].append({"frame": os.path.basename(frame_path), "yolo_ms": yolo_ms, "face_ms": face_ms, "total_ms": total_ms})

    # Compute summary metrics
    if metrics["per_frame"]:
        ay = sum(m["yolo_ms"] for m in metrics["per_frame"]) / len(metrics["per_frame"])
        af = sum(m["face_ms"] for m in metrics["per_frame"]) / len(metrics["per_frame"])
        at = sum(m["total_ms"] for m in metrics["per_frame"]) / len(metrics["per_frame"])
        metrics["summary"].update({"avg_yolo_ms": ay, "avg_face_ms": af, "avg_total_ms": at})

    # Write JSON outputs
    combined_json_path = os.path.join(output_base, "combined_results.json")
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_json, f, indent=2)

    metrics_path = os.path.join(output_base, "combined_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Make a combined video
    out_video_path = os.path.join(output_base, "combined_detections_video.mp4")
    make_video_from_images(combined_frames_dir, out_video_path, fps=fps)

    return {
        "combined_json": combined_json_path,
        "metrics_json": metrics_path,
        "combined_frames_dir": combined_frames_dir,
        "combined_video": out_video_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel face + object detection on video frames with unified overlay")
    parser.add_argument("--url", type=str, default=None, help="YouTube URL to download and process")
    parser.add_argument("--video", type=str, default=None, help="Local video file path to process")
    parser.add_argument("--fps", type=int, default=1, help="FPS for combined output video")
    parser.add_argument("--yolo-conf", type=float, default=0.7, help="Confidence threshold for YOLO objects")
    parser.add_argument("--face-conf", type=float, default=0.8, help="Confidence threshold for face detections")
    parser.add_argument("--face-device", choices=["cuda", "cpu"], default="cuda", help="Device for InsightFace detector")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directories()

    # Resolve source
    if args.video:
        video_path = os.path.abspath(args.video)
        base = os.path.splitext(os.path.basename(video_path))[0]
        video_id = sanitize_id(base)
        source_desc = f"local:{video_path}"
    elif args.url:
        video_path = download_video(args.url, VIDEOS_DIR)
        if not video_path:
            print("Error: Video download failed.", file=sys.stderr)
            sys.exit(1)
        video_id = extract_video_id_from_url(args.url) or sanitize_id(os.path.splitext(os.path.basename(video_path))[0])
        source_desc = args.url
    else:
        print("Error: Provide either --url or --video.", file=sys.stderr)
        sys.exit(1)

    if not validate_video_id(video_id or ""):
        print("Error: Invalid video ID.", file=sys.stderr)
        sys.exit(1)

    paths = get_video_results_paths(video_id)
    if not ensure_video_results_dirs(video_id):
        print("Error: Failed to create per-video results directories.", file=sys.stderr)
        sys.exit(1)

    # Prevent overwrite of base processed_frames (not strictly required for combined but keep consistent)
    if os.path.exists(paths["detection_json"]) or (
        os.path.isdir(paths["processed_frames"]) and any(os.scandir(paths["processed_frames"]))
    ):
        print(
            f"Warning: Existing object detection results found for video_id '{video_id}'. The combined overlay will write separate outputs.",
            file=sys.stderr,
        )

    # Extract frames (always overwrite frames for fresh run)
    extract_frames(video_path, OBJ_FRAMES_DIR)

    # Run parallel fusion and write outputs under results/{video_id}/
    fusion_outputs = run_parallel_pipeline(
        frames_dir=OBJ_FRAMES_DIR,
        output_base=paths["base"],
        yolo_conf=args.yolo_conf,
        face_conf=args.face_conf,
        face_device=args.face_device,
        fps=args.fps,
        video_id=video_id,
    )

    # Write a small metadata entry for fusion
    meta_path = os.path.join(paths["base"], "combined_metadata.txt")
    try:
        device = "cpu"
        if HAS_TORCH:
            import torch  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"video_id: {video_id}\n")
        f.write(f"source: {source_desc}\n")
        f.write(f"yolo_conf_threshold: {args.yolo_conf}\n")
        f.write(f"face_conf_threshold: {args.face_conf}\n")
        f.write(f"device: {device}\n")
        f.write(f"combined_results_json: {os.path.relpath(fusion_outputs['combined_json'], paths['base'])}\n")
        f.write(f"combined_metrics_json: {os.path.relpath(fusion_outputs['metrics_json'], paths['base'])}\n")
        f.write(f"combined_video: {os.path.relpath(fusion_outputs['combined_video'], paths['base'])}\n")

    print(f"Combined fusion complete. See results in '{paths['base']}'.")


if __name__ == "__main__":
    main()