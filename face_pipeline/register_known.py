import argparse
import os
import json
from glob import glob
from typing import List

import cv2
import numpy as np

from .detection import load_detector, detect_faces, FACE_MODEL_CHOICES
from .embeddings import get_embedding, save_embedding
from .paths import KNOWN_FACES_DIR


def find_images(folder: str) -> List[str]:
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob(os.path.join(folder, p), recursive=True))
    return sorted(files)


def register_faces_from_folder(
    images_dir: str,
    label: str,
    device: str = "cpu",
    model_name: str = "buffalo_l",
    conf_thresh: float = 0.8,
    embeddings_dir: str | None = None,
    labels_path: str | None = None,
    silent: bool = False,
) -> tuple[int, str]:
    """Register all faces from a directory under a single label (e.g. celebrity name).

    Loads existing labels.json if present and merges new embeddings. Returns (count, error_message).
    If count >= 0 and error_message is empty, success; else error_message describes the failure.
    silent: if True, suppress InsightFace/ONNX verbose output when loading the detector.
    """
    emb_dir = embeddings_dir or os.path.join(str(KNOWN_FACES_DIR), "embeddings")
    labels_out = labels_path or os.path.join(str(KNOWN_FACES_DIR), "labels.json")
    os.makedirs(emb_dir, exist_ok=True)

    labels: dict = {}
    if os.path.exists(labels_out):
        try:
            with open(labels_out, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            labels = {}

    detector = load_detector(device=device, model_name=model_name, silent=silent)
    images = find_images(images_dir)
    if not images:
        return 0, "No images found in directory"

    count = 0
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        dets = detect_faces(detector, img, conf_thresh=conf_thresh)
        if not dets:
            continue
        dets.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        emb = get_embedding(dets[0].get("face_obj"))
        if emb is None:
            continue
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{label}_{base}_{idx}.npy"
        out_path = os.path.join(emb_dir, out_name)
        save_embedding(out_path, emb)
        labels[out_name] = label
        count += 1

    try:
        with open(labels_out, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
    except Exception as e:
        return count, str(e)
    return count, ""


def main():
    parser = argparse.ArgumentParser(description="Register known faces by computing embeddings from images")
    parser.add_argument("--images-dir", required=True, help="Directory containing face images (one person per image or folder)")
    parser.add_argument("--labels-out", default=os.path.join(str(KNOWN_FACES_DIR), "labels.json"), help="Path to labels.json output")
    parser.add_argument("--embeddings-dir", default=os.path.join(str(KNOWN_FACES_DIR), "embeddings"), help="Output directory for embeddings")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for InsightFace")
    parser.add_argument("--model", choices=list(FACE_MODEL_CHOICES), default="buffalo_l", help="Face model: buffalo_l, buffalo_s, buffalo_sc")
    parser.add_argument("--conf", type=float, default=0.8, help="Face detection confidence threshold")
    args = parser.parse_args()

    os.makedirs(args.embeddings_dir, exist_ok=True)
    labels = {}

    # Initialize detector
    detector = load_detector(device=args.device, model_name=args.model)

    images = find_images(args.images_dir)
    if not images:
        print(f"No images found in {args.images_dir}")
        return 1

    count = 0
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: failed to load {img_path}")
            continue
        dets = detect_faces(detector, img, conf_thresh=args.conf)
        if not dets:
            print(f"Warning: no face detected in {img_path}")
            continue
        # Use the highest confidence detection
        dets.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        emb = get_embedding(dets[0].get("face_obj"))
        if emb is None:
            print(f"Warning: no embedding for {img_path}")
            continue
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{base}.npy"
        out_path = os.path.join(args.embeddings_dir, out_name)
        save_embedding(out_path, emb)
        labels[out_name] = base
        count += 1

    # Write labels.json
    try:
        with open(args.labels_out, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        print(f"Registered {count} known faces. Embeddings: {args.embeddings_dir}")
    except Exception as e:
        print(f"Failed to write labels.json: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())