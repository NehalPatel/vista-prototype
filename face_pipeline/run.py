import argparse
import os
import json
from glob import glob
from typing import Dict, Any

import cv2
import numpy as np

from .paths import (
    ensure_dirs,
    FRAMES_DIR,
    CROPS_DIR,
    EMBED_DIR,
    FACE_RESULTS_DIR,
    KNOWN_FACES_DIR,
)
from .detection import load_detector, detect_faces, crop_face, save_image, FACE_MODEL_CHOICES
from .embeddings import get_embedding, save_embedding
from .recognition import load_known_embeddings, match


def find_frames(frames_dir: str) -> list:
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    files = []
    for p in patterns:
        files.extend(glob(os.path.join(frames_dir, p), recursive=True))
    files = sorted(files)
    return files


def main():
    parser = argparse.ArgumentParser(description="Face pipeline: detection, alignment (implicit), embeddings, optional recognition")
    parser.add_argument("--frames-dir", default=str(FRAMES_DIR), help="Directory containing frames to process")
    parser.add_argument("--detect-conf", type=float, default=0.8, help="Minimum confidence for face detections")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for InsightFace")
    parser.add_argument("--model", choices=list(FACE_MODEL_CHOICES), default="buffalo_l", help="Face model: buffalo_l (best), buffalo_s, buffalo_sc")
    parser.add_argument("--force", action="store_true", help="Overwrite existing crops/embeddings")
    parser.add_argument("--do-recognition", action="store_true", help="Perform recognition against known faces")
    parser.add_argument("--known-faces-dir", default=str(KNOWN_FACES_DIR), help="Directory containing known face embeddings and labels.json")

    args = parser.parse_args()
    ensure_dirs()

    frames = find_frames(args.frames_dir)
    if not frames:
        print(f"No frames found in {args.frames_dir}")
        return 1

    # Initialize detector (with recognition model enabled)
    detector = load_detector(device=args.device, model_name=args.model)

    faces_json: Dict[str, Any] = {}

    for idx, frame_path in enumerate(frames, start=1):
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: failed to load frame {frame_path}")
            continue

        dets = detect_faces(detector, img, conf_thresh=args.detect_conf)
        frame_key = os.path.basename(frame_path)
        faces_json.setdefault(frame_key, [])

        for det_i, det in enumerate(dets, start=1):
            bbox = det["bbox"]
            crop = crop_face(img, bbox)
            crop_name = f"face_{idx:04d}_{det_i:02d}.jpg"
            crop_path = os.path.join(str(CROPS_DIR), crop_name)
            embed_name = f"face_{idx:04d}_{det_i:02d}.npy"
            embed_path = os.path.join(str(EMBED_DIR), embed_name)

            if not args.force and os.path.exists(crop_path) and os.path.exists(embed_path):
                # Already processed; record and continue
                record = {
                    "bbox": bbox,
                    "confidence": det["confidence"],
                    "aligned": True,  # InsightFace provides aligned embeddings
                    "face_crop": crop_path.replace("\\", "/"),
                    "embedding_file": embed_path.replace("\\", "/"),
                }
                if "landmarks" in det:
                    record["landmarks"] = det["landmarks"]
                faces_json[frame_key].append(record)
                continue

            save_image(crop, crop_path)

            emb = get_embedding(det["face_obj"])  # embedding from detector pipeline
            if emb is None:
                # As a fallback, we could re-run recognition module, but for now skip
                print(f"Warning: embedding missing for {frame_path} det #{det_i}")
                continue

            save_embedding(embed_path, emb)

            record = {
                "bbox": bbox,
                "confidence": det["confidence"],
                "aligned": True,
                "face_crop": crop_path.replace("\\", "/"),
                "embedding_file": embed_path.replace("\\", "/"),
            }
            if "landmarks" in det:
                record["landmarks"] = det["landmarks"]

            faces_json[frame_key].append(record)

    faces_json_path = os.path.join(str(FACE_RESULTS_DIR), "faces.json")
    with open(faces_json_path, "w", encoding="utf-8") as f:
        json.dump(faces_json, f, indent=2)
    print(f"Wrote {faces_json_path}")

    if args.do_recognition:
        known = load_known_embeddings(args.known_faces_dir)
        thresholds = {"same": 0.6, "maybe": 0.8}
        recog: Dict[str, Any] = {}
        for frame_key, entries in faces_json.items():
            recog.setdefault(frame_key, [])
            for entry in entries:
                try:
                    vec = np.load(entry["embedding_file"]).astype("float32")
                except Exception:
                    continue
                m = match(vec, known, thresholds)
                recog_entry = {
                    "face_crop": entry["face_crop"],
                    "embedding_file": entry["embedding_file"],
                    "recognized_as": m["label"],
                    "distance": m["distance"],
                    "confidence": m["confidence"],
                }
                recog[frame_key].append(recog_entry)

        recog_json_path = os.path.join(str(FACE_RESULTS_DIR), "recognition.json")
        with open(recog_json_path, "w", encoding="utf-8") as f:
            json.dump(recog, f, indent=2)
        print(f"Wrote {recog_json_path}")


if __name__ == "__main__":
    main()