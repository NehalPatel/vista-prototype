#!/usr/bin/env python3
"""Build face recognition and monument classification models from training_data (CLI, no web).

Reads from:
  vista-prototype/training_data/faces/<name>/   -> face embeddings in known_faces/
  vista-prototype/training_data/monuments/<name>/ + training_data/dataset/ -> monument classifier in monument_model/

Run from repo root:
  python scripts/build_models.py              # build both
  python scripts/build_models.py --faces-only
  python scripts/build_models.py --monuments-only
"""

from __future__ import annotations

import argparse
import os
import sys

# Run from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.paths import (
    TRAINING_FACES_DIR,
    TRAINING_MONUMENTS_DIR,
    TRAINING_DATASET_DIR,
    MONUMENT_MODEL_DIR,
)
from face_pipeline.paths import KNOWN_FACES_DIR


def build_face_model(device: str = "cpu", face_model: str = "buffalo_l") -> tuple[bool, str]:
    """Register all faces from training_data/faces/<name>/ into known_faces/. Rebuilds from scratch each run."""
    if not os.path.isdir(TRAINING_FACES_DIR):
        return False, "training_data/faces/ not found"
    try:
        from face_pipeline.register_known import register_faces_from_folder
    except Exception as e:
        return False, f"face_pipeline import failed: {e}"

    # Start fresh: clear existing embeddings and labels so this run = full rebuild from training_data/faces
    emb_dir = os.path.join(str(KNOWN_FACES_DIR), "embeddings")
    labels_path = os.path.join(str(KNOWN_FACES_DIR), "labels.json")
    os.makedirs(emb_dir, exist_ok=True)
    for f in os.listdir(emb_dir) if os.path.isdir(emb_dir) else []:
        if f.lower().endswith(".npy"):
            try:
                os.remove(os.path.join(emb_dir, f))
            except Exception:
                pass
    if os.path.isfile(labels_path):
        try:
            os.remove(labels_path)
        except Exception:
            pass
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("{}")

    total = 0
    errors = []
    for name in sorted(os.listdir(TRAINING_FACES_DIR)):
        path = os.path.join(TRAINING_FACES_DIR, name)
        if not os.path.isdir(path):
            continue
        count, err = register_faces_from_folder(
            path, name, device=device, model_name=face_model, conf_thresh=0.8
        )
        if err:
            errors.append(f"{name}: {err}")
        else:
            total += count
    if errors:
        return True, f"Registered {total} faces. Warnings: {'; '.join(errors)}"
    return True, f"Registered {total} faces."


def build_monument_model(device: str = "cpu") -> tuple[bool, str]:
    """Train monument classifier from training_data/monuments/ and training_data/dataset/."""
    try:
        from pipeline.monuments import build_and_train_monument_model
    except Exception as e:
        return False, f"pipeline.monuments import failed: {e}"

    result = build_and_train_monument_model(
        dataset_dir=TRAINING_DATASET_DIR,
        monuments_dir=TRAINING_MONUMENTS_DIR,
        model_dir=MONUMENT_MODEL_DIR,
        device=device,
    )
    if result.get("trained"):
        n = result.get("n_samples", 0)
        c = result.get("n_classes", 0)
        names = result.get("class_names", [])
        return True, f"Monument model built: {c} classes, {n} samples. Classes: {', '.join(names)}"
    return False, result.get("error", "Training failed")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build face and/or monument models from training_data (CLI)."
    )
    parser.add_argument("--faces-only", action="store_true", help="Only build face recognition model")
    parser.add_argument("--monuments-only", action="store_true", help="Only build monument classifier")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Device (default: cuda if available)")
    parser.add_argument("--face-model", default="buffalo_l", choices=["buffalo_l", "buffalo_s", "buffalo_sc"], help="InsightFace model for faces")
    args = parser.parse_args()

    device = args.device
    if device is None:
        try:
            import torch  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"Using device: {device}")

    do_faces = args.faces_only or (not args.monuments_only)
    do_monuments = args.monuments_only or (not args.faces_only)

    if do_faces:
        print("Building face model from training_data/faces/ ...")
        ok, msg = build_face_model(device=device, face_model=args.face_model)
        if ok:
            print("Faces:", msg)
        else:
            print("Faces failed:", msg)
            if do_monuments:
                print("Continuing with monuments...")

    if do_monuments:
        print("Building monument model from training_data/monuments/ and training_data/dataset/ ...")
        ok, msg = build_monument_model(device=device)
        if ok:
            print("Monuments:", msg)
        else:
            print("Monuments failed:", msg)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
