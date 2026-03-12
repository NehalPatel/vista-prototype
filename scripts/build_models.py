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

# Suppress ONNX Runtime verbose output (Applied providers, find model, etc.) during build
def _suppress_onnx_verbose() -> None:
    try:
        import onnxruntime as ort  # type: ignore
        if hasattr(ort, "set_default_logger_severity"):
            ort.set_default_logger_severity(3)  # 3 = Error only
    except Exception:
        pass

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
    _suppress_onnx_verbose()
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

    names = [n for n in sorted(os.listdir(TRAINING_FACES_DIR))
             if os.path.isdir(os.path.join(TRAINING_FACES_DIR, n))]
    n_total = len(names)
    total = 0
    errors = []
    for idx, name in enumerate(names, start=1):
        path = os.path.join(TRAINING_FACES_DIR, name)
        print(f"  [{idx}/{n_total}] {name}...", flush=True)
        count, err = register_faces_from_folder(
            path, name, device=device, model_name=face_model, conf_thresh=0.8, silent=True
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

    def _progress(msg: str) -> None:
        print(msg, flush=True)

    result = build_and_train_monument_model(
        dataset_dir=TRAINING_DATASET_DIR,
        monuments_dir=TRAINING_MONUMENTS_DIR,
        model_dir=MONUMENT_MODEL_DIR,
        device=device,
        progress_callback=_progress,
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
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Force device for both models (default: auto-detect GPU per backend)",
    )
    parser.add_argument("--face-model", default="buffalo_l", choices=["buffalo_l", "buffalo_s", "buffalo_sc"], help="InsightFace model for faces")
    args = parser.parse_args()

    # Auto-detect GPU per backend when not forced: faces use ONNX/CUDA, monuments use PyTorch/CUDA
    def _gpu_available_onnx() -> bool:
        try:
            import onnxruntime as ort  # type: ignore
            return "CUDAExecutionProvider" in getattr(ort, "get_available_providers", lambda: [])()
        except Exception:
            return False

    def _gpu_available_torch() -> bool:
        try:
            import torch  # type: ignore
            return torch.cuda.is_available()
        except Exception:
            return False

    if args.device is not None:
        face_device = monument_device = args.device
    else:
        face_device = "cuda" if _gpu_available_onnx() else "cpu"
        monument_device = "cuda" if _gpu_available_torch() else "cpu"
    print(f"Using device – faces: {face_device}, monuments: {monument_device}")

    do_faces = args.faces_only or (not args.monuments_only)
    do_monuments = args.monuments_only or (not args.faces_only)

    if do_faces:
        print("Building face model from training_data/faces/ ...")
        ok, msg = build_face_model(device=face_device, face_model=args.face_model)
        if ok:
            print("Faces:", msg)
        else:
            print("Faces failed:", msg)
            if do_monuments:
                print("Continuing with monuments...")

    if do_monuments:
        print("Building monument model from training_data/monuments/ and training_data/dataset/ ...")
        ok, msg = build_monument_model(device=monument_device)
        if ok:
            print("Monuments:", msg)
        else:
            print("Monuments failed:", msg)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
