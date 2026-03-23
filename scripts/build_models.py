#!/usr/bin/env python3
"""Build face recognition and monument classification models from training_data (CLI, no web).

Reads from:
  vista-prototype/training_data/faces/<name>/   -> face embeddings in known_faces/
  vista-prototype/training_data/monuments/<name>/ + training_data/dataset/ -> monument classifier in monument_model/

Run from repo root:
  python scripts/build_models.py              # build both (incremental)
  python scripts/build_models.py --full       # from-scratch rebuild for both
  python scripts/build_models.py --faces-only
  python scripts/build_models.py --monuments-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Run from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Load .env from repo root so MONGODB_URI is set when running this script (e.g. python scripts/build_models.py)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except Exception:
    pass

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
from pipeline.utils import canonical_display_name

BUILD_STATE_FILENAME = "build_state.json"
FACE_DB_FILENAME = "face_database.npy"


def _rel_path(path: str, base: str) -> str:
    """Return path relative to base, normalized (forward slashes for portability)."""
    p = os.path.normpath(os.path.relpath(os.path.abspath(path), os.path.abspath(base)))
    return p.replace("\\", "/")


def _migrate_json_state_to_mongo(state_path: str, training_faces_dir: str) -> None:
    """If state_path (JSON) exists and has data, load it and write to MongoDB using relative paths."""
    if not os.path.isfile(state_path):
        return
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        return
    if not state:
        return
    try:
        from pipeline.mongodb_store import save_person_face_state
    except Exception:
        return
    base = os.path.abspath(training_faces_dir)
    for person, mapping in state.items():
        state_person = {}
        for abs_path, emb in mapping.items():
            try:
                rel = _rel_path(abs_path, base)
                state_person[rel] = emb
            except Exception:
                continue
        save_person_face_state(person, state_person)

# #region agent log
def _debug_log(message: str, data: dict, hypothesis_id: str = "") -> None:
    import time
    log_path = os.path.join(REPO_ROOT, "debug-b6867e.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": hypothesis_id,
                        "timestamp": int(time.time() * 1000),
                        "location": "build_models.py",
                        "message": message,
                        "data": data,
                    }
                )
                + "\n"
            )
    except Exception:
        pass
# #endregion


def _embedding_index(emb_filename: str) -> int:
    """Parse index from embedding filename (label_base_IDX.npy). Last numeric segment before .npy."""
    try:
        base = emb_filename[:-4]  # drop .npy
        return int(base.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def _load_face_database(known_faces_dir: str) -> dict:
    """Load face_database.npy if it exists. Returns {person: {"mean": vec, "count": n}}.
    Converts legacy format (person -> vec) to mean+count.
    """
    db_path = os.path.join(known_faces_dir, FACE_DB_FILENAME)
    if not os.path.isfile(db_path):
        return {}
    try:
        data = np.load(db_path, allow_pickle=True).item()
        if not isinstance(data, dict):
            return {}
        out: dict = {}
        for person, val in data.items():
            try:
                if isinstance(val, dict) and "mean" in val:
                    out[str(person)] = {
                        "mean": np.asarray(val["mean"], dtype=np.float32),
                        "count": int(val.get("count", 1)),
                    }
                else:
                    out[str(person)] = {
                        "mean": np.asarray(val, dtype=np.float32),
                        "count": 1,
                    }
            except Exception:
                continue
        return out
    except Exception:
        return {}


def _save_face_database(known_faces_dir: str, db: dict) -> None:
    """Save face database to known_faces_dir/face_database.npy. db: {person: {"mean": vec, "count": n}}."""
    db_path = os.path.join(known_faces_dir, FACE_DB_FILENAME)
    np.save(db_path, db, allow_pickle=True)


def build_face_model(
    device: str = "cpu",
    face_model: str = "buffalo_l",
    from_scratch: bool = False,
) -> tuple[bool, str]:
    """Register faces from training_data/faces/<name>/ into known_faces/.

    If from_scratch is True, clears existing embeddings and rebuilds all.
    Otherwise does incremental build: only processes new images and removes embeddings for deleted images.
    """
    if not os.path.isdir(TRAINING_FACES_DIR):
        return False, "training_data/faces/ not found"
    _suppress_onnx_verbose()
    try:
        from face_pipeline.detection import load_detector
        from face_pipeline.register_known import find_images, get_embeddings_for_paths
    except Exception as e:
        return False, f"face_pipeline import failed: {e}"

    known_faces = str(KNOWN_FACES_DIR)
    training_faces_dir = TRAINING_FACES_DIR
    emb_dir = os.path.join(known_faces, "embeddings")
    state_path_json = os.path.join(known_faces, BUILD_STATE_FILENAME)
    os.makedirs(emb_dir, exist_ok=True)

    detector = load_detector(device=device, model_name=face_model, silent=True)

    try:
        from pipeline.mongodb_store import (
            load_face_build_state,
            save_person_face_state,
            remove_person_face_state,
            clear_face_build_state,
            get_db,
            ensure_indexes,
        )
    except Exception as e:
        return False, f"pipeline.mongodb_store import failed: {e}"

    if get_db() is not None:
        ensure_indexes()
    elif not from_scratch:
        print("  Warning: MongoDB not connected (set MONGODB_URI in .env). Incremental state will not be saved; every run will reprocess all images.", flush=True)

    if from_scratch:
        for f in os.listdir(emb_dir) if os.path.isdir(emb_dir) else []:
            if f.lower().endswith(".npy"):
                try:
                    os.remove(os.path.join(emb_dir, f))
                except Exception:
                    pass
        clear_face_build_state()
        state: dict = {}
        db: dict = {}
    else:
        state = load_face_build_state()
        if not state and os.path.isfile(state_path_json):
            _migrate_json_state_to_mongo(state_path_json, training_faces_dir)
            state = load_face_build_state()
        db = _load_face_database(known_faces)

    # #region agent log
    _debug_log(
        "state/db load",
        {
            "mongo_used": get_db() is not None,
            "len_state": len(state),
            "len_db": len(db),
            "known_faces_resolved": os.path.abspath(known_faces),
            "training_faces_dir_resolved": os.path.abspath(training_faces_dir),
        },
        "H1_H3",
    )
    # #endregion

    current_names = [
        n
        for n in sorted(os.listdir(TRAINING_FACES_DIR))
        if os.path.isdir(os.path.join(TRAINING_FACES_DIR, n))
    ]

    if not from_scratch:
        # Remove state and db entries for persons no longer in training_data/faces (no per-image .npy to delete)
        for name in list(state):
            if name not in current_names:
                del state[name]
                db.pop(name, None)
                db.pop(canonical_display_name(name), None)
                remove_person_face_state(name)

    n_total = len(current_names)
    total_new = 0
    errors = []

    for idx, name in enumerate(current_names, start=1):
        display_name = canonical_display_name(name)
        path = os.path.join(training_faces_dir, name)
        images = find_images(path)
        current_rel_paths = {_rel_path(im, training_faces_dir) for im in images}

        if from_scratch:
            print(f"  [{idx}/{n_total}] {name}...", flush=True)
            pairs = get_embeddings_for_paths(detector, images, conf_thresh=0.8)
            if not pairs:
                errors.append(f"{name}: no faces detected")
                state[name] = {}
            else:
                vecs = np.stack([p[1] for p in pairs], axis=0)
                mean_vec = np.mean(vecs, axis=0).astype(np.float32)
                db[display_name] = {"mean": mean_vec, "count": len(pairs)}
                if display_name != name:
                    db.pop(name, None)
                state[name] = {_rel_path(p[0], training_faces_dir): "" for p in pairs}
                total_new += len(pairs)
            save_person_face_state(name, state[name])
            continue

        state_person = state.setdefault(name, {})
        new_paths = [p for p in images if _rel_path(p, training_faces_dir) not in state_person]
        removed_rel_paths = [r for r in state_person if r not in current_rel_paths]

        # #region agent log
        if idx == 1:
            first_im = images[0] if images else None
            rel_first = _rel_path(first_im, training_faces_dir) if first_im else None
            state_keys_sample = list(state_person.keys())[:2]
            _debug_log(
                "first person incremental",
                {
                    "name": name,
                    "first_image_path": first_im,
                    "rel_first_image": rel_first,
                    "state_person_keys_sample": state_keys_sample,
                    "rel_first_in_state": rel_first in state_person if rel_first else False,
                    "len_state_person": len(state_person),
                    "len_images": len(images),
                    "len_new_paths": len(new_paths),
                },
                "H2",
            )
        # #endregion

        for rel in removed_rel_paths:
            state_person.pop(rel, None)

        if removed_rel_paths:
            # Re-embed all remaining images for this person to recompute mean (no per-image .npy stored)
            remaining = [im for im in images if _rel_path(im, training_faces_dir) not in removed_rel_paths]
            if remaining:
                print(f"  [{idx}/{n_total}] {name} (re-embed {len(remaining)} after remove)...", flush=True)
                pairs = get_embeddings_for_paths(detector, remaining, conf_thresh=0.8)
                if pairs:
                    vecs = np.stack([p[1] for p in pairs], axis=0)
                    db[display_name] = {"mean": np.mean(vecs, axis=0).astype(np.float32), "count": len(pairs)}
                    if display_name != name:
                        db.pop(name, None)
                    state_person.clear()
                    for p in pairs:
                        state_person[_rel_path(p[0], training_faces_dir)] = ""
            else:
                db.pop(name, None)
                db.pop(display_name, None)
                state_person.clear()
        elif new_paths:
            print(f"  [{idx}/{n_total}] {name} (+{len(new_paths)} new)...", flush=True)
            pairs = get_embeddings_for_paths(detector, new_paths, conf_thresh=0.8)
            if not pairs:
                pass  # no new embeddings; state unchanged
            else:
                total_new += len(pairs)
                existing = db.get(display_name) or db.get(name)
                if existing is None:
                    vecs = np.stack([p[1] for p in pairs], axis=0)
                    db[display_name] = {"mean": np.mean(vecs, axis=0).astype(np.float32), "count": len(pairs)}
                else:
                    # Incremental mean: new_mean = (old_mean * n + sum(new)) / (n + len(new))
                    n_old = existing["count"]
                    mean_old = existing["mean"]
                    new_vecs = np.stack([p[1] for p in pairs], axis=0)
                    n_new = len(pairs)
                    new_mean = (mean_old * n_old + np.sum(new_vecs, axis=0)) / (n_old + n_new)
                    db[display_name] = {"mean": new_mean.astype(np.float32), "count": n_old + n_new}
                if display_name != name:
                    db.pop(name, None)
                for p in pairs:
                    state_person[_rel_path(p[0], training_faces_dir)] = ""
        else:
            print(f"  [{idx}/{n_total}] {name} (no new images)", flush=True)

        save_person_face_state(name, state_person)

    # Save single face_database.npy (one mean per person; no per-image .npy in known_faces).
    _save_face_database(known_faces, db)
    n_persons = len(db)
    print(f"Face DB: Saved face_database.npy for {n_persons} person(s)")

    # #region agent log
    _debug_log(
        "state save",
        {
            "mongo_used": get_db() is not None,
            "len_state": len(state),
            "first_person_state_len": len(state.get(current_names[0], {})) if current_names else 0,
        },
        "H4",
    )
    # #endregion

    n_persons = len(db)
    if errors:
        return True, f"Registered {total_new} new faces ({n_persons} persons in face_database.npy). Warnings: {'; '.join(errors)}"
    return True, f"Registered {total_new} new faces ({n_persons} persons in face_database.npy)."


def build_monument_model(
    device: str = "cpu", clear_feature_cache: bool = False
) -> tuple[bool, str]:
    """Train monument classifier from training_data/monuments/ and training_data/dataset/.

    If clear_feature_cache is True, discards cached ResNet18 features and extracts all again.
    """
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
        clear_feature_cache=clear_feature_cache,
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
        "--full",
        action="store_true",
        help="From-scratch rebuild (clear existing face embeddings and monument cache before building)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove known_faces outputs (face_database.npy, embeddings/*.npy, build_state.json) and MongoDB face state before building. Use with --full for a clean full build.",
    )
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

    if args.clean:
        known_faces = str(KNOWN_FACES_DIR)
        emb_dir = os.path.join(known_faces, "embeddings")
        for f in os.listdir(emb_dir) if os.path.isdir(emb_dir) else []:
            if f.lower().endswith(".npy"):
                try:
                    os.remove(os.path.join(emb_dir, f))
                except Exception:
                    pass
        for fname in [FACE_DB_FILENAME, BUILD_STATE_FILENAME, "labels.json"]:
            p = os.path.join(known_faces, fname)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        try:
            from pipeline.mongodb_store import clear_face_build_state
            clear_face_build_state()
            print("Cleaned known_faces and face build state.")
        except Exception:
            print("Cleaned known_faces (MongoDB clear skipped).")

    do_faces = args.faces_only or (not args.monuments_only)
    do_monuments = args.monuments_only or (not args.faces_only)

    if do_faces:
        mode = "from scratch" if args.full else "incremental"
        print(f"Building face model from training_data/faces/ ({mode})...")
        ok, msg = build_face_model(
            device=face_device, face_model=args.face_model, from_scratch=args.full
        )
        if ok:
            print("Faces:", msg)
        else:
            print("Faces failed:", msg)
            if do_monuments:
                print("Continuing with monuments...")

    if do_monuments:
        mode = "from scratch" if args.full else "incremental (using feature cache)"
        print(f"Building monument model from training_data/monuments/ and training_data/dataset/ ({mode})...")
        ok, msg = build_monument_model(
            device=monument_device, clear_feature_cache=args.full
        )
        if ok:
            print("Monuments:", msg)
        else:
            print("Monuments failed:", msg)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
