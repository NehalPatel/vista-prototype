#!/usr/bin/env python3
"""Option B: Build face model with cropped-face fallback for small images (~4KB).

Duplicate of build_models.py for testing: when an image is small (max side < 256px),
we upscale it and run detection so cropped-face datasets can still get embeddings.
Same usage as build_models.py; writes to the same known_faces/ and MongoDB.

Run from repo root:
  python scripts/build_models_cropped_faces.py
  python scripts/build_models_cropped_faces.py --full
  python scripts/build_models_cropped_faces.py --faces-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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

import cv2
from face_pipeline.embed_cropped_face import (
    CROPPED_FACE_MAX_SIZE,
    get_embedding_for_small_crop,
)
from face_pipeline.detection import detect_faces, load_detector
from face_pipeline.embeddings import get_embedding, save_embedding

BUILD_STATE_FILENAME = "build_state.json"


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
                        "location": "build_models_cropped_faces.py",
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


def _get_embedding_for_image(detector, img, conf_thresh: float = 0.8):
    """Option B: For small images use upscale+detect; else normal detect. Returns embedding array or None."""
    h, w = img.shape[:2]
    if max(h, w) < CROPPED_FACE_MAX_SIZE:
        emb = get_embedding_for_small_crop(detector, img, conf_thresh=conf_thresh)
        if emb is not None:
            return emb
    dets = detect_faces(detector, img, conf_thresh=conf_thresh)
    if not dets:
        return None
    dets.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
    return get_embedding(dets[0].get("face_obj"))


def _register_faces_from_folder_cropped(
    images_dir: str,
    label: str,
    device: str,
    model_name: str,
    embeddings_dir: str,
    labels_path: str,
    labels: dict,
    silent: bool,
    conf_thresh: float = 0.8,
):
    """Like register_faces_from_folder but uses Option B: small-crop upscale fallback."""
    from face_pipeline.register_known import find_images
    detector = load_detector(device=device, model_name=model_name, silent=silent)
    images = find_images(images_dir)
    if not images:
        return 0, ""
    count = 0
    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        emb = _get_embedding_for_image(detector, img, conf_thresh=conf_thresh)
        if emb is None:
            continue
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{label}_{base}_{idx}.npy"
        out_path = os.path.join(embeddings_dir, out_name)
        save_embedding(out_path, emb)
        labels[out_name] = label
        count += 1
    try:
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
    except Exception as e:
        return count, str(e)
    return count, ""


def _register_faces_from_paths_cropped(
    image_paths: list,
    label: str,
    device: str,
    model_name: str,
    embeddings_dir: str,
    labels_path: str,
    labels: dict,
    silent: bool,
    conf_thresh: float = 0.8,
):
    """Like register_faces_from_paths but uses Option B: small-crop upscale fallback. Returns (count, err, path_to_emb)."""
    path_to_embedding = {}
    count = 0
    detector = load_detector(device=device, model_name=model_name, silent=silent)
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        emb = _get_embedding_for_image(detector, img, conf_thresh=conf_thresh)
        if emb is None:
            continue
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{label}_{base}_{idx}.npy"
        out_path = os.path.join(embeddings_dir, out_name)
        save_embedding(out_path, emb)
        labels[out_name] = label
        count += 1
        path_to_embedding[os.path.normpath(os.path.abspath(img_path))] = out_name
    try:
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
    except Exception as e:
        return count, str(e), path_to_embedding
    return count, "", path_to_embedding


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
        from face_pipeline.register_known import find_images
    except Exception as e:
        return False, f"face_pipeline import failed: {e}"

    known_faces = str(KNOWN_FACES_DIR)
    training_faces_dir = TRAINING_FACES_DIR
    emb_dir = os.path.join(known_faces, "embeddings")
    labels_path = os.path.join(known_faces, "labels.json")
    state_path_json = os.path.join(known_faces, BUILD_STATE_FILENAME)
    os.makedirs(emb_dir, exist_ok=True)

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
        if os.path.isfile(labels_path):
            try:
                os.remove(labels_path)
            except Exception:
                pass
        with open(labels_path, "w", encoding="utf-8") as f:
            f.write("{}")
        clear_face_build_state()
        state: dict = {}
    else:
        state = load_face_build_state()
        if not state and os.path.isfile(state_path_json):
            _migrate_json_state_to_mongo(state_path_json, training_faces_dir)
            state = load_face_build_state()
        if not os.path.isfile(labels_path):
            with open(labels_path, "w", encoding="utf-8") as f:
                f.write("{}")

    labels: dict = {}
    if os.path.isfile(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            labels = {}

    # #region agent log
    _debug_log(
        "state/labels load",
        {
            "mongo_used": get_db() is not None,
            "len_state": len(state),
            "len_labels": len(labels),
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

    if not from_scratch and not state and labels:
        # Bootstrap state from existing labels + folder order (e.g. first run after adding incremental, or state was deleted)
        print("  Bootstrapping state from existing embeddings...", flush=True)
        # #region agent log
        _debug_log("bootstrap starting", {"current_names_count": len(current_names)}, "H5")
        # #endregion
        for name in current_names:
            path = os.path.join(training_faces_dir, name)
            images = find_images(path)
            emb_files = sorted(
                (f for f, label in labels.items() if label == name),
                key=_embedding_index,
            )
            state[name] = {}
            for i in range(min(len(images), len(emb_files))):
                state[name][_rel_path(images[i], training_faces_dir)] = emb_files[i]
        for name in state:
            save_person_face_state(name, state[name])
        # #region agent log
        first_name = current_names[0] if current_names else None
        first_state_keys = list(state.get(first_name, {}).keys())[:2] if first_name else []
        _debug_log(
            "bootstrap done",
            {
                "first_person": first_name,
                "len_state_keys_first": len(state.get(first_name, {})),
                "first_state_key_sample": first_state_keys[0] if first_state_keys else None,
                "len_state_after": len(state),
            },
            "H5",
        )
        # #endregion

    if not from_scratch:
        # Remove state and embeddings for persons no longer in training_data/faces
        for name in list(state):
            if name not in current_names:
                for rel_path, emb_file in state[name].items():
                    p = os.path.join(emb_dir, emb_file)
                    if os.path.isfile(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                    labels.pop(emb_file, None)
                del state[name]
                remove_person_face_state(name)

    n_total = len(current_names)
    total_new = 0
    errors = []

    for idx, name in enumerate(current_names, start=1):
        path = os.path.join(training_faces_dir, name)
        images = find_images(path)
        current_rel_paths = {_rel_path(im, training_faces_dir) for im in images}

        if from_scratch:
            print(f"  [{idx}/{n_total}] {name}...", flush=True)
            count, err = _register_faces_from_folder_cropped(
                path, name, device, face_model, emb_dir, labels_path, labels, True, 0.8
            )
            if err:
                errors.append(f"{name}: {err}")
            else:
                total_new += count
            state[name] = {}
            for i, im in enumerate(images):
                rel = _rel_path(im, training_faces_dir)
                base = os.path.splitext(os.path.basename(im))[0]
                state[name][rel] = f"{name}_{base}_{i}.npy"
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
            emb_file = state_person.pop(rel, None)
            if emb_file:  # only remove .npy and labels when we had an embedding (skip "" = tried but no face)
                p = os.path.join(emb_dir, emb_file)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                labels.pop(emb_file, None)

        if new_paths:
            print(f"  [{idx}/{n_total}] {name} (+{len(new_paths)} new)...", flush=True)
            count, err, path_to_emb = _register_faces_from_paths_cropped(
                new_paths, name, device, face_model, emb_dir, labels_path, labels, True, 0.8
            )
            if err:
                errors.append(f"{name}: {err}")
            else:
                total_new += count
            # Record every attempted image in state so we don't retry failed (no face detected) images every run.
            # Use "" for images that had no embedding (small/low-confidence); they stay in state so we skip them next time.
            for p in new_paths:
                rel = _rel_path(p, training_faces_dir)
                abs_norm = os.path.normpath(os.path.abspath(p))
                emb_file = path_to_emb.get(abs_norm, "") or ""
                state_person[rel] = emb_file
                if emb_file:
                    labels[emb_file] = name
        else:
            print(f"  [{idx}/{n_total}] {name} (no new images)", flush=True)

        save_person_face_state(name, state_person)

    if not from_scratch:
        try:
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, indent=2)
        except Exception as e:
            errors.append(f"labels save: {e}")

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

    n_total_emb = len([f for f in os.listdir(emb_dir) if f.lower().endswith(".npy")]) if os.path.isdir(emb_dir) else 0
    if errors:
        return True, f"Registered {total_new} new faces ({n_total_emb} total). Warnings: {'; '.join(errors)}"
    return True, f"Registered {total_new} new faces ({n_total_emb} total)."


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
        mode = "from scratch" if args.full else "incremental"
        print(f"Building face model from training_data/faces/ (Option B: cropped-face fallback, {mode})...")
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
