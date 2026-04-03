from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

from pipeline.paths import (
    MONUMENT_MODEL_DIR,
    TRAINING_DATASET_DIR,
    TRAINING_FACES_DIR,
    TRAINING_MONUMENTS_DIR,
    ensure_directories,
)
from pipeline.utils import sanitize_dataset_name, sanitize_id


training_bp = Blueprint("training", __name__)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _training_upload_dir(dataset_type: str, name: str):
    safe_name = sanitize_dataset_name(name)
    if not safe_name:
        return None
    if dataset_type == "face":
        return os.path.join(TRAINING_FACES_DIR, safe_name)
    if dataset_type == "monument":
        return os.path.join(TRAINING_MONUMENTS_DIR, safe_name)
    return None


@training_bp.post("/api/training/upload")
def api_training_upload():
    ensure_directories()
    name = (request.form.get("name") or "").strip()
    dataset_type = (request.form.get("type") or "face").strip().lower()
    if dataset_type not in ("face", "monument"):
        return jsonify({"error": "type must be 'face' or 'monument'"}), 400
    target_dir = _training_upload_dir(dataset_type, name)
    if not target_dir:
        return jsonify({"error": "Invalid or empty name"}), 400

    os.makedirs(target_dir, exist_ok=True)
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    saved = 0
    for f in files:
        if not f or not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        safe_filename = sanitize_id(os.path.basename(f.filename)) or "image"
        base, _ = os.path.splitext(safe_filename)
        path = os.path.join(target_dir, f"{base}{ext}")
        idx = 0
        while os.path.exists(path):
            idx += 1
            path = os.path.join(target_dir, f"{base}_{idx}{ext}")
        try:
            f.save(path)
            saved += 1
        except Exception:
            pass

    return jsonify({"saved": saved, "path": target_dir})


@training_bp.post("/api/training/train-faces")
def api_training_train_faces():
    payload = request.get_json(force=True) or {}
    celebrity_name = (payload.get("celebrity_name") or "").strip()
    train_all = bool(payload.get("all", False))

    device = "cpu"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    face_model = (payload.get("face_model") or "buffalo_l").strip().lower()
    if face_model not in ("buffalo_l", "buffalo_s", "buffalo_sc"):
        face_model = "buffalo_l"

    if train_all:
        if not os.path.isdir(TRAINING_FACES_DIR):
            return jsonify({"error": "No face datasets found", "registered": 0}), 400
        total_registered = 0
        errors: list[str] = []
        for subdir in sorted(os.listdir(TRAINING_FACES_DIR)):
            path = os.path.join(TRAINING_FACES_DIR, subdir)
            if not os.path.isdir(path):
                continue
            try:
                from face_pipeline.register_known import register_faces_from_folder

                count, err = register_faces_from_folder(
                    path, subdir, device=device, model_name=face_model, conf_thresh=0.8
                )
                if err:
                    errors.append(f"{subdir}: {err}")
                else:
                    total_registered += count
            except Exception as e:
                errors.append(f"{subdir}: {e}")
        return jsonify({"registered": total_registered, "errors": errors})

    safe_name = sanitize_dataset_name(celebrity_name)
    if not safe_name:
        return jsonify({"error": "Invalid or empty celebrity name"}), 400
    images_dir = os.path.join(TRAINING_FACES_DIR, safe_name)
    if not os.path.isdir(images_dir):
        return jsonify({"error": f"No dataset found for '{celebrity_name}'"}), 404
    try:
        from face_pipeline.register_known import register_faces_from_folder

        count, err = register_faces_from_folder(
            images_dir, safe_name, device=device, model_name=face_model, conf_thresh=0.8
        )
        if err:
            return jsonify({"error": err, "registered": count}), 500
        return jsonify({"registered": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.post("/api/training/build-monument-model")
def api_training_build_monument_model():
    from pipeline.monuments import build_and_train_monument_model  # noqa: WPS433

    ensure_directories()
    payload = request.get_json(silent=True) or {}
    preprocess = (payload.get("preprocess") or "none").strip().lower()
    if preprocess not in ("none", "rembg"):
        preprocess = "none"
    clear_cache = bool(payload.get("clear_feature_cache", False))
    class_weight_balanced = bool(payload.get("class_weight_balanced", False))

    device = "cpu"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    try:
        result = build_and_train_monument_model(
            dataset_dir=TRAINING_DATASET_DIR,
            monuments_dir=TRAINING_MONUMENTS_DIR,
            model_dir=MONUMENT_MODEL_DIR,
            device=device,
            clear_feature_cache=clear_cache,
            preprocess=preprocess,
            class_weight_balanced=class_weight_balanced,
        )
        if result.get("trained"):
            return jsonify(result)
        return jsonify({"error": result.get("error", "Training failed")}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.get("/api/training/datasets")
def api_training_datasets():
    ensure_directories()
    faces = []
    for name in (
        sorted(os.listdir(TRAINING_FACES_DIR)) if os.path.isdir(TRAINING_FACES_DIR) else []
    ):
        path = os.path.join(TRAINING_FACES_DIR, name)
        if not os.path.isdir(path):
            continue
        count = sum(
            1
            for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
        )
        faces.append({"name": name, "count": count})
    monuments = []
    for name in (
        sorted(os.listdir(TRAINING_MONUMENTS_DIR))
        if os.path.isdir(TRAINING_MONUMENTS_DIR)
        else []
    ):
        path = os.path.join(TRAINING_MONUMENTS_DIR, name)
        if not os.path.isdir(path):
            continue
        count = sum(
            1
            for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
        )
        monuments.append({"name": name, "count": count})
    return jsonify({"faces": faces, "monuments": monuments})


@training_bp.get("/api/training/datasets/<dataset_type>/<name>")
def api_training_dataset_images(dataset_type: str, name: str):
    if dataset_type not in ("face", "monument"):
        return jsonify({"error": "type must be face or monument"}), 400
    safe_name = sanitize_dataset_name(name)
    if not safe_name:
        return jsonify({"error": "Invalid name"}), 400
    base = TRAINING_FACES_DIR if dataset_type == "face" else TRAINING_MONUMENTS_DIR
    path = os.path.join(base, safe_name)
    if not os.path.isdir(path):
        return jsonify({"error": "Dataset not found", "images": []}), 404
    images = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
        and os.path.splitext(f)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    ]
    return jsonify({"name": safe_name, "type": dataset_type, "images": sorted(images)})


@training_bp.delete("/api/training/image")
def api_training_delete_image():
    payload = request.get_json(force=True) or {}
    dataset_type = (payload.get("type") or "").strip().lower()
    name = (payload.get("name") or "").strip()
    filename = (payload.get("filename") or "").strip()
    if dataset_type not in ("face", "monument"):
        return jsonify({"error": "type must be face or monument"}), 400
    safe_name = sanitize_dataset_name(name)
    if not safe_name or not filename:
        return jsonify({"error": "name and filename required"}), 400
    if ".." in filename or os.path.sep in filename:
        return jsonify({"error": "Invalid filename"}), 400
    base = TRAINING_FACES_DIR if dataset_type == "face" else TRAINING_MONUMENTS_DIR
    path = os.path.join(base, safe_name, filename)
    if not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404
    try:
        os.remove(path)
        return jsonify({"deleted": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

