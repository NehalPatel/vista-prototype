import os
import sys
import json
import shutil
import platform
import time
import subprocess
from flask import Flask, render_template, request, jsonify, send_from_directory

# Ensure the parent directory is on sys.path for 'pipeline' imports
CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from pipeline.paths import (
    ensure_directories,
    get_video_results_paths,
    ensure_video_results_dirs,
    TRAINING_FACES_DIR,
    TRAINING_MONUMENTS_DIR,
    TRAINING_DATASET_DIR,
    MONUMENT_MODEL_DIR,
)
from pipeline.utils import (
    extract_video_id_from_url,
    sanitize_id,
    validate_video_id,
    sanitize_dataset_name,
)
from pipeline.video import download_video, extract_frames
from pipeline.detection import (
    run_yolo,
    generate_summary,
    save_detection_results,
    write_metadata,
    _resolve_model_path,
    OBJECT_MODEL_CHOICES,
)
from pipeline.render import make_video_from_images
from pipeline.faces import run_face_detection
from pipeline.monuments import (
    build_and_train_monument_model,
    run_monument_recognition,
    load_monument_model,
)

try:
    import yt_dlp  # type: ignore
except Exception:
    yt_dlp = None


app = Flask(__name__, static_folder='static', template_folder='templates')

BASE_DIR = os.path.abspath(os.getcwd())
RESULTS_BASE = os.path.join(BASE_DIR, 'vista-prototype', 'results')
FRAMES_DIR = os.path.join(BASE_DIR, 'vista-prototype', 'frames')
VIDEOS_DIR = os.path.join(BASE_DIR, 'vista-prototype', 'videos')

# Ensure runtime directories (including training_data) exist at startup
ensure_directories()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/training')
def training_page():
    """Training Data Manager page: face and monument dataset upload/train."""
    return render_template('training.html')


@app.route('/results/<path:filename>')
def serve_results(filename: str):
    return send_from_directory(RESULTS_BASE, filename)


# --- Training Data Manager API ---

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _training_upload_dir(dataset_type: str, name: str):
    """Return the directory path for a given dataset type and sanitized name."""
    safe_name = sanitize_dataset_name(name)
    if not safe_name:
        return None
    if dataset_type == "face":
        return os.path.join(TRAINING_FACES_DIR, safe_name)
    if dataset_type == "monument":
        return os.path.join(TRAINING_MONUMENTS_DIR, safe_name)
    return None


@app.route("/api/training/upload", methods=["POST"])
def api_training_upload():
    """Upload images for a face or monument dataset. Form: name, type (face|monument), files (multiple)."""
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
        safe_name = sanitize_id(os.path.basename(f.filename)) or "image"
        base, _ = os.path.splitext(safe_name)
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


@app.route("/api/training/train-faces", methods=["POST"])
def api_training_train_faces():
    """Train/register face dataset: run embeddings from training_data/faces and update known_faces."""
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
        errors = []
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
        return jsonify({
            "registered": total_registered,
            "errors": errors,
        })
    else:
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


@app.route("/api/training/build-monument-model", methods=["POST"])
def api_training_build_monument_model():
    """Build and train the monument classifier from training_data/dataset and training_data/monuments."""
    ensure_directories()
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
        )
        if result.get("trained"):
            return jsonify(result)
        return jsonify({"error": result.get("error", "Training failed")}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/training/datasets", methods=["GET"])
def api_training_datasets():
    """List face and monument datasets with image counts."""
    ensure_directories()
    faces = []
    for name in sorted(os.listdir(TRAINING_FACES_DIR)) if os.path.isdir(TRAINING_FACES_DIR) else []:
        path = os.path.join(TRAINING_FACES_DIR, name)
        if not os.path.isdir(path):
            continue
        count = sum(1 for f in os.listdir(path) if os.path.splitext(f)[1].lower() in ALLOWED_IMAGE_EXTENSIONS)
        faces.append({"name": name, "count": count})
    monuments = []
    for name in sorted(os.listdir(TRAINING_MONUMENTS_DIR)) if os.path.isdir(TRAINING_MONUMENTS_DIR) else []:
        path = os.path.join(TRAINING_MONUMENTS_DIR, name)
        if not os.path.isdir(path):
            continue
        count = sum(1 for f in os.listdir(path) if os.path.splitext(f)[1].lower() in ALLOWED_IMAGE_EXTENSIONS)
        monuments.append({"name": name, "count": count})
    return jsonify({"faces": faces, "monuments": monuments})


@app.route("/api/training/datasets/<dataset_type>/<name>", methods=["GET"])
def api_training_dataset_images(dataset_type: str, name: str):
    """List image filenames in a dataset. dataset_type: face | monument."""
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
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    ]
    return jsonify({"name": safe_name, "type": dataset_type, "images": sorted(images)})


@app.route("/api/training/image", methods=["DELETE"])
def api_training_delete_image():
    """Delete one image from a dataset. JSON body: type (face|monument), name, filename."""
    payload = request.get_json(force=True) or {}
    dataset_type = (payload.get("type") or "").strip().lower()
    name = (payload.get("name") or "").strip()
    filename = (payload.get("filename") or "").strip()
    if dataset_type not in ("face", "monument"):
        return jsonify({"error": "type must be face or monument"}), 400
    safe_name = sanitize_dataset_name(name)
    if not safe_name or not filename:
        return jsonify({"error": "name and filename required"}), 400
    # Prevent path traversal
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


def get_system_info():
    """Return Python version, CPU name, and GPU name (if available)."""
    python_version = platform.python_version()
    cpu_name = platform.processor() or "—"
    if sys.platform == "win32" and (not cpu_name or cpu_name.strip() == ""):
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"],
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                timeout=5,
            )
            lines = out.decode("utf-8", errors="replace").strip().splitlines()
            if len(lines) >= 2:
                cpu_name = lines[1].strip() or cpu_name
        except Exception:
            pass
    gpu_name = "—"
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0) or "CUDA GPU"
    except Exception:
        pass
    return {
        "python_version": python_version,
        "cpu": cpu_name,
        "gpu": gpu_name,
    }


@app.route("/api/system-info", methods=["GET"])
def api_system_info():
    return jsonify(get_system_info())


def get_video_metadata(url: str):
    if not yt_dlp:
        return {}


 
    try:
        ydl_opts = {"quiet": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title")
            duration = info.get("duration")  # seconds
            # Prefer max resolution thumbnail
            thumbs = info.get("thumbnails") or []
            thumb_url = None
            if thumbs:
                thumbs_sorted = sorted(thumbs, key=lambda t: t.get('width', 0), reverse=True)
                thumb_url = (thumbs_sorted[0] or {}).get('url')
            return {"title": title, "duration": duration, "thumbnail": thumb_url}
    except Exception:
        return {}


@app.route('/api/process', methods=['POST'])
def api_process():
    payload = request.get_json(force=True) or {}
    url = payload.get('url', '').strip()
    conf_threshold = float(payload.get('conf_threshold', 0.5))
    fps = int(payload.get('fps', 1))
    force_rescan = bool(payload.get('force_rescan', False))
    scan_mode = str(payload.get('scan_mode', 'both')).lower()
    object_model = (payload.get('object_model') or 'yolov8n').strip().lower()
    if object_model not in OBJECT_MODEL_CHOICES:
        object_model = 'yolov8n'

    scan_start_seconds = float(payload.get('scan_start_seconds', 0))
    scan_end_seconds = payload.get('scan_end_seconds')
    if scan_end_seconds is not None:
        scan_end_seconds = float(scan_end_seconds)
    else:
        scan_end_seconds = scan_start_seconds + 180
    if scan_end_seconds <= scan_start_seconds:
        return jsonify({"error": "Scan end time must be greater than scan start time."}), 400

    if not url:
        return jsonify({"error": "URL is required"}), 400

    ensure_directories()

    # Derive video_id
    video_id = extract_video_id_from_url(url) or sanitize_id(url)
    if not validate_video_id(video_id):
        return jsonify({"error": "Invalid or unsupported video ID derived from URL"}), 400

    paths = get_video_results_paths(video_id)

    # If force rescan, remove existing results and frames for this video
    if force_rescan:
        base = paths["base"]
        frames_dir_video = os.path.join(FRAMES_DIR, video_id)
        if os.path.isdir(base):
            try:
                shutil.rmtree(base)
            except Exception:
                pass
        if os.path.isdir(frames_dir_video):
            try:
                shutil.rmtree(frames_dir_video)
            except Exception:
                pass

    if not ensure_video_results_dirs(video_id):
        return jsonify({"error": "Failed to create results directories"}), 500

    # Return cached results if they exist and we're not forcing a rescan
    if not force_rescan and (
        os.path.exists(paths['detection_json']) or (
            os.path.isdir(paths['processed_frames']) and any(os.scandir(paths['processed_frames']))
        )
    ):
        print(f"[trace] Returning CACHED results (force_rescan={force_rescan})")
        meta = get_video_metadata(url)
        total_frames = 0
        total_dets = 0
        by_class = {}
        conf_used = conf_threshold
        try:
            with open(paths['detection_json'], 'r', encoding='utf-8') as f:
                dj = json.load(f)
            frames_list = dj.get('frames') or []
            results_by_frame = { (fr.get('frame') or ''): (fr.get('detections') or []) for fr in frames_list }
            total_dets, by_class = generate_summary(results_by_frame)
            total_frames = len(results_by_frame)
            total_face_detections = sum(len(fr.get('faces') or []) for fr in frames_list)
            conf_used = dj.get('confidence_threshold', conf_used)
            object_model_cached = dj.get('object_model', 'yolov8n')
            face_model_cached = dj.get('face_model', payload.get('face_model', 'buffalo_l'))
        except Exception:
            object_model_cached = 'yolov8n'
            face_model_cached = payload.get('face_model', 'buffalo_l')
            total_face_detections = 0

        return jsonify({
            "status": "cached",
            "video_id": video_id,
            "metadata": meta,
            "summary": {
                "total_frames": total_frames,
                "total_detections": total_dets,
                "total_face_detections": total_face_detections,
                "by_class": by_class,
                "confidence_threshold": conf_used,
                "object_model": object_model_cached,
                "face_model": face_model_cached,
            },
            "results": {
                "output_video_url": f"/results/{video_id}/detections_video.mp4",
                "detection_json_url": f"/results/{video_id}/detection_results.json",
                "metadata_url": f"/results/{video_id}/metadata.txt",
            }
        })

    # Optional metadata
    meta = get_video_metadata(url)
    run_stats = {}

    try:
        print("[trace] Starting fresh run (download -> frames -> detection -> face)")
        # Download video (required)
        t0 = time.perf_counter()
        video_path = download_video(url, VIDEOS_DIR)
        run_stats["download_sec"] = round(time.perf_counter() - t0, 2)
        if not video_path or not os.path.isfile(video_path):
            return jsonify({
                "error": "Video download failed. Try again or use a different URL; some videos may be restricted.",
                "video_id": video_id
            }), 500

        # Per-video frames directory so we only process this video's frames
        frames_dir_this_video = os.path.join(FRAMES_DIR, video_id)
        os.makedirs(frames_dir_this_video, exist_ok=True)
        t1 = time.perf_counter()
        saved_frames = extract_frames(
            video_path,
            frames_dir_this_video,
            start_seconds=scan_start_seconds,
            end_seconds=scan_end_seconds,
        )
        run_stats["extract_frames_sec"] = round(time.perf_counter() - t1, 2)

        if not saved_frames:
            return jsonify({
                "error": "No frames could be extracted from the video (file may be corrupted or unreadable).",
                "video_id": video_id
            }), 500

        # Decide which pipelines to run based on scan_mode
        run_objects = scan_mode in ("objects", "both")
        run_faces = scan_mode in ("faces", "both")

        results_by_frame: dict = {}
        total_dets = 0
        by_class: dict = {}
        run_stats["detection_sec"] = 0.0

        # Detect device once (GPU if available) and use for both YOLO and face detection
        device = "cpu"
        gpu_name = None
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass
        run_stats["device"] = device
        run_stats["gpu_name"] = gpu_name

        # Object detection (YOLO) – only when enabled
        if run_objects:
            model_path = _resolve_model_path(object_model, BASE_DIR)
            t2 = time.perf_counter()
            results_by_frame = run_yolo(
                frames_dir=frames_dir_this_video,
                detections_dir=paths['processed_frames'],
                model_path=model_path,
                conf_threshold=conf_threshold,
                device=device,
            )
            run_stats["detection_sec"] = round(time.perf_counter() - t2, 2)
            total_dets, by_class = generate_summary(results_by_frame)
        else:
            # Faces-only mode: copy raw frames into processed_frames so we can draw faces + render video
            import cv2

            os.makedirs(paths['processed_frames'], exist_ok=True)
            for fname in sorted(os.listdir(frames_dir_this_video)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                src = os.path.join(frames_dir_this_video, fname)
                dst = os.path.join(paths['processed_frames'], fname)
                img = cv2.imread(src)
                if img is None:
                    continue
                cv2.imwrite(dst, img)
                results_by_frame[fname] = []

        total_frames = len(results_by_frame)

        # Face detection: run on original frames for better recall, draw on annotated frames (only when enabled)
        face_model_name = payload.get("face_model", "buffalo_l")
        faces_by_frame: dict = {}
        total_face_detections = 0
        print(f"[trace] scan_mode={scan_mode!r} run_objects={run_objects} run_faces={run_faces}")
        if run_faces:
            print("[trace] Calling run_face_detection(...)")
            t_face = time.perf_counter()
            try:
                face_conf_threshold = float(payload.get("face_conf_threshold", 0.5))
            except Exception:
                face_conf_threshold = 0.5
            try:
                faces_by_frame = run_face_detection(
                    paths["processed_frames"],
                    face_model=face_model_name,
                    device=device,
                    face_conf_threshold=face_conf_threshold,
                    source_frames_dir=frames_dir_this_video,
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Face detection failed: %s", e, exc_info=True
                )
            run_stats["face_detection_sec"] = round(time.perf_counter() - t_face, 2)
            total_face_detections = sum(len(v) for v in faces_by_frame.values())

        # Monument recognition (if model was built from training_data/dataset or monuments)
        monuments_by_frame = {}
        if load_monument_model(MONUMENT_MODEL_DIR) is not None:
            try:
                t_mon = time.perf_counter()
                monument_conf = float(payload.get("monument_conf_threshold", 0.5))
                monuments_by_frame = run_monument_recognition(
                    paths["processed_frames"],
                    MONUMENT_MODEL_DIR,
                    device=device,
                    confidence_threshold=monument_conf,
                )
                run_stats["monument_recognition_sec"] = round(time.perf_counter() - t_mon, 2)
                # Draw monument label on each frame
                import cv2
                for fname, info in monuments_by_frame.items():
                    label = info.get("label")
                    conf = info.get("confidence", 0)
                    if label and label != "Unknown" and conf >= monument_conf:
                        path_img = os.path.join(paths["processed_frames"], fname)
                        img = cv2.imread(path_img)
                        if img is not None:
                            cv2.putText(
                                img, f"Monument: {label} ({conf:.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                            )
                            cv2.imwrite(path_img, img)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Monument recognition failed: %s", e)

        write_metadata(
            metadata_path=paths['metadata_txt'],
            video_id=video_id,
            source=url,
            total_frames=total_frames,
            total_detections=total_dets,
            by_class=by_class,
            model_name=f'{object_model}.pt',
            device=device,
            conf_threshold=conf_threshold,
        )

        # Render video
        t3 = time.perf_counter()
        out_video = os.path.join(paths['base'], 'detections_video.mp4')
        make_video_from_images(paths['processed_frames'], out_video, fps=fps)
        run_stats["render_sec"] = round(time.perf_counter() - t3, 2)
        run_stats["total_sec"] = round(
            run_stats.get("download_sec", 0)
            + run_stats.get("extract_frames_sec", 0)
            + run_stats.get("detection_sec", 0)
            + run_stats.get("face_detection_sec", 0)
            + run_stats.get("monument_recognition_sec", 0)
            + run_stats.get("render_sec", 0),
            2,
        )

        save_detection_results(
            results_by_frame=results_by_frame,
            output_json_path=paths['detection_json'],
            video_id=video_id,
            conf_threshold=conf_threshold,
            object_model=object_model,
            face_model=face_model_name,
            run_stats=run_stats,
            faces_by_frame=faces_by_frame,
            monuments_by_frame=monuments_by_frame,
        )

        return jsonify({
            "status": "completed",
            "video_id": video_id,
            "metadata": meta,
            "summary": {
                "total_frames": total_frames,
                "total_detections": total_dets,
                "total_face_detections": total_face_detections,
                "by_class": by_class,
                "confidence_threshold": conf_threshold,
                "object_model": object_model,
                "face_model": face_model_name,
                "run_stats": run_stats,
            },
            "results": {
                "output_video_url": f"/results/{video_id}/detections_video.mp4",
                "detection_json_url": f"/results/{video_id}/detection_results.json",
                "metadata_url": f"/results/{video_id}/metadata.txt",
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Use 0.0.0.0 so preview is accessible; port 8000 for clarity
    app.run(host='0.0.0.0', port=8000, debug=True)