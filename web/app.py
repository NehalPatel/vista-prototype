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
)
from pipeline.utils import (
    extract_video_id_from_url,
    sanitize_id,
    validate_video_id,
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

try:
    import yt_dlp  # type: ignore
except Exception:
    yt_dlp = None


app = Flask(__name__, static_folder='static', template_folder='templates')

BASE_DIR = os.path.abspath(os.getcwd())
RESULTS_BASE = os.path.join(BASE_DIR, 'vista-prototype', 'results')
FRAMES_DIR = os.path.join(BASE_DIR, 'vista-prototype', 'frames')
VIDEOS_DIR = os.path.join(BASE_DIR, 'vista-prototype', 'videos')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results/<path:filename>')
def serve_results(filename: str):
    return send_from_directory(RESULTS_BASE, filename)


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