from __future__ import annotations

import json
import logging
import os
import shutil
import time

from flask import Blueprint, jsonify, request

from pipeline.paths import (
    FRAMES_DIR,
    MONUMENT_MODEL_DIR,
    VIDEOS_DIR,
    ensure_directories,
    ensure_video_results_dirs,
    get_video_results_paths,
)
from pipeline.utils import extract_video_id_from_url, sanitize_id, validate_video_id
from web.api.system import get_video_metadata


processing_bp = Blueprint("processing", __name__)
logger = logging.getLogger(__name__)


def _repo_root() -> str:
    # web/api/processing.py -> web -> repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _pick_device() -> tuple[str, str | None]:
    device = "cpu"
    gpu_name = None
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return device, gpu_name


@processing_bp.post("/api/process")
def api_process():
    # Import heavier pipeline modules lazily so the web server starts fast.
    from pipeline.detection import (  # noqa: WPS433
        OBJECT_MODEL_CHOICES,
        _resolve_model_path,
        generate_summary,
        run_yolo,
        save_detection_results,
        write_metadata,
    )
    from pipeline.faces import run_face_detection  # noqa: WPS433
    from pipeline.monuments import (  # noqa: WPS433
        load_monument_model,
        run_monument_recognition,
    )
    from pipeline.mongodb_store import index_detection_results_to_mongodb  # noqa: WPS433
    from pipeline.render import make_video_from_images  # noqa: WPS433
    from pipeline.video import download_video, extract_frames  # noqa: WPS433

    payload = request.get_json(force=True) or {}
    url = str(payload.get("url", "")).strip()
    conf_threshold = float(payload.get("conf_threshold", 0.5))
    fps = int(payload.get("fps", 1))
    force_rescan = bool(payload.get("force_rescan", False))
    scan_mode = str(payload.get("scan_mode", "both")).lower()
    object_model = (payload.get("object_model") or "yolov8n").strip().lower()
    if object_model not in OBJECT_MODEL_CHOICES:
        object_model = "yolov8n"

    scan_start_seconds = float(payload.get("scan_start_seconds", 0))
    scan_end_seconds = payload.get("scan_end_seconds")
    if scan_end_seconds is not None:
        scan_end_seconds = float(scan_end_seconds)
    else:
        scan_end_seconds = scan_start_seconds + 180
    if scan_end_seconds <= scan_start_seconds:
        return jsonify({"error": "Scan end time must be greater than scan start time."}), 400

    if not url:
        return jsonify({"error": "URL is required"}), 400

    ensure_directories()

    video_id = extract_video_id_from_url(url) or sanitize_id(url)
    if not validate_video_id(video_id):
        return jsonify({"error": "Invalid or unsupported video ID derived from URL"}), 400

    paths = get_video_results_paths(video_id)

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
        os.path.exists(paths["detection_json"])
        or (
            os.path.isdir(paths["processed_frames"])
            and any(os.scandir(paths["processed_frames"]))
        )
    ):
        meta = get_video_metadata(url)
        total_frames = 0
        total_dets = 0
        by_class: dict = {}
        conf_used = conf_threshold
        try:
            with open(paths["detection_json"], "r", encoding="utf-8") as f:
                dj = json.load(f)
            frames_list = dj.get("frames") or []
            results_by_frame = {
                (fr.get("frame") or ""): (fr.get("detections") or []) for fr in frames_list
            }
            total_dets, by_class = generate_summary(results_by_frame)
            total_frames = len(results_by_frame)
            total_face_detections = sum(
                len(fr.get("faces") or []) for fr in frames_list
            )
            conf_used = dj.get("confidence_threshold", conf_used)
            object_model_cached = dj.get("object_model", "yolov8n")
            face_model_cached = dj.get("face_model", payload.get("face_model", "buffalo_l"))
        except Exception:
            object_model_cached = "yolov8n"
            face_model_cached = payload.get("face_model", "buffalo_l")
            total_face_detections = 0

        return jsonify(
            {
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
                },
            }
        )

    meta = get_video_metadata(url)
    run_stats: dict = {}

    try:
        # Download video (required)
        t0 = time.perf_counter()
        video_path = download_video(url, VIDEOS_DIR)
        run_stats["download_sec"] = round(time.perf_counter() - t0, 2)
        if not video_path or not os.path.isfile(video_path):
            return (
                jsonify(
                    {
                        "error": "Video download failed. Try again or use a different URL; some videos may be restricted.",
                        "video_id": video_id,
                    }
                ),
                500,
            )

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
            return (
                jsonify(
                    {
                        "error": "No frames could be extracted from the video (file may be corrupted or unreadable).",
                        "video_id": video_id,
                    }
                ),
                500,
            )

        run_objects = scan_mode in ("objects", "both")
        run_faces = scan_mode in ("faces", "both")

        results_by_frame: dict = {}
        total_dets = 0
        by_class: dict = {}
        run_stats["detection_sec"] = 0.0

        device, gpu_name = _pick_device()
        run_stats["device"] = device
        run_stats["gpu_name"] = gpu_name

        if run_objects:
            model_path = _resolve_model_path(object_model, _repo_root())
            t2 = time.perf_counter()
            results_by_frame = run_yolo(
                frames_dir=frames_dir_this_video,
                detections_dir=paths["processed_frames"],
                model_path=model_path,
                conf_threshold=conf_threshold,
                device=device,
            )
            run_stats["detection_sec"] = round(time.perf_counter() - t2, 2)
            total_dets, by_class = generate_summary(results_by_frame)
        else:
            import cv2

            os.makedirs(paths["processed_frames"], exist_ok=True)
            for fname in sorted(os.listdir(frames_dir_this_video)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                src = os.path.join(frames_dir_this_video, fname)
                dst = os.path.join(paths["processed_frames"], fname)
                img = cv2.imread(src)
                if img is None:
                    continue
                cv2.imwrite(dst, img)
                results_by_frame[fname] = []

        total_frames = len(results_by_frame)

        face_model_name = payload.get("face_model", "buffalo_l")
        faces_by_frame: dict = {}
        total_face_detections = 0
        if run_faces:
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
                logger.warning("Face detection failed: %s", e, exc_info=True)
            run_stats["face_detection_sec"] = round(time.perf_counter() - t_face, 2)
            total_face_detections = sum(len(v) for v in faces_by_frame.values())

        monuments_by_frame = {}
        if load_monument_model(MONUMENT_MODEL_DIR) is not None:
            try:
                t_mon = time.perf_counter()
                # Stricter than object conf by default: avoid labeling street/crowd frames.
                try:
                    monument_conf = float(payload.get("monument_conf_threshold", max(0.75, conf_threshold)))
                except Exception:
                    monument_conf = max(0.75, conf_threshold)
                try:
                    monument_margin = float(payload.get("monument_margin_threshold", 0.15))
                except Exception:
                    monument_margin = 0.15
                monuments_by_frame = run_monument_recognition(
                    paths["processed_frames"],
                    MONUMENT_MODEL_DIR,
                    device=device,
                    confidence_threshold=monument_conf,
                    margin_threshold=monument_margin,
                    detections_by_frame=results_by_frame if run_objects else None,
                    max_person_count=int(payload.get("monument_max_person_count", 3)),
                    max_person_area_ratio=float(payload.get("monument_max_person_area_ratio", 0.25)),
                )
                run_stats["monument_recognition_sec"] = round(time.perf_counter() - t_mon, 2)
                run_stats["monument_conf_threshold"] = monument_conf
                run_stats["monument_margin_threshold"] = monument_margin

                import cv2

                for fname, info in monuments_by_frame.items():
                    label = info.get("label")
                    conf = info.get("confidence", 0)
                    if label and label != "Unknown" and conf >= monument_conf:
                        path_img = os.path.join(paths["processed_frames"], fname)
                        img = cv2.imread(path_img)
                        if img is not None:
                            h_img, w_img = img.shape[:2]
                            margin = max(6, int(round(0.02 * min(h_img, w_img))))
                            x1, y1 = margin, margin
                            x2, y2 = max(x1 + 1, w_img - margin), max(y1 + 1, h_img - margin)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(
                                img,
                                f"Monument: {label} ({conf:.2f})",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )
                            cv2.imwrite(path_img, img)
                            try:
                                info["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
                            except Exception:
                                pass
            except Exception as e:
                logger.warning("Monument recognition failed: %s", e)

        write_metadata(
            metadata_path=paths["metadata_txt"],
            video_id=video_id,
            source=url,
            total_frames=total_frames,
            total_detections=total_dets,
            by_class=by_class,
            model_name=f"{object_model}.pt",
            device=device,
            conf_threshold=conf_threshold,
        )

        t3 = time.perf_counter()
        out_video = os.path.join(paths["base"], "detections_video.mp4")
        make_video_from_images(paths["processed_frames"], out_video, fps=fps)
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
            output_json_path=paths["detection_json"],
            video_id=video_id,
            conf_threshold=conf_threshold,
            object_model=object_model,
            face_model=face_model_name,
            run_stats=run_stats,
            faces_by_frame=faces_by_frame,
            monuments_by_frame=monuments_by_frame,
        )

        try:
            mongo_ok = index_detection_results_to_mongodb(
                video_id=video_id,
                source_url=url,
                meta=meta,
                run_stats=run_stats,
                results_by_frame=results_by_frame,
                faces_by_frame=faces_by_frame,
                monuments_by_frame=monuments_by_frame,
                by_class=by_class,
                confidence_threshold=conf_threshold,
                object_model=object_model,
                face_model=face_model_name,
                fps=float(fps),
                clip_start_sec=float(scan_start_seconds),
            )
            if mongo_ok:
                logger.info("[mongo] Indexed video %r into MongoDB.", video_id)
            else:
                logger.info("[mongo] MongoDB indexing skipped or failed for video %r.", video_id)
        except Exception as e:
            logger.warning("MongoDB index skipped: %s", e)

        return jsonify(
            {
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
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

