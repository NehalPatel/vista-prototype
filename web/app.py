import os
import sys
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
)
from pipeline.render import make_video_from_images

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
    conf_threshold = float(payload.get('conf_threshold', 0.7))
    fps = int(payload.get('fps', 1))

    if not url:
        return jsonify({"error": "URL is required"}), 400

    ensure_directories()

    # Derive video_id
    video_id = extract_video_id_from_url(url) or sanitize_id(url)
    if not validate_video_id(video_id):
        return jsonify({"error": "Invalid or unsupported video ID derived from URL"}), 400

    paths = get_video_results_paths(video_id)
    if not ensure_video_results_dirs(video_id):
        return jsonify({"error": "Failed to create results directories"}), 500

    # Prevent overwrite
    if os.path.exists(paths['detection_json']) or (
        os.path.isdir(paths['processed_frames']) and any(os.scandir(paths['processed_frames']))
    ):
        return jsonify({
            "error": f"Existing results found for video_id '{video_id}'. Remove them or use a different URL.",
            "video_id": video_id
        }), 409

    # Optional metadata
    meta = get_video_metadata(url)

    try:
        # Download, extract, detect, save, render
        video_path = download_video(url, VIDEOS_DIR)
        extract_frames(video_path, FRAMES_DIR)

        results_by_frame = run_yolo(
            frames_dir=FRAMES_DIR,
            detections_dir=paths['processed_frames'],
            model_path='yolov8n.pt',
            conf_threshold=conf_threshold,
        )

        total_dets, by_class = generate_summary(results_by_frame)
        total_frames = len(results_by_frame)

        save_detection_results(
            results_by_frame=results_by_frame,
            output_json_path=paths['detection_json'],
            video_id=video_id,
            conf_threshold=conf_threshold,
        )

        # Metadata file
        device = 'cpu'
        try:
            import torch  # type: ignore
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'

        write_metadata(
            metadata_path=paths['metadata_txt'],
            video_id=video_id,
            source=url,
            total_frames=total_frames,
            total_detections=total_dets,
            by_class=by_class,
            model_name='yolov8n.pt',
            device=device,
            conf_threshold=conf_threshold,
        )

        # Render video
        out_video = os.path.join(paths['base'], 'detections_video.mp4')
        make_video_from_images(paths['processed_frames'], out_video, fps=fps)

        return jsonify({
            "status": "completed",
            "video_id": video_id,
            "metadata": meta,
            "summary": {
                "total_frames": total_frames,
                "total_detections": total_dets,
                "by_class": by_class,
                "confidence_threshold": conf_threshold,
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