# 🎥 VISTA – Video Intelligence Search & Tagging Assistant

<img src="assets/logo.png" alt="VISTA Logo" width="128" />

Face Detection • Face Recognition • Object Detection • Video Understanding

## Overview

VISTA is an intelligent video-processing system. This repository implements the object detection prototype with optional face detection, a simple web UI, and a CLI pipeline. It:

- Downloads YouTube videos
- Extracts keyframes (configurable, default 1 frame/sec)
- Runs YOLOv8 object detection (with optional per-object color)
- Runs optional InsightFace face detection (Buffalo L/S/SC) on frames
- Saves annotated frames (objects + face boxes), a single JSON of detections and faces, and a summary
- Renders an annotated video from processed frames

**Note:** Face detection is integrated but may report zero faces in some videos; this is under investigation. Object detection (YOLOv8) is stable.

## Processing Flow

![VISTA Processing Flow](assets/project-flow-diagram.png)

## Project Phases

1) Prototype (Current)
- YouTube download (PyTube, with `yt-dlp` fallback)
- Frame extraction (OpenCV)
- YOLOv8 object detection (Ultralytics)
- Single JSON results per video
- Annotated frames and rendered output video
- Web UI (Flask) for easy processing

2) Full System (Later)
- Backend (FastAPI)
- Processing Engine (GPU workers)
- PostgreSQL + pgvector
- Vector search (FAISS/Milvus)
- React/Next.js frontend
- Full video search engine with face + CLIP embeddings

## Folder Structure

At runtime, data is written under a nested `vista-prototype/` directory inside the repo root to keep outputs contained.

```
/ (repo root)
├── pipeline/                 # modular pipeline (download, frames, detection, render)
├── web/                      # Flask web UI (HTML/CSS/JS + API)
├── implementation.py         # CLI entrypoint for the pipeline
├── README.md                 # this file
└── vista-prototype/          # runtime data (auto-created)
    ├── videos/               # downloaded YouTube MP4s
    ├── frames/               # extracted raw frames
    └── results/
        └── <video_id>/
            ├── processed_frames/       # annotated frames (YOLO)
            ├── detection_results.json  # all detections for the video
            ├── metadata.txt            # summary + device info
            └── detections_video.mp4    # rendered video (may fall back to .avi)
```

Notes
- `video_id` is derived from the YouTube URL (or sanitized filename). Existing per-video results will not be overwritten; delete the folder to re-run.
- The pipeline defaults to `yolov8n.pt` for speed. Ultralytics will download the model automatically.
- GPU is optional. If `torch` with CUDA is available, YOLO can run on GPU; otherwise CPU is used.

## Technologies Used

- Object Detection: YOLOv8 (Ultralytics)
- Face Detection: InsightFace (Buffalo L/S/SC, optional; requires `insightface` + `onnxruntime` or `onnxruntime-gpu`)
- Frame Extraction: OpenCV
- Video Download: PyTube (fallback: `yt-dlp`)
- Web UI: Flask
- Optional: `tqdm` for progress bars, `torch` for GPU

## Prototype: Development Process

1) Download YouTube Video
- Highest-quality MP4 using PyTube; falls back to `yt-dlp` when PyTube fails.
- Saved into `vista-prototype/videos/`.

2) Extract Frames
- OpenCV; extracts 1 frame per second.
- Saved into `vista-prototype/frames/`.

3) Object Detection (YOLOv8)
- Runs on each frame; filters by confidence threshold.
- Annotated frames saved into `vista-prototype/results/<video_id>/processed_frames/`.
- Single JSON with all detections saved as `vista-prototype/results/<video_id>/detection_results.json`.

4) Metadata + Summary
- `metadata.txt` includes device, model, confidence threshold, counts per class.
- Total frames and detections reported via API/CLI.

5) Render Annotated Video
- Frames in `processed_frames/` are rendered to `detections_video.mp4` (fallback to `.avi` if MP4 writer is unavailable).

## Installation

Create a virtual environment and install dependencies.

Windows (PowerShell)
```
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
pip install ultralytics opencv-python pytube flask tqdm yt-dlp pillow
```

macOS/Linux (bash)
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install ultralytics opencv-python pytube flask tqdm yt-dlp pillow
```

**Optional – face detection:** For face detection in the web UI and pipeline, install InsightFace and an ONNX runtime:
```
pip install insightface onnxruntime        # CPU
pip install insightface onnxruntime-gpu    # GPU (CUDA)
```
- `yt-dlp` is optional but recommended; the pipeline uses it when PyTube cannot download.
- If you plan to use GPU for YOLO, install a CUDA-enabled `torch` build.

## Running

CLI (process a video once)
```
python implementation.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --conf-threshold 0.7 --fps 1
```
Or process a local file
```
python implementation.py --video "path/to/video.mp4" --conf-threshold 0.7 --fps 1
```
Outputs are written to `vista-prototype/results/<video_id>/`.

Web UI (interactive)
```
python web/app.py
```
- Open `http://localhost:8000/`
- Paste a YouTube URL, set confidence threshold, face model (if using face detection), and output FPS, then Process
- Links to the annotated video, detection JSON, and metadata are provided
- Detected objects (and faces, when detected) appear as clickable badges; click to view frames containing that class

API
- Endpoint: `POST /api/process`
- Body: `{ "url": string, "conf_threshold": float, "fps": int, "face_model": "buffalo_l" | "buffalo_s" | "buffalo_sc", ... }`
- Returns: `video_id`, `summary` (including `total_face_detections` when face detection runs), and URLs to output files under `/results/<video_id>/...`

## Output Summary

- Total frames processed
- Total objects detected
- Total face detections (when InsightFace is installed and used)
- Counts per class (including Face in the object badges when faces are detected)
- Confidence threshold and models used

## Suggested Test Videos

- Busy Street Intersection — `2vjEKevuV4k`
- People Walking with Dogs — `iQZM1zO0Fdk`
- Cat Playing — `J---aiyznGQ`

## Later Expansion – Full System Architecture

Three separate projects will compose the full VISTA system:

- Frontend (React/Next.js): upload, search panel, view detections
- Backend (FastAPI): stores metadata, exposes search API, vector search integration
- Processing Engine (GPU workers): download, keyframes, YOLO, face (RetinaFace + ArcFace), CLIP indexing, push data to backend
- Vector Database (pgvector/Milvus): face + CLIP embeddings, text-to-video search

Future Features
- Multi-object tracking
- Person action recognition
- Animal breed classification
- Landmark/monument recognition
- Depth estimation (MiDaS)
- Full video search engine
- FaceTag identity system
- Admin panel for label correction
- Multi-language caption-based search

## Author

Prof. Nehal Patel — AI Researcher • Developer • Educator