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
├── scripts/                  # organize_training_data.py, build_models.py
├── implementation.py        # CLI entrypoint for the pipeline
├── README.md                 # this file
└── vista-prototype/         # runtime data (auto-created)
    ├── videos/               # downloaded YouTube MP4s
    ├── frames/               # extracted raw frames
    ├── results/
    │   └── <video_id>/
    │       ├── processed_frames/       # annotated frames (YOLO + faces)
    │       ├── detection_results.json  # all detections for the video
    │       ├── metadata.txt            # summary + device info
    │       └── detections_video.mp4   # rendered video (may fall back to .avi)
    ├── training_data/        # datasets for face & monument training (see below)
    ├── known_faces/          # face recognition model (embeddings; built by build_models.py)
    └── monument_model/       # monument classifier (built by build_models.py)
```

## Datasets and training

To **recognize faces** (e.g. celebrities) and **classify monuments** in videos, add training images and build the models.

### Where to put datasets

All paths below are under `vista-prototype/` (created at first run).

| Purpose | Folder | Layout |
|--------|--------|--------|
| **Face recognition** (final) | `vista-prototype/training_data/faces/<name>/` | One folder per person; put images of that person inside. |
| **Monument classification** (final) | `vista-prototype/training_data/monuments/<name>/` | One folder per monument; put images inside. |
| **Unorganized faces** | `vista-prototype/training_data/inbox_faces/<name>/` | One subfolder per person; run organize script to copy into `faces/`. |
| **Unorganized monuments** | `vista-prototype/training_data/inbox_monuments/<name>/` | One subfolder per monument; run organize script to copy into `monuments/`. |
| **Bulk datasets** (e.g. Kaggle) | `vista-prototype/training_data/datasets/faces/` and `.../datasets/monuments/` | Place downloaded datasets here; use `--from-datasets` when running the organize script. |

See `Dataset.md` for example Kaggle datasets (e.g. Indian cricketers, Indian monuments).

### Commands to run (from repo root)

**1. Organize images** (copy from inbox or from `datasets/` into `faces/` and `monuments/`):

```bash
# Preview what will be copied (dry run)
python scripts/organize_training_data.py --from-datasets --dry-run

# Organize from training_data/datasets/ into faces/ and monuments/
python scripts/organize_training_data.py --from-datasets

# Only faces
python scripts/organize_training_data.py --from-datasets --faces-only

# Only monuments
python scripts/organize_training_data.py --from-datasets --monuments-only

# Organize from inbox_faces/ and inbox_monuments/ (no --from-datasets)
python scripts/organize_training_data.py
```

**2. Build models** (face recognition + monument classifier):

```bash
# Build both face and monument models
python scripts/build_models.py

# Only face recognition (from training_data/faces/ → vista-prototype/known_faces/)
python scripts/build_models.py --faces-only

# Only monument model (from training_data/monuments/ and training_data/dataset/ → vista-prototype/monument_model/)
python scripts/build_models.py --monuments-only

# Use GPU if available
python scripts/build_models.py --device cuda
```

After building, process a video in the web UI or CLI; recognized faces and monuments will appear in the results. You can also upload images and train from the **Training** page in the web UI (`/training`).

### Notes
- `video_id` is derived from the YouTube URL (or sanitized filename). Existing per-video results will not be overwritten; delete the folder to re-run.
- The pipeline defaults to `yolov8n.pt` for speed. Ultralytics will download the model automatically.
- GPU is optional. If `torch` with CUDA is available, YOLO can run on GPU; otherwise CPU is used.

## MongoDB (optional)

To persist detection results for a **separate search engine project** (e.g. query "Nehal in red car" and rank videos by face + object), set:

- **`MONGODB_URI`** (or `MONGO_URI`): connection string (e.g. `mongodb://localhost:27017` or Atlas SRV).
- **`VISTA_DB_NAME`** (optional): database name; default `vista_search`.

After each successful video run, the app writes to two collections:

- **`videos`**: one document per video (video_id, source_url, title, duration_sec, thumbnail, face_labels, object_labels, monument_labels, summary, run_stats).
- **`frames`**: one document per frame (video_id, frame_filename, frame_index, time_sec, objects, faces, monument).

Indexes are created for efficient search on `faces.label`, `objects.class`, `objects.color`, `objects.label`. If `MONGODB_URI` is not set, indexing is skipped and the app behaves as before (JSON and files only). See `MONGODB_SEARCH_ENGINE_PLAN.md` and `pipeline/mongodb_store.py` for the full schema.

## Technologies Used

- Object Detection: YOLOv8 (Ultralytics)
- Face Detection: InsightFace (Buffalo L/S/SC, optional; requires `insightface` + `onnxruntime` or `onnxruntime-gpu`)
- Frame Extraction: OpenCV
- Video Download: PyTube (fallback: `yt-dlp`)
- Web UI: Flask
- Optional: `pymongo` for MongoDB, `tqdm` for progress bars, `torch` for GPU

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