# Detection Code Flow – Face, Object & Monument (vista-prototype)

This document describes how **face detection**, **object detection (YOLO)**, and **monument recognition** are wired in the project: which files and functions are called, from where, and what each does. Use it to trace execution and debug the Process pipeline.

---

## 1. High-level flow (when user clicks "Process")

```
Browser (POST /api/process)
    → web/app.py  api_process()
        → get_video_results_paths()              [pipeline/paths.py]
        → download_video()                       [pipeline/video.py]  → saves MP4 under videos/
        → extract_frames()                       [pipeline/video.py]  → raw frames (frame_0001.jpg, …)
        → run_yolo() or copy raw frames         [pipeline/detection.py]  → annotated frames in processed_frames/
        → run_face_detection()                   [pipeline/faces.py]  → face boxes + optional recognition
        │   → load_detector(), detect_faces()   [face_pipeline/detection.py]
        │   → load_known_embeddings(), match() [face_pipeline/recognition.py] (optional)
        → run_monument_recognition()            [pipeline/monuments.py]  → frame-level monument labels
        → save_detection_results()              [pipeline/detection.py]  → detection_results.json (objects, faces, monuments)
        → make_video_from_images()               [pipeline/render.py]
        → return JSON (summary: total_detections, total_face_detections, etc.)
```

---

## 2. Call chain: where each function is called

| Function | Defined in | Called from |
|----------|------------|-------------|
| `api_process()` | `web/app.py` | Flask route `POST /api/process` |
| `get_video_results_paths()`, `ensure_video_results_dirs()`, `ensure_directories()` | `pipeline/paths.py` | `web/app.py` |
| `extract_video_id_from_url()`, `sanitize_id()`, `validate_video_id()`, `sanitize_dataset_name()` | `pipeline/utils.py` | `web/app.py` |
| `download_video()`, `extract_frames()` | `pipeline/video.py` | `web/app.py` → `api_process()` |
| `run_yolo()`, `generate_summary()`, `save_detection_results()`, `write_metadata()`, `_resolve_model_path()` | `pipeline/detection.py` | `web/app.py` → `api_process()` |
| **`run_face_detection()`** | **`pipeline/faces.py`** | **`web/app.py` → `api_process()`** |
| `load_detector()`, `detect_faces()` | `face_pipeline/detection.py` | `pipeline/faces.py`, `face_pipeline/register_known.py` |
| `load_known_embeddings()`, `match()` | `face_pipeline/recognition.py` | `pipeline/faces.py` |
| `get_embedding()`, `save_embedding()` | `face_pipeline/embeddings.py` | `pipeline/faces.py`, `face_pipeline/register_known.py` |
| `load_monument_model()`, `run_monument_recognition()`, `build_and_train_monument_model()` | `pipeline/monuments.py` | `web/app.py` → `api_process()` or `api_training_build_monument_model()` |
| `make_video_from_images()`, `_list_images_sorted()` | `pipeline/render.py` | `web/app.py` → `api_process()` |
| `_get_onnx_providers()`, `_get_face_attr()` | `face_pipeline/detection.py` | `face_pipeline/detection.py` (internal) |
| `register_faces_from_folder()` | `face_pipeline/register_known.py` | `web/app.py` → `api_training_train_faces()` |

---

## 3. File-by-file: what each function does

### 3.1 `web/app.py`

| Function | Summary |
|----------|---------|
| `index()` | Serves main page (`index.html`). |
| `training_page()` | Serves Training Data Manager page (`training.html`). |
| `serve_results(filename)` | Sends result files (video, JSON) from `RESULTS_BASE`. |
| `_training_upload_dir(dataset_type, name)` | Returns directory path for a face or monument dataset (sanitized name). |
| `api_training_upload()` | POST: upload images for face or monument dataset; saves under `TRAINING_FACES_DIR` or `TRAINING_MONUMENTS_DIR`. |
| `api_training_train_faces()` | POST: train/register face datasets; calls `register_faces_from_folder()` for one or all datasets; updates `known_faces`. |
| `api_training_build_monument_model()` | POST: build and train monument classifier from `TRAINING_DATASET_DIR` and `TRAINING_MONUMENTS_DIR`; calls `build_and_train_monument_model()`. |
| `api_training_datasets()` | GET: list face and monument dataset names and image counts. |
| `api_training_dataset_images(dataset_type, name)` | GET: list image filenames in a dataset. |
| `api_training_delete_image()` | DELETE: remove one image from a dataset. |
| `get_system_info()` | Returns Python version, CPU, GPU, MongoDB status. |
| `api_system_info()` | GET: returns `get_system_info()` as JSON. |
| `get_video_metadata(url)` | Fetches title, duration, thumbnail for YouTube URL (yt-dlp, no download). |
| **`api_process()`** | **POST /api/process:** Validates URL and payload; gets `video_id` and paths; on cache hit returns cached summary. On fresh run: downloads video, extracts frames; runs YOLO (or copies frames if objects disabled); runs face detection when `scan_mode` in ("faces", "both"); runs monument recognition if model exists; writes metadata; renders video; saves `detection_results.json` (objects, faces, monuments); optionally indexes to MongoDB; returns summary and result URLs. |

---

### 3.2 `pipeline/paths.py`

| Function | Summary |
|----------|---------|
| `ensure_directories()` | Creates `VIDEOS_DIR`, `FRAMES_DIR`, `RESULTS_DIR`, `TRAINING_*`, `MONUMENT_MODEL_DIR`, etc. |
| `get_video_results_paths(video_id)` | Returns dict: `base`, `detection_json`, `processed_frames`, `metadata_txt` for that video. |
| `ensure_video_results_dirs(video_id)` | Creates per-video results directories; returns True on success. |

---

### 3.3 `pipeline/utils.py`

| Function | Summary |
|----------|---------|
| `safe_print(message)` | Prints with flush. |
| `extract_video_id_from_url(url)` | Extracts YouTube video ID from URL. |
| `sanitize_id(name)` | Sanitizes string for directory-friendly ID. |
| `validate_video_id(video_id)` | Returns True if valid for folder naming. |
| `sanitize_dataset_name(name)` | Sanitizes for training dataset folder name. |

---

### 3.4 `pipeline/video.py`

| Function | Summary |
|----------|---------|
| `download_video(url, output_dir)` | Downloads highest-resolution MP4 for YouTube URL (pytube; fallback yt-dlp). Returns path or None. |
| `extract_frames(video_path, frames_dir, start_seconds, end_seconds)` | Extracts one frame per second in time range; saves as `frame_0001.jpg`, …; returns list of filenames. Clears existing frames first. |

---

### 3.5 `pipeline/detection.py` (object detection + shared save)

| Function | Summary |
|----------|---------|
| `_get_dominant_color_name(crop_bgr)` | Infers dominant color name from BGR crop (HSV) for object labels. |
| `_resolve_model_path(model_key, base_dir)` | Returns `.pt` path for YOLO model key; prefers local file. |
| `_inference_device()` | Returns `'cuda'` or `'cpu'` from PyTorch. |
| `run_yolo(frames_dir, detections_dir, model_path, conf_threshold, device)` | Runs YOLOv8 on each frame; saves annotated images; returns `Dict[frame_filename, list of detections]` (bbox, class, color, label, conf). |
| `generate_summary(results_by_frame)` | Returns (total_detections, by_class) from YOLO results. |
| **`save_detection_results(..., faces_by_frame=None, monuments_by_frame=None)`** | **Builds `frames` array from `results_by_frame`; attaches `faces_by_frame.get(frame, [])` and `monuments_by_frame.get(frame, {})` per frame; writes one JSON with video_id, thresholds, frames (detections, faces, monument), run_stats.** |
| `write_metadata(...)` | Writes text metadata file with run parameters and class counts. |

---

### 3.6 `pipeline/render.py`

| Function | Summary |
|----------|---------|
| `_list_images_sorted(images_dir)` | Returns sorted list of `.jpg` filenames (frame_XXXX order). |
| `make_video_from_images(images_dir, output_path, fps)` | Creates MP4 (or AVI fallback) from images at given fps. |

---

### 3.7 `pipeline/faces.py` (face detection)

| Function | Summary |
|----------|---------|
| **`run_face_detection(annotated_frames_dir, face_model, device, face_conf_threshold, source_frames_dir)`** | Loads InsightFace detector; optionally loads known-face embeddings. For each frame: runs `detect_faces()` on raw frame path; optionally matches with `get_embedding()`/`match()`; draws cyan face boxes and labels on annotated images. Returns `Dict[frame_filename, list of {bbox, confidence, label?, recognition_confidence?}]`. |

---

### 3.8 `pipeline/monuments.py` (monument recognition)

| Function | Summary |
|----------|---------|
| `_get_device()` | Returns `'cuda'` or `'cpu'` from PyTorch. |
| `_load_image_cv(path)` | Loads image as RGB (OpenCV BGR→RGB). |
| `_extract_features_batch(image_paths, device, resize)` | Extracts ResNet18 features (512-d) for image paths. |
| `collect_monument_images(dataset_dir, monuments_dir)` | Collects (image_path, monument_label) from folder-per-class dirs. |
| `build_and_train_monument_model(dataset_dir, monuments_dir, model_dir, device)` | Collects images, extracts features, trains LogisticRegression classifier; saves meta, scaler, coef/intercept to `model_dir`. Returns summary dict. |
| `load_monument_model(model_dir)` | Loads meta, scaler, coef, intercept; returns dict with `class_names`, `predict_fn`, or None. |
| `_softmax(x)` | Softmax for logits. |
| `predict_monument(image_path, model_dir, device)` | Predicts monument label for one image; returns (label, confidence). |
| **`run_monument_recognition(frames_dir, model_dir, device, confidence_threshold)`** | **Runs monument model on each image in `frames_dir`; returns `Dict[frame_filename, {label, confidence}]`.** |

---

### 3.9 `face_pipeline/detection.py` (face detection)

| Function | Summary |
|----------|---------|
| `_get_onnx_providers(device)` | Returns ONNX providers (CUDA+CPU if available, else CPU). |
| `load_detector(device, det_size, model_name)` | Creates InsightFace `FaceAnalysis` app; prepares with `det_size`; returns app. |
| `_get_face_attr(face, key, default)` | Gets attribute from face object (bbox, det_score, kps, etc.). |
| **`detect_faces(detector, frame_bgr_or_path, conf_thresh)`** | **Runs detector on one image; filters by conf_thresh; returns list of {bbox, confidence, face_obj, landmarks?}.** |
| `crop_face(frame_bgr, bbox)` | Returns BGR crop of face region. |
| `save_image(img_bgr, path)` | Writes BGR image to path. |

---

### 3.10 `face_pipeline/paths.py`

| Symbol | Summary |
|--------|---------|
| Constants | `KNOWN_FACES_DIR`, `FRAMES_DIR`, `FACES_DIR`, `CROPS_DIR`, `EMBED_DIR`, `FACE_RESULTS_DIR`. |
| `ensure_dirs()` | Creates face-pipeline directories. |

---

### 3.11 `face_pipeline/recognition.py`

| Function | Summary |
|----------|---------|
| `load_known_embeddings(known_dir)` | Loads `.npy` from `known_dir/embeddings/` and optional `labels.json`; returns list of (embedding, label). |
| `cosine_distance(a, b)` | Returns 1 − cosine similarity. |
| `match(embedding, known, thresholds)` | Best match by cosine distance; returns {label, distance, confidence}. |

---

### 3.12 `face_pipeline/embeddings.py`

| Function | Summary |
|----------|---------|
| `get_embedding(face_obj)` | Returns 512-d embedding from InsightFace face object. |
| `save_embedding(path, embedding)` | Saves embedding as `.npy`. |

---

### 3.13 `face_pipeline/register_known.py`

| Function | Summary |
|----------|---------|
| `find_images(folder)` | Recursively finds image files in folder. |
| `register_faces_from_folder(images_dir, label, device, model_name, conf_thresh, ...)` | Detects faces, extracts embeddings, saves to `known_faces/embeddings` and updates `labels.json`. Returns (count, error_message). |
| `main()` | CLI: register faces from `--images-dir`. |

---

## 4. Data flow summary

| Stage | Location / variable | Content |
|-------|---------------------|--------|
| After extract | `frames/<video_id>/` | Raw frames: `frame_0001.jpg`, … |
| After YOLO | `results/<video_id>/processed_frames/` | Annotated frames (same filenames); face boxes and monument overlay added later |
| Face detection | `source_frames_dir` → `detect_faces()` | Raw frame paths; output: `faces_by_frame` + drawing on processed_frames |
| Monument recognition | `processed_frames/` → `run_monument_recognition()` | Frame-level labels; drawn on processed_frames; `monuments_by_frame` |
| Final JSON | `results/<video_id>/detection_results.json` | `frames[i].frame`, `frames[i].detections`, `frames[i].faces`, `frames[i].monument` |

---

## 5. Where "0 faces" or empty results can come from

- **Face:** Import/detector load failure; ONNX providers; confidence threshold; `scan_mode` not "faces"/"both"; cached results; key mismatch between `faces_by_frame` and `results_by_frame`.
- **Object:** YOLO model/device; confidence threshold; no objects in frames.
- **Monument:** No trained model (`load_monument_model` returns None); confidence threshold; no matching class in frames.

---

## 6. Quick checklist

- [ ] Paths and directories created (`ensure_directories`, `ensure_video_results_dirs`).
- [ ] Video downloaded and frames extracted.
- [ ] For faces: no "Face detection skipped" in logs; `scan_mode` is "faces" or "both"; InsightFace/ONNX installed.
- [ ] For monuments: model built via "Build monument model" and `load_monument_model` returns non-None.
- [ ] Force rescan when expecting fresh results.

This document reflects the code flow for **face**, **object**, and **monument** detection; update it when call sites or entry points change.
