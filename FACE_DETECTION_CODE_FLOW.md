# Face Detection Code Flow – vista-prototype

This document describes **how face detection is wired**: which functions are called, from where, and what each one does. Use it to trace execution and debug when "Total face detections: 0" appears.

---

## 1. High-level flow (when user clicks "Process")

```
Browser (POST /api/process)
    → web/app.py  api_process()
        → download_video()
        → extract_frames()                    → writes raw frames (e.g. frame_0001.jpg)
        → run_yolo()                         → writes annotated frames to processed_frames/
        → run_face_detection()                ← FACE DETECTION ENTRY
        → save_detection_results()           → writes detection_results.json (includes faces)
        → make_video_from_images()
        → return JSON (summary.total_face_detections, etc.)
```

Face detection runs **after** YOLO. It reads **original (raw) frames** for detection and draws face boxes on the **annotated** frames so the output video shows both.

---

## 2. Call chain: where each function is called

| Function | Defined in | Called from |
|----------|------------|-------------|
| `api_process()` | `web/app.py` | Flask route `POST /api/process` (form submit) |
| `download_video()` | `pipeline/video.py` | `web/app.py` inside `api_process()` |
| `extract_frames()` | `pipeline/video.py` | `web/app.py` inside `api_process()` |
| `run_yolo()` | `pipeline/detection.py` | `web/app.py` inside `api_process()` |
| **`run_face_detection()`** | **`pipeline/faces.py`** | **`web/app.py` inside `api_process()`** |
| `load_detector()` | `face_pipeline/detection.py` | `pipeline/faces.py` inside `run_face_detection()` |
| `detect_faces()` | `face_pipeline/detection.py` | `pipeline/faces.py` inside `run_face_detection()` (per frame) |
| `save_detection_results()` | `pipeline/detection.py` | `web/app.py` inside `api_process()` |
| `generate_summary()` | `pipeline/detection.py` | `web/app.py` inside `api_process()` |

---

## 3. File-by-file: what each function does

### 3.1 `web/app.py`

**Route:** `POST /api/process` (body: `url`, `conf_threshold`, `fps`, `face_model`, `scan_start_seconds`, `scan_end_seconds`, etc.)

- **`api_process()`**
  - Validates URL, gets `video_id`, sets up paths via `get_video_results_paths(video_id)`.
  - If not force rescan and results exist: loads `detection_results.json`, returns cached summary (including `total_face_detections` from existing JSON).
  - Otherwise:
    1. **Download:** `video_path = download_video(url, VIDEOS_DIR)` → saves MP4 under `vista-prototype/videos/`.
    2. **Frames:** `saved_frames = extract_frames(video_path, frames_dir_this_video, start_seconds, end_seconds)`  
       - `frames_dir_this_video` = `vista-prototype/frames/<video_id>/`  
       - Writes `frame_0001.jpg`, `frame_0002.jpg`, … (1 frame per second in the scan range).
    3. **YOLO:** `results_by_frame = run_yolo(frames_dir_this_video, paths['processed_frames'], ...)`  
       - Reads raw frames, runs YOLOv8, writes **annotated** images to `paths['processed_frames']` = `vista-prototype/results/<video_id>/processed_frames/`  
       - Returns `Dict[frame_filename, list of object detections]` (e.g. `frame_0001.jpg` → list of `{bbox, class, color, label, conf}`).
    4. **Device:** Sets `device` and `gpu_name` from PyTorch (`torch.cuda.is_available()`). Used only for logging and passed to face detection; **face detection itself uses ONNX providers**, not this `device`, in `face_pipeline/detection.py`.
    5. **Face detection (the critical block):**
       ```python
       face_conf_threshold = float(payload.get("face_conf_threshold", 0.5))
       faces_by_frame = run_face_detection(
           paths["processed_frames"],           # where to draw face boxes (annotated images)
           face_model=face_model_name,         # e.g. "buffalo_l"
           device=device,                      # "cuda" or "cpu" (for API compat; detector uses ONNX providers)
           face_conf_threshold=face_conf_threshold,
           source_frames_dir=frames_dir_this_video,  # run detection on RAW frames
       )
       ```
       - On exception: logs with `exc_info=True` and leaves `faces_by_frame = {}`.
    6. **Total face count:** `total_face_detections = sum(len(v) for v in faces_by_frame.values())`.
    7. **Save:** `save_detection_results(..., faces_by_frame=faces_by_frame)` writes `detection_results.json` with a `frames` array; each frame entry has `frame`, `detections`, and `faces` (list of `{bbox, confidence}`).
    8. **Video:** `make_video_from_images(paths['processed_frames'], ...)` builds the output video from the annotated (and now face-drawn) frames.
  - Response includes `summary.total_face_detections` and result URLs.

So: **face detection is invoked only from `api_process()` in `web/app.py`**, and only when a new run is performed (not when serving cached results).

---

### 3.2 `pipeline/faces.py`

- **`run_face_detection(annotated_frames_dir, face_model="buffalo_l", device="cuda", face_conf_threshold=0.5, source_frames_dir=None)`**
  - **Purpose:** Run InsightFace face detection on each frame, then draw cyan face boxes on the annotated images and return per-frame face lists.
  - **Called from:** `web/app.py` only (see above).
  - **Behavior:**
    1. **Import:** `from face_pipeline.detection import load_detector, detect_faces`. On failure: logs warning and returns `{}` (no faces, no crash).
    2. **Load detector:** `detector = load_detector(device=device, model_name=face_model, det_size=(640, 640))`. On failure: logs warning and returns `{}`.
    3. **Frame loop:**  
       - List dir: `source_frames_dir` if provided and valid, else `annotated_frames_dir`.  
       - For each image file (`fname` like `frame_0001.jpg`):
         - **Detection input:** If `source_frames_dir` is set, read image from `source_frames_dir/fname` (raw frame); else from `annotated_frames_dir/fname`.
         - **Detect:** `dets = detect_faces(detector, img_for_detection, conf_thresh=face_conf_threshold)`.
         - **Store:** `faces_by_frame[fname] = [{"bbox": ..., "confidence": ...}, ...]`.
         - **Draw:** Read annotated image from `annotated_frames_dir/fname`, draw rectangles and "face 0.xx" text, write back with `cv2.imwrite`.
    4. **Return:** `faces_by_frame`: `Dict[frame_filename, list of {bbox, confidence}]`.
  - **Important:** Keys of `faces_by_frame` are **frame filenames** (e.g. `frame_0001.jpg`). These must match the keys of `results_by_frame` from YOLO so that `save_detection_results` can attach `faces` to each frame in the JSON.

---

### 3.3 `face_pipeline/detection.py`

- **`_get_onnx_providers()`**
  - **Purpose:** Choose ONNX execution providers from **actual** availability (not from PyTorch `device`).
  - **Called from:** `load_detector()` only.
  - **Returns:** `["CUDAExecutionProvider", "CPUExecutionProvider"]` if `onnxruntime.get_available_providers()` includes `CUDAExecutionProvider`, else `["CPUExecutionProvider"]`.

- **`load_detector(device="cuda", det_size=(640, 640), model_name="buffalo_l")`**
  - **Purpose:** Create and prepare the InsightFace `FaceAnalysis` app (Buffalo L/S/SC).
  - **Called from:** `pipeline/faces.py` → `run_face_detection()` (once per run).
  - **Behavior:**
    1. Import `insightface.app.FaceAnalysis`; on failure raise.
    2. `providers = _get_onnx_providers()` (ignore `device` for provider choice).
    3. `app = FaceAnalysis(name=model_name, providers=providers)`.
    4. `ctx_id = 0` if CUDA is in providers, else `-1`.
    5. `app.prepare(ctx_id=ctx_id, det_size=det_size)`.
    6. Prints one line: `Face detector ready: model=..., providers=..., det_size=...`.
    7. Returns `app` (used as `detector` in `detect_faces()`).

- **`_get_face_attr(face, key, default=None)`**
  - **Purpose:** Read an attribute from an InsightFace face object whether it is object- or dict-like (e.g. `bbox`, `det_score`, `kps`).
  - **Called from:** `detect_faces()` only.

- **`detect_faces(detector, frame_bgr, conf_thresh=0.5)`**
  - **Purpose:** Run the detector on one BGR image and return a list of face records (bbox, confidence, optional landmarks).
  - **Called from:** `pipeline/faces.py` → `run_face_detection()` **once per frame**.
  - **Behavior:**
    1. `faces = detector.get(frame_bgr)` (InsightFace native call).
    2. For each face: get `det_score` and `bbox` via `_get_face_attr`; skip if score &lt; conf_thresh or bbox missing/invalid.
    3. Clip bbox to image size; optionally add landmarks from `landmark_5` or `kps`.
    4. Return `List[Dict]` with `bbox`, `confidence`, `face_obj`, and optionally `landmarks`.

If **all** faces are filtered out (e.g. wrong provider, bad image, or threshold too high), this returns `[]` and the pipeline reports 0 faces for that frame.

---

### 3.4 `pipeline/detection.py` (relevant to face output)

- **`save_detection_results(..., faces_by_frame=None)`**
  - **Purpose:** Write one JSON file with all object detections and all face detections per frame.
  - **Called from:** `web/app.py` after `run_face_detection()`.
  - **Behavior:** Builds `frames_payload` by iterating `results_by_frame.items()`. For each `frame` (e.g. `frame_0001.jpg`), sets `entry["faces"] = faces_by_frame.get(frame, [])`. So **keys of `faces_by_frame` must match keys of `results_by_frame`** (frame filenames).

---

## 4. Data flow summary

| Stage | Directory / variable | Content |
|-------|----------------------|--------|
| After extract | `vista-prototype/frames/<video_id>/` | Raw frames: `frame_0001.jpg`, … |
| After YOLO | `vista-prototype/results/<video_id>/processed_frames/` | Annotated frames (same filenames) |
| Face detection input | Read from `frames/<video_id>/` (when `source_frames_dir` set) | BGR images for `detector.get()` |
| Face detection output | `faces_by_frame` dict + overwrite in `processed_frames/` | Keys: frame filenames; values: list of `{bbox, confidence}`; images get cyan face boxes |
| Final JSON | `vista-prototype/results/<video_id>/detection_results.json` | `frames[i].frame`, `frames[i].detections`, `frames[i].faces` |

---

## 5. Where "0 faces" can come from

1. **Import / detector load failure**  
   - `run_face_detection` catches and returns `{}`; logs: "Face detection skipped: insightface not available" or "failed to load detector".  
   - Check server logs for these and for the line `Face detector ready: ...` (if missing, detector didn’t get ready).

2. **ONNX providers**  
   - If ONNX only has CPU but code assumed CUDA (or vice versa), `detector.get()` can misbehave or return nothing.  
   - `load_detector` now uses `_get_onnx_providers()` and logs `providers`; confirm in the "Face detector ready" line.

3. **Every frame returns no faces**  
   - `detect_faces()` is called per frame; if `detector.get(frame_bgr)` always returns an empty list or every face is below `conf_thresh`, then `faces_by_frame` has only empty lists and `total_face_detections` is 0.  
   - Optional log in `pipeline/faces.py`: "Face detection ran but found no faces in first frame" when the first frame has 0 detections.

4. **Key mismatch**  
   - If for some reason frame filenames from the face loop differ from `results_by_frame` keys, `save_detection_results` would still write the JSON but could attach faces to the wrong frame or leave them empty. In the current design, both use the same `fname` from the same frame set, so this should match.

5. **Cached results**  
   - If the UI is showing old results, it may be reading a previous `detection_results.json` with 0 faces. Use "Force rescan" to re-run the pipeline and regenerate the JSON.

---

## 6. Quick checklist for debugging

- [ ] Server log shows: `[vista-prototype] Face detector ready: model=buffalo_l, providers=[...], det_size=(640, 640)`.
- [ ] No log line: "Face detection skipped: ..." or "Face detection failed: ...".
- [ ] Environment: `pip install insightface onnxruntime-gpu` (or `onnxruntime` for CPU); same as the working project.
- [ ] Force rescan used so the run is fresh and JSON is rewritten.
- [ ] Video actually contains clear, front-facing faces in the scanned segment.

This document reflects the code as of the last update; if you change call sites or add new entry points, update this flow accordingly.
