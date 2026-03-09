# Face Detection Fix Plan: vista-prototype

This document describes changes to apply in **vista-prototype** so that Buffalo L (InsightFace) face detection works reliably, aligned with the **vista-face-recognition** project where face detection is already working (e.g. on [this YouTube video](https://www.youtube.com/watch?v=I-C3cz_Kank)).

---

## Summary of Root Causes

| Area | vista-face-recognition (working) | vista-prototype (no faces) |
|------|----------------------------------|----------------------------|
| **ONNX providers** | Uses `get_onnx_providers()` — checks **actual** ONNX Runtime availability (CUDA vs CPU) | Uses PyTorch `device` to choose providers; if ONNX doesn't have CUDA, this can fail or behave badly |
| **ctx_id** | Always `ctx_id=0` with providers that match (CUDA or CPU) | `ctx_id=0` for cuda, `-1` for cpu, but `device` comes from PyTorch, not from ONNX — can mismatch |
| **det_size** | `(640, 640)` | `(896, 896)` in pipeline/faces.py — can cause issues on some setups |
| **Confidence** | `0.5` (config) | `0.3` in web app — already more permissive; not likely the cause |
| **Dependencies** | `onnxruntime-gpu` + auto-detect | May use `onnxruntime` only or wrong runtime; no explicit provider check |

---

## File-by-File Change Plan

### 1. `face_pipeline/detection.py`

**Goal:** Use ONNX Runtime's actual available providers (like vista-face-recognition) and set `ctx_id` from that, not from PyTorch.

**Changes:**

1. **Add a helper to get ONNX providers by availability (not by device string):**

   ```python
   def _get_onnx_providers() -> List[str]:
       """Return ONNX execution providers: CUDA if available, else CPU (matches vista-face-recognition)."""
       try:
           import onnxruntime as ort
           if "CUDAExecutionProvider" in ort.get_available_providers():
               return ["CUDAExecutionProvider", "CPUExecutionProvider"]
       except Exception:
           pass
       return ["CPUExecutionProvider"]
   ```

2. **In `load_detector`, stop using `device` for provider selection.** Use the new helper and derive `ctx_id` from the chosen providers:

   - Replace the existing `_get_providers(device)` usage with `providers = _get_onnx_providers()`.
   - Set `ctx_id = 0` if `"CUDAExecutionProvider"` is in `providers`, else `ctx_id = -1` (CPU).

   So:

   - `app = FaceAnalysis(name=name, providers=providers)`  
   - `ctx_id = 0 if "CUDAExecutionProvider" in providers else -1`  
   - `app.prepare(ctx_id=ctx_id, det_size=det_size)`

3. **Keep `device` only for API compatibility** (e.g. callers still pass `device="cuda"` or `"cpu"`). You can ignore it for provider/ctx_id, or use it only for logging. The important part is that **InsightFace uses providers and ctx_id that match what ONNX Runtime actually has**.

4. **Optional:** Add a log line after `app.prepare(...)` to print the providers and `det_size` used (e.g. `print("Face detector: providers=%s, det_size=%s" % (providers, det_size))`) so you can confirm in logs.

---

### 2. `pipeline/faces.py`

**Goal:** Use the same detection input size as the working project and avoid failures when the detector loads.

**Changes:**

1. **Use `det_size=(640, 640)`** when calling `load_detector`, to match vista-face-recognition. The working project uses `(640, 640)`; vista-prototype currently uses `(896, 896)`, which can behave differently or cause issues on some environments.

   - Change:  
     `detector = load_detector(device=device, model_name=face_model, det_size=(896, 896))`  
     to  
     `detector = load_detector(device=device, model_name=face_model, det_size=(640, 640))`

2. **Optional:** If you still get no faces, you can try lowering the confidence for debugging (e.g. `face_conf_threshold=0.2`) or add a short log when the first frame is processed (e.g. "Running face detection on frame X, threshold=…").

---

### 3. `web/app.py`

**Goal:** Align face confidence with the working project and make failures visible.

**Changes:**

1. **Use the same face confidence as vista-face-recognition:** pass `face_conf_threshold=0.5` instead of `0.3` when calling `run_face_detection`. The working project uses `DETECTION_CONFIDENCE = 0.5`. You can later make this a payload parameter (e.g. `payload.get('face_conf_threshold', 0.5)`).

2. **Improve error visibility:** In the `except` block around `run_face_detection`, log the full exception (e.g. `logging.getLogger(__name__).warning("Face detection failed: %s", e, exc_info=True)`) so that provider/load errors are visible in logs.

---

### 4. Dependencies (vista-prototype environment)

**Goal:** Use the same runtime as the working project so that CUDA is used when available.

- Install **onnxruntime-gpu** if you have an NVIDIA GPU and CUDA installed:  
  `pip install onnxruntime-gpu`
- If you don't have a GPU or CUDA, use:  
  `pip install onnxruntime`  
  With the new `_get_onnx_providers()` logic, the code will use CPU without requesting CUDA.

Ensure **insightface** is installed:  
`pip install insightface`

---

## Checklist (apply in vista-prototype)

- [ ] **face_pipeline/detection.py**: Add `_get_onnx_providers()`, use it in `load_detector`, and set `ctx_id` from the chosen providers (0 for CUDA, -1 for CPU).
- [ ] **face_pipeline/detection.py**: Optionally log providers and `det_size` after `app.prepare(...)`.
- [ ] **pipeline/faces.py**: Change `det_size` from `(896, 896)` to `(640, 640)` in the `load_detector` call.
- [ ] **web/app.py**: Use `face_conf_threshold=0.5` (or a payload-driven value with default 0.5) when calling `run_face_detection`.
- [ ] **web/app.py**: Log face detection failures with `exc_info=True` (or full traceback) so provider/load errors are visible.
- [ ] **Environment**: Install `insightface` and either `onnxruntime-gpu` (with CUDA) or `onnxruntime` (CPU-only).

---

## Quick reference: working project (vista-face-recognition)

- **Config:** `config.get_onnx_providers()` → `['CUDAExecutionProvider', 'CPUExecutionProvider']` if CUDA is available, else `['CPUExecutionProvider']`.
- **Face detector:** `FaceAnalysis(name='buffalo_l', providers=config.get_onnx_providers())` then `app.prepare(ctx_id=0, det_size=(640, 640))`.
- **Confidence:** `DETECTION_CONFIDENCE = 0.5`.
- **Frames:** BGR from `cv2.imread(frame_path)`; detection on full frame.

After applying these changes in vista-prototype, run the same YouTube video ([e.g.](https://www.youtube.com/watch?v=I-C3cz_Kank)) and check logs for the detector line and any face-detection exception; total face detections should be > 0 when the video contains clear faces.
