# Scripts

CLI tools for organizing training data and building models (no web UI).

## Directory layout (under `vista-prototype/training_data/`)

- **`faces/<name>/`** – One folder per person; images inside. Used for face recognition.
- **`monuments/<name>/`** – One folder per monument; images inside. Used for monument classifier.
- **`inbox_faces/<name>/`** – Drop unorganized face images here (one subfolder per person), then run organize.
- **`inbox_monuments/<name>/`** – Drop unorganized monument images here (one subfolder per monument), then run organize.
- **`dataset/`** – Optional; e.g. Kaggle download with folder-per-class. Also used for monument training.

All of these are in `.gitignore` so images are not pushed to GitHub.

## 1. Organize images

After downloading datasets or adding unorganized images, put them in inbox folders **with one subfolder per identity/monument**:

```
vista-prototype/training_data/inbox_faces/
  Celebrity_A/   <- all images of person A
  Celebrity_B/
vista-prototype/training_data/inbox_monuments/
  Taj_Mahal/
  Eiffel_Tower/
```

Then from repo root:

```bash
python scripts/organize_training_data.py
```

This **moves** images from inbox into `faces/` and `monuments/` (sanitized folder names). Use `--faces-only` or `--monuments-only` to do one kind. Use `--dry-run` to see what would be done.

## 2. Build models

From repo root:

```bash
# Build both face and monument models
python scripts/build_models.py

# Only face recognition (from training_data/faces/)
python scripts/build_models.py --faces-only

# Only monument classifier (from training_data/monuments/ and training_data/dataset/)
python scripts/build_models.py --monuments-only

# Force GPU (or omit to auto-detect per backend)
python scripts/build_models.py --device cuda
```

### GPU not being used?

Run the GPU test with the **same Python environment** you use for the app (activate your venv/conda first):

```bash
python scripts/test_gpu.py
```

It reports NVIDIA driver, PyTorch CUDA, and ONNX Runtime (InsightFace). For GPU:

- **Faces**: `pip install onnxruntime-gpu` (replaces CPU-only `onnxruntime`). If you see **`cublasLt64_12.dll` missing**, see [docs/GPU.md](../docs/GPU.md) (Option A: use CUDA 11.8 build to match PyTorch; Option B: install CUDA 12 Toolkit and add to PATH).
- **Monuments**: Install PyTorch with CUDA from [pytorch.org](https://pytorch.org) (e.g. CUDA 11.8 or 12.x).

- **Face model**: writes to `vista-prototype/known_faces/` (embeddings + labels). Used by video processing for face recognition.
- **Monument model**: writes to `vista-prototype/monument_model/`. Used by video processing for monument labels on frames.

You can keep adding images to `faces/` and `monuments/` and re-run `build_models.py` to rebuild.

### List low-resolution face persons (JSON report)

Get a JSON list of person names whose folder has any low-resolution image (one low-res → whole folder is treated as low-res; scan stops per person so it’s fast). Use the names to search for better images.

```bash
python scripts/list_low_resolution_faces.py
python scripts/list_low_resolution_faces.py -o report.json
```

Output: `{"low_resolution_persons": ["Name1", "Name2", ...]}`. Default: file size &lt; 20 KB counts as low-res (`--size-kb` to change).

### Option B: Cropped-face fallback (test script)

If you have a dataset of **small cropped face images** (e.g. ~4KB) that the normal detector misses, you can try the duplicate script that upscales small images before detection:

```bash
python scripts/build_models_cropped_faces.py --faces-only
python scripts/build_models_cropped_faces.py --full --faces-only
```

Same args as `build_models.py`; it writes to the same `known_faces/` and MongoDB. Images with max dimension &lt; 256px are upscaled to 640×640 so the detector can find the face. Use this only for testing; prefer a dataset with larger face crops for production.
