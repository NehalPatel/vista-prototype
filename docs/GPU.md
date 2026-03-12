# Using Your Graphics Card (GPU) with VISTA

Object detection (YOLO) and **face recognition (InsightFace)** can use your **NVIDIA GPU** when the right packages and, for the face model, CUDA libraries are installed. If the app is using CPU only, follow the steps below.

## 1. Check if PyTorch sees your GPU

In your project virtual environment, run:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

- If you see **`CUDA available: True`** and a GPU name, PyTorch is already using your graphics card. The app will use it automatically; no code change needed.
- If you see **`CUDA available: False`**, PyTorch is **CPU-only**. You need to install a CUDA build (step 2).

## 2. Install PyTorch with CUDA (Windows)

The default `pip install torch` often installs the **CPU-only** build. To use your NVIDIA GPU:

1. **Uninstall current PyTorch** (in your project venv):
   ```bash
   pip uninstall torch torchvision -y
   ```

2. **Install PyTorch with CUDA** from [pytorch.org](https://pytorch.org/get-started/locally/):
   - Select: **Windows** → **pip** → **Python** → **CUDA 12.1** (or the version that matches your driver; 12.x is common).
   - Run the command they give, for example:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   For **CUDA 11.8**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Confirm again**:
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
   ```

## 3. Face model (ONNX Runtime GPU) and `cublasLt64_12.dll` missing

If you see an error like **`onnxruntime_providers_cuda.dll`** or **`cublasLt64_12.dll` not found**, the face pipeline is trying to use the GPU but the CUDA 12 libraries are not on your system.

**Option A – Use CUDA 11.8 build (matches PyTorch, no Toolkit needed)**  
If you already have PyTorch with CUDA 11.8 (e.g. `torch 2.x+cu118`), install the ONNX Runtime GPU build that uses CUDA 11.8 so it can reuse PyTorch’s DLLs:

```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install coloredlogs flatbuffers numpy packaging protobuf sympy
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

Then ensure **PyTorch is imported before any ONNX Runtime session** (the app and `scripts/build_models.py` do this when both are used). Run `python scripts/test_gpu.py` to confirm `CUDAExecutionProvider` is available.

**Option B – Use default CUDA 12 build and install CUDA Toolkit**  
If you prefer the default `pip install onnxruntime-gpu` (CUDA 12.x):

1. Install **CUDA Toolkit 12.x** from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (e.g. 12.2 or 12.4).
2. Install **cuDNN** for CUDA 12 from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn).
3. Add to your **PATH** (adjust version if needed):
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
   - The cuDNN `bin` folder (e.g. where `cudnn64_8.dll` or similar lives).

After that, `cublasLt64_12.dll` and other CUDA 12 DLLs will be found and the face model can use the GPU.

## 4. NVIDIA driver

- You need a recent **NVIDIA driver** that supports the CUDA version used by PyTorch (e.g. CUDA 12.1) and/or ONNX Runtime.
- Update drivers from [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx) if needed.
- For **PyTorch only**, you do **not** need to install the full CUDA Toolkit; the PyTorch wheel includes what it needs. For **onnxruntime-gpu** with the default (CUDA 12) pip package, you need the CUDA 12 Toolkit (or use Option A above).

## 5. What the app does

- The detection pipeline calls `_inference_device()` and uses **`cuda`** when `torch.cuda.is_available()` is True, otherwise **`cpu`**.
- Inference is run with `model(frame_path, device=device)`, so when PyTorch has CUDA, the GPU is used.
- The **System** section in the UI and the **Analysis Summary** (after a run) show the **Device** and **Graphics card** that were used.

If you still see CPU after installing the CUDA build, restart the Flask app and run a new job; the first run after a fresh install will use the GPU when available.
