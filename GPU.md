# Using Your Graphics Card (GPU) with VISTA

Object detection (YOLO) runs on your **NVIDIA GPU** when PyTorch is installed with CUDA support. If the app is using CPU only, follow the steps below.

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

## 3. NVIDIA driver

- You need a recent **NVIDIA driver** that supports the CUDA version used by PyTorch (e.g. CUDA 12.1).
- Update drivers from [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx) if needed.
- You do **not** need to install the full CUDA Toolkit separately; the PyTorch wheel includes what it needs.

## 4. What the app does

- The detection pipeline calls `_inference_device()` and uses **`cuda`** when `torch.cuda.is_available()` is True, otherwise **`cpu`**.
- Inference is run with `model(frame_path, device=device)`, so when PyTorch has CUDA, the GPU is used.
- The **System** section in the UI and the **Analysis Summary** (after a run) show the **Device** and **Graphics card** that were used.

If you still see CPU after installing the CUDA build, restart the Flask app and run a new job; the first run after a fresh install will use the GPU when available.
