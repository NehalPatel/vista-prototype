# Using GPU (CUDA) for inference

The app uses **PyTorch** (YOLO object detection) and **ONNX Runtime** (InsightFace face detection). By default, `pip install` may give you CPU-only builds. To use your NVIDIA GPU:

## 1. PyTorch with CUDA (for YOLO)

Install PyTorch with CUDA from the official index. Pick the CUDA version that matches your driver (e.g. 11.8 or 12.1):

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

See [pytorch.org](https://pytorch.org/get-started/locally/) for your OS/CUDA combination.

## 2. ONNX Runtime GPU (for InsightFace face detection)

Replace the CPU-only ONNX Runtime with the GPU build:

```bash
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

## 3. Check that GPU is used

- **PyTorch:** In Python, `import torch; print(torch.cuda.is_available())` should be `True`, and `torch.cuda.get_device_name(0)` should show your GPU name.
- **ONNX:** With `onnxruntime-gpu` installed, the face pipeline will use `CUDAExecutionProvider` when device is `cuda`.
- **Web app:** After processing a video, the response or run stats should show `"device": "cuda"` and your GPU name when GPU is available.

If you still see CPU usage, ensure your NVIDIA drivers and (if needed) CUDA toolkit match the PyTorch/onnxruntime-gpu versions.
