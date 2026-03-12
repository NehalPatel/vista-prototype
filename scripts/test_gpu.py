#!/usr/bin/env python3
"""Check GPU availability for Vista Prototype: PyTorch, ONNX Runtime, and InsightFace.

Run with the same Python you use for build_models.py (activate your venv/conda first):
  python scripts/test_gpu.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

# #region agent log
DEBUG_LOG = "debug-837bfb.log"
def _dbg(hid: str, location: str, message: str, data: dict) -> None:
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "837bfb", "hypothesisId": hid, "location": location, "message": message, "data": data, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass
# #endregion


def main() -> int:
    print("=" * 60)
    print("GPU / device availability for Vista Prototype")
    print("=" * 60)
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")

    # 0. NVIDIA driver (optional; shows GPU even if Python libs not installed)
    print("\n--- NVIDIA driver (nvidia-smi) ---")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.strip().split("\n"):
                print(f"  {line.strip()}")
        else:
            print("  nvidia-smi not found or no GPU. Install NVIDIA drivers if you have a GPU.")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  nvidia-smi not available: {e}")

    # 1. PyTorch (used by monument model)
    print("\n--- PyTorch ---")
    try:
        import torch  # type: ignore
        print(f"  PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  torch.cuda.is_available(): {cuda_available}")
        if cuda_available:
            print(f"  Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Current device: {torch.cuda.current_device()}")
        else:
            print("  -> Monument model will use CPU. Install PyTorch with CUDA: https://pytorch.org")
    except ImportError as e:
        print(f"  PyTorch not installed: {e}")
        cuda_available = False

    # 2. ONNX Runtime (used by InsightFace for face detection/embeddings)
    print("\n--- ONNX Runtime ---")
    try:
        import onnxruntime as ort  # type: ignore
        ort_version = getattr(ort, "__version__", "?")
        print(f"  onnxruntime version: {ort_version}")
        providers = getattr(ort, "get_available_providers", lambda: [])()
        print(f"  Available providers: {providers}")
        has_cuda = "CUDAExecutionProvider" in providers
        print(f"  CUDAExecutionProvider available: {has_cuda}")
        # #region agent log
        _dbg("H1", "test_gpu.py:ONNX", "onnx_providers", {"available_providers": providers, "has_cuda": has_cuda, "ort_version": ort_version})
        # #endregion
        if not has_cuda:
            print("  -> Face model will use CPU. Install: pip install onnxruntime-gpu")
    except ImportError as e:
        print(f"  onnxruntime not installed: {e}")
        # #region agent log
        _dbg("H1", "test_gpu.py:ONNX", "onnx_import_error", {"error": str(e)})
        # #endregion

    # 3. Quick InsightFace check (what build_models actually uses for faces)
    print("\n--- InsightFace (face model) ---")
    try:
        from insightface.app import FaceAnalysis  # type: ignore
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            import onnxruntime as ort  # type: ignore
            avail = getattr(ort, "get_available_providers", lambda: [])()
            if "CUDAExecutionProvider" not in avail:
                providers = ["CPUExecutionProvider"]
        except Exception:
            providers = ["CPUExecutionProvider"]
        print(f"  Will use providers: {providers}")
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        # #region agent log
        _dbg("H2", "test_gpu.py:InsightFace", "insightface_providers", {"providers": providers, "ctx_id": ctx_id, "using_gpu": ctx_id == 0})
        # #endregion
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("  InsightFace loaded successfully.")
    except Exception as e:
        print(f"  InsightFace load failed: {e}")

    print("\n" + "=" * 60)
    print("Run this script with the same Python/env you use for build_models.py.")
    print("For GPU: PyTorch with CUDA, and/or pip install onnxruntime-gpu (for faces).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
