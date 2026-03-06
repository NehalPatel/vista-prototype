"""Monument recognition: build model from dataset and predict on images/frames.

Uses a pretrained CNN (ResNet18) to extract features, then trains a classifier
on top for monument labels. Dataset: folder-per-class under training_data/dataset/
and training_data/monuments/.
"""

from __future__ import annotations

import json
import logging
import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default paths (import from pipeline.paths at runtime to avoid circular import)
_FEATURE_DIM = 512  # ResNet18 penultimate layer
_ALLOWED_EXT = (".jpg", ".jpeg", ".png")


def _get_device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_image_cv(path: str):
    import cv2  # type: ignore
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _extract_features_batch(
    image_paths: List[str],
    device: str,
    resize: Tuple[int, int] = (224, 224),
) -> List[Optional[Any]]:
    """Extract ResNet18 features (no final FC) for a list of image paths. Returns list of 512-d vectors or None."""
    import numpy as np
    import torch  # type: ignore
    import torchvision.transforms as T  # type: ignore
    from torchvision.models import resnet18, ResNet18_Weights  # type: ignore

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(resize),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    for path in image_paths:
        img = _load_image_cv(path)
        if img is None:
            features.append(None)
            continue
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model(t)
        features.append(f.cpu().numpy().flatten())
    return features


def collect_monument_images(
    dataset_dir: str,
    monuments_dir: str,
) -> List[Tuple[str, str]]:
    """Collect (image_path, monument_label) from dataset/ (folder=class) and monuments/ (folder=name)."""
    pairs: List[Tuple[str, str]] = []

    for base_dir in (dataset_dir, monuments_dir):
        if not os.path.isdir(base_dir):
            continue
        for name in sorted(os.listdir(base_dir)):
            folder = os.path.join(base_dir, name)
            if not os.path.isdir(folder):
                continue
            for ext in _ALLOWED_EXT:
                for path in glob(os.path.join(folder, "*" + ext)):
                    if os.path.isfile(path):
                        pairs.append((path, name))

    return pairs


def build_and_train_monument_model(
    dataset_dir: str,
    monuments_dir: str,
    model_dir: str,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Build feature index from images, train a classifier, save to model_dir. Returns summary dict."""
    import numpy as np

    device = device or _get_device()
    os.makedirs(model_dir, exist_ok=True)

    pairs = collect_monument_images(dataset_dir, monuments_dir)
    if not pairs:
        return {"error": "No images found in dataset or monuments directories", "trained": False}

    paths, labels = zip(*pairs)
    paths = list(paths)
    labels = list(labels)
    class_names = sorted(set(labels))
    n_classes = len(class_names)
    label2idx = {c: i for i, c in enumerate(class_names)}

    # Extract features in chunks to avoid OOM
    batch_size = 32
    all_features: List[Optional[Any]] = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        all_features.extend(_extract_features_batch(batch, device))

    X_list = []
    y_list = []
    for feat, label in zip(all_features, labels):
        if feat is not None:
            X_list.append(feat)
            y_list.append(label2idx[label])
    if not X_list:
        return {"error": "No valid features extracted from images", "trained": False}

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Train classifier: use sklearn LogisticRegression or a simple linear layer
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "scikit-learn required: pip install scikit-learn", "trained": False}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", random_state=42)
    clf.fit(X_scaled, y)

    # Save: class names, scaler params, classifier coeffs
    meta = {
        "class_names": class_names,
        "n_classes": n_classes,
        "feature_dim": X.shape[1],
    }
    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    np.save(os.path.join(model_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(model_dir, "scaler_scale.npy"), scaler.scale_)
    np.save(os.path.join(model_dir, "coef.npy"), clf.coef_)
    np.save(os.path.join(model_dir, "intercept.npy"), clf.intercept_)

    return {
        "trained": True,
        "n_samples": len(X_list),
        "n_classes": n_classes,
        "class_names": class_names,
        "model_dir": model_dir,
    }


def load_monument_model(model_dir: str) -> Optional[Dict[str, Any]]:
    """Load meta, scaler, and classifier params. Returns dict with class_names, predict_fn, or None."""
    import numpy as np

    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    coef = np.load(os.path.join(model_dir, "coef.npy"))
    intercept = np.load(os.path.join(model_dir, "intercept.npy"))
    mean = np.load(os.path.join(model_dir, "scaler_mean.npy"))
    scale = np.load(os.path.join(model_dir, "scaler_scale.npy"))

    def predict(features: np.ndarray) -> Tuple[List[str], List[float]]:
        # features: (N, D)
        x = (features - mean) / (scale + 1e-8)
        logits = x @ coef.T + intercept
        probs = _softmax(logits)
        pred_idx = np.argmax(probs, axis=1)
        labels = [meta["class_names"][i] for i in pred_idx]
        confs = [float(probs[i, pred_idx[i]]) for i in range(len(pred_idx))]
        return labels, confs

    meta["predict_fn"] = predict
    meta["_coef"] = coef
    meta["_intercept"] = intercept
    meta["_mean"] = mean
    meta["_scale"] = scale
    return meta


def _softmax(x: "np.ndarray") -> "np.ndarray":
    import numpy as np
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_monument(
    image_path: str,
    model_dir: str,
    device: Optional[str] = None,
) -> Tuple[Optional[str], float]:
    """Predict monument label for one image. Returns (label, confidence) or (None, 0.0)."""
    import numpy as np

    model = load_monument_model(model_dir)
    if model is None:
        return None, 0.0
    device = device or _get_device()
    feats = _extract_features_batch([image_path], device)
    if not feats or feats[0] is None:
        return None, 0.0
    X = np.array([feats[0]], dtype=np.float32)
    labels, confs = model["predict_fn"](X)
    return labels[0], confs[0]


def run_monument_recognition(
    frames_dir: str,
    model_dir: str,
    device: Optional[str] = None,
    confidence_threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Run monument recognition on each image in frames_dir. Returns { frame_filename: { label, confidence } }."""
    import numpy as np

    model = load_monument_model(model_dir)
    if model is None:
        return {}

    device = device or _get_device()
    results = {}
    frame_files = [
        f for f in sorted(os.listdir(frames_dir))
        if f.lower().endswith(_ALLOWED_EXT)
    ]
    if not frame_files:
        return results

    paths = [os.path.join(frames_dir, f) for f in frame_files]
    batch_size = 16
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_names = frame_files[i : i + batch_size]
        feats = _extract_features_batch(batch_paths, device)
        valid = []
        valid_names = []
        for j, f in enumerate(feats):
            if f is not None:
                valid.append(f)
                valid_names.append(batch_names[j])
        if not valid:
            continue
        X = np.array(valid, dtype=np.float32)
        labels, confs = model["predict_fn"](X)
        for name, label, conf in zip(valid_names, labels, confs):
            if conf >= confidence_threshold:
                results[name] = {"label": label, "confidence": float(conf)}
            else:
                results[name] = {"label": "Unknown", "confidence": float(conf)}

    return results
