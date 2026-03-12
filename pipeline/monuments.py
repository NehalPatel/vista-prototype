"""Monument recognition: build model from dataset and predict on images/frames.

Uses a pretrained CNN (ResNet18) to extract features, then trains a classifier
on top for monument labels. Dataset: folder-per-class under training_data/dataset/
and training_data/monuments/.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default paths (import from pipeline.paths at runtime to avoid circular import)
_FEATURE_DIM = 512  # ResNet18 penultimate layer
_ALLOWED_EXT = (".jpg", ".jpeg", ".png")
_FEATURE_CACHE_FILENAME = "feature_cache.npz"


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


def _norm_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def _load_feature_cache(cache_path: str) -> Tuple[List[str], Optional[Any]]:
    """Load paths and features from cache file. Returns (paths_list, features_array or None)."""
    import numpy as np

    if not os.path.isfile(cache_path):
        return [], None
    try:
        data = np.load(cache_path, allow_pickle=True)
        paths = [str(p) for p in data["paths"]]
        features = data["features"]
        return paths, features
    except Exception:
        return [], None


def _save_feature_cache(cache_path: str, paths: List[str], features: Any) -> None:
    """Save paths and features to cache file."""
    import numpy as np

    np.savez(
        cache_path,
        paths=np.array(paths, dtype=object),
        features=np.array(features, dtype=np.float32),
    )


def build_and_train_monument_model(
    dataset_dir: str,
    monuments_dir: str,
    model_dir: str,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    clear_feature_cache: bool = False,
) -> Dict[str, Any]:
    """Build feature index from images, train a classifier, save to model_dir.

    Uses a feature cache so only new images are run through ResNet18 on incremental runs.
    Set clear_feature_cache=True for a from-scratch run. Returns summary dict.
    """
    import numpy as np

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            logger.info(msg)

    device = device or _get_device()
    os.makedirs(model_dir, exist_ok=True)
    cache_path = os.path.join(model_dir, _FEATURE_CACHE_FILENAME)

    if clear_feature_cache and os.path.isfile(cache_path):
        try:
            os.remove(cache_path)
        except Exception:
            pass
        _progress("Feature cache cleared.")

    _progress("Collecting images...")
    pairs = collect_monument_images(dataset_dir, monuments_dir)
    if not pairs:
        return {"error": "No images found in dataset or monuments directories", "trained": False}

    paths, labels = zip(*pairs)
    paths = list(paths)
    labels = list(labels)
    class_names = sorted(set(labels))
    n_classes = len(class_names)
    label2idx = {c: i for i, c in enumerate(class_names)}

    cached_paths, cached_features = _load_feature_cache(cache_path)
    path_to_feature: Dict[str, Any] = {}
    if cached_features is not None and len(cached_paths) == cached_features.shape[0]:
        for i, p in enumerate(cached_paths):
            path_to_feature[_norm_path(p)] = cached_features[i]

    paths_to_extract = [p for p in paths if _norm_path(p) not in path_to_feature]
    n_cached = len(paths) - len(paths_to_extract)
    if n_cached:
        _progress(f"Using {n_cached} cached features, extracting {len(paths_to_extract)} new...")
    else:
        _progress(f"Loaded {len(paths)} images, {n_classes} classes. Extracting features...")

    all_features: List[Optional[Any]] = []
    if paths_to_extract:
        batch_size = 32
        n_batches = (len(paths_to_extract) + batch_size - 1) // batch_size
        for i in range(0, len(paths_to_extract), batch_size):
            batch_num = i // batch_size + 1
            _progress(
                f"  Features batch {batch_num}/{n_batches} ({min(i + batch_size, len(paths_to_extract))}/{len(paths_to_extract)} images)"
            )
            batch = paths_to_extract[i : i + batch_size]
            extracted = _extract_features_batch(batch, device)
            for j, p in enumerate(batch):
                if j < len(extracted) and extracted[j] is not None:
                    path_to_feature[_norm_path(p)] = extracted[j]

    for p in paths:
        norm = _norm_path(p)
        if norm in path_to_feature:
            all_features.append(path_to_feature[norm])
        else:
            all_features.append(None)

    # Update cache: only store paths that have valid features
    new_cache_paths = []
    new_cache_features = []
    for p, feat in zip(paths, all_features):
        if feat is not None:
            new_cache_paths.append(_norm_path(p))
            new_cache_features.append(feat)
    _save_feature_cache(cache_path, new_cache_paths, new_cache_features)

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

    _progress("Training classifier...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "scikit-learn required: pip install scikit-learn", "trained": False}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*multi_class.*")
        clf = LogisticRegression(max_iter=1000, random_state=42)
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
