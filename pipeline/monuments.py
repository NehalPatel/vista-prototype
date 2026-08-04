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
from pipeline.paths import MONUMENT_POLICY_PATH, MONUMENT_YOLO_CLS_WEIGHTS
from pipeline.utils import canonical_display_name

logger = logging.getLogger(__name__)

# ResNet18 penultimate layer
_FEATURE_DIM = 512
_ALLOWED_EXT = (".jpg", ".jpeg", ".png")
_DEFAULT_BACKEND = "yolo_cls"  # preferred when weights exist; falls back to resnet


def resolve_monument_backend(explicit: Optional[str] = None) -> str:
    """Return 'yolo_cls' or 'resnet'. Default prefers YOLO-cls when best.pt exists."""
    raw = (explicit or os.environ.get("VISTA_MONUMENT_BACKEND") or "").strip().lower()
    if raw in ("yolo_cls", "yolo", "ultralytics"):
        return "yolo_cls"
    if raw in ("resnet", "legacy", "sklearn"):
        return "resnet"
    if os.path.isfile(MONUMENT_YOLO_CLS_WEIGHTS):
        return _DEFAULT_BACKEND
    return "resnet"


def monument_model_available(model_dir: str, backend: Optional[str] = None) -> bool:
    be = resolve_monument_backend(backend)
    if be == "yolo_cls":
        return os.path.isfile(MONUMENT_YOLO_CLS_WEIGHTS)
    return os.path.isfile(os.path.join(model_dir, "meta.json"))


def _feature_cache_basename(preprocess: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in preprocess.strip().lower())
    return f"feature_cache_{safe or 'none'}.npz"


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


def _load_excluded_monument_classes(policy_path: Optional[str]) -> set[str]:
    path = policy_path or MONUMENT_POLICY_PATH
    if not path or not os.path.isfile(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("excluded_classes") or []
        return {str(x).strip().lower() for x in raw if str(x).strip()}
    except Exception:
        logger.warning("Could not read monument policy at %s", path)
        return set()


def _rembg_rgb_from_array(img_rgb: Any) -> Optional[Any]:
    """Return RGB uint8 image with background removed (composited on white)."""
    import numpy as np
    import cv2  # type: ignore

    if img_rgb is None:
        return None
    try:
        from rembg import remove  # type: ignore
    except ImportError:
        logger.warning("rembg not installed; skip rembg preprocess (pip install -r requirements-monuments-extra.txt)")
        return None
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return None
    try:
        out_bytes = remove(buf.tobytes())
    except Exception as exc:
        logger.warning("rembg failed: %s", exc)
        return None
    arr = np.frombuffer(out_bytes, dtype=np.uint8)
    dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if dec is None:
        return None
    if dec.shape[2] == 4:
        b, g, r, a = cv2.split(dec)
        a_f = a.astype(np.float32) / 255.0
        rgb = np.stack([r, g, b], axis=2).astype(np.float32)
        bg = np.full_like(rgb, 255.0)
        comp = rgb * a_f[..., np.newaxis] + bg * (1.0 - a_f[..., np.newaxis])
        return np.clip(comp, 0, 255).astype(np.uint8)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def _extract_features_batch(
    image_paths: List[str],
    device: str,
    resize: Tuple[int, int] = (224, 224),
    preprocess: str = "none",
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

    use_rembg = preprocess == "rembg"
    features = []
    for path in image_paths:
        img = _load_image_cv(path)
        if img is None:
            features.append(None)
            continue
        if use_rembg:
            img = _rembg_rgb_from_array(img)
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
    policy_path: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Collect (image_path, monument_label) from dataset/ (folder=class) and monuments/ (folder=name).

    Skips folder names listed under excluded_classes in monument_policy.json (see MONUMENT_POLICY_PATH).
    """
    pairs: List[Tuple[str, str]] = []
    excluded = _load_excluded_monument_classes(policy_path)

    for base_dir in (dataset_dir, monuments_dir):
        if not os.path.isdir(base_dir):
            continue
        for name in sorted(os.listdir(base_dir)):
            folder = os.path.join(base_dir, name)
            if not os.path.isdir(folder):
                continue
            if name.strip().lower() in excluded:
                continue
            display_name = canonical_display_name(name)
            if display_name.strip().lower() in excluded:
                continue
            for ext in _ALLOWED_EXT:
                for path in glob(os.path.join(folder, "*" + ext)):
                    if os.path.isfile(path):
                        pairs.append((path, display_name))

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
    policy_path: Optional[str] = None,
    preprocess: str = "none",
    class_weight_balanced: bool = False,
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
    preprocess = preprocess if preprocess in ("none", "rembg") else "none"
    pol = policy_path if policy_path is not None else MONUMENT_POLICY_PATH
    cache_path = os.path.join(model_dir, _feature_cache_basename(preprocess))

    if clear_feature_cache:
        removed_any = False
        for fn in glob(os.path.join(model_dir, "feature_cache*.npz")):
            try:
                os.remove(fn)
                removed_any = True
            except OSError:
                pass
        if removed_any:
            _progress("Feature cache cleared.")

    _progress("Collecting images...")
    pairs = collect_monument_images(dataset_dir, monuments_dir, policy_path=pol)
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
            extracted = _extract_features_batch(batch, device, preprocess=preprocess)
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
        lr_kw: Dict[str, Any] = {"max_iter": 1000, "random_state": 42}
        if class_weight_balanced:
            lr_kw["class_weight"] = "balanced"
        clf = LogisticRegression(**lr_kw)
        clf.fit(X_scaled, y)

    # Save: class names, scaler params, classifier coeffs
    meta = {
        "class_names": class_names,
        "n_classes": n_classes,
        "feature_dim": X.shape[1],
        "preprocess": preprocess,
        "class_weight_balanced": bool(class_weight_balanced),
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
    meta.setdefault("preprocess", "none")
    meta.setdefault("class_weight_balanced", False)
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

    def predict_with_margin(
        features: np.ndarray,
    ) -> Tuple[List[str], List[float], List[float]]:
        """Return (labels, top1_conf, top1-top2 margin) per row."""
        x = (features - mean) / (scale + 1e-8)
        logits = x @ coef.T + intercept
        probs = _softmax(logits)
        # sort descending per row
        order = np.argsort(-probs, axis=1)
        top1 = order[:, 0]
        top2 = order[:, 1] if probs.shape[1] > 1 else order[:, 0]
        labels = [meta["class_names"][i] for i in top1]
        confs = [float(probs[i, top1[i]]) for i in range(len(top1))]
        margins = [
            float(probs[i, top1[i]] - probs[i, top2[i]]) if probs.shape[1] > 1 else float(probs[i, top1[i]])
            for i in range(len(top1))
        ]
        return labels, confs, margins

    meta["predict_fn"] = predict
    meta["predict_with_margin_fn"] = predict_with_margin
    meta["_coef"] = coef
    meta["_intercept"] = intercept
    meta["_mean"] = mean
    meta["_scale"] = scale
    return meta


def _softmax(x: "np.ndarray") -> "np.ndarray":
    import numpy as np
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _person_stats_from_detections(dets: List[Dict[str, Any]], frame_w: int, frame_h: int) -> Tuple[int, float]:
    """Count YOLO person boxes and their total area ratio from detection dicts."""
    img_area = float(max(1, frame_w * frame_h))
    count = 0
    area = 0.0
    for d in dets or []:
        cls = str(d.get("class") or "").lower()
        if cls != "person":
            continue
        bbox = d.get("bbox") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        count += 1
        area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return count, float(area / img_area)


def _frame_should_skip_monument(
    dets: Optional[List[Dict[str, Any]]],
    frame_path: str,
    max_person_count: int,
    max_person_area_ratio: float,
) -> Tuple[bool, str]:
    """Heuristic gate: skip frames dominated by people (street/crowd) for monument ID."""
    if dets is None:
        return False, ""
    try:
        import cv2  # type: ignore
        img = cv2.imread(frame_path)
        if img is None:
            h, w = 1, 1
        else:
            h, w = img.shape[:2]
    except Exception:
        h, w = 1, 1
    n_person, person_ratio = _person_stats_from_detections(dets, w, h)
    if n_person > max_person_count:
        return True, "too_many_people"
    if person_ratio > max_person_area_ratio:
        return True, "person_dominant"
    return False, ""


def predict_monument(
    image_path: str,
    model_dir: str,
    device: Optional[str] = None,
    confidence_threshold: float = 0.75,
    margin_threshold: float = 0.15,
    backend: Optional[str] = None,
) -> Tuple[Optional[str], float]:
    """Predict monument label for one image. Returns (label, confidence) or (None, 0.0).

    Rejects low-confidence or ambiguous (small top-1 vs top-2 margin) predictions as Unknown.
    """
    be = resolve_monument_backend(backend)
    if be == "yolo_cls":
        out = _predict_yolo_cls_one(
            image_path,
            device or _get_device(),
            confidence_threshold,
            margin_threshold,
        )
        if out is None:
            return None, 0.0
        return out["label"], float(out["confidence"])

    import numpy as np

    model = load_monument_model(model_dir)
    if model is None:
        return None, 0.0
    device = device or _get_device()
    preprocess = str(model.get("preprocess", "none"))
    feats = _extract_features_batch([image_path], device, preprocess=preprocess)
    if not feats or feats[0] is None:
        return None, 0.0
    X = np.array([feats[0]], dtype=np.float32)
    predict_m = model.get("predict_with_margin_fn") or model["predict_fn"]
    if predict_m is model["predict_fn"]:
        labels, confs = predict_m(X)
        margins = [1.0]
    else:
        labels, confs, margins = predict_m(X)
    label, conf, margin = labels[0], confs[0], margins[0]
    if conf < confidence_threshold or margin < margin_threshold:
        return "Unknown", float(conf)
    return label, float(conf)


def _yolo_device_arg(device: Optional[str]) -> str:
    if not device or device == "cuda":
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def _predict_yolo_cls_one(
    image_path: str,
    device: str,
    confidence_threshold: float,
    margin_threshold: float,
    model: Any = None,
) -> Optional[Dict[str, Any]]:
    import numpy as np

    if not os.path.isfile(MONUMENT_YOLO_CLS_WEIGHTS) and model is None:
        return None
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        logger.warning("ultralytics not available for yolo_cls backend: %s", exc)
        return None

    if model is None:
        model = YOLO(MONUMENT_YOLO_CLS_WEIGHTS)
    res = model.predict(image_path, verbose=False, device=_yolo_device_arg(device))
    if not res:
        return {"label": "Unknown", "confidence": 0.0, "reject_reason": "predict_empty"}
    r0 = res[0]
    probs = getattr(r0, "probs", None)
    if probs is None:
        return {"label": "Unknown", "confidence": 0.0, "reject_reason": "no_probs"}
    top1 = int(probs.top1)
    names = r0.names if isinstance(r0.names, dict) else {i: n for i, n in enumerate(r0.names)}
    label = str(names.get(top1, top1))
    conf = float(probs.top1conf)
    data = probs.data.cpu().numpy() if hasattr(probs.data, "cpu") else np.asarray(probs.data)
    arr = np.asarray(data).reshape(-1)
    order = np.argsort(-arr)
    margin = float(arr[order[0]] - arr[order[1]]) if len(order) > 1 else float(arr[order[0]])
    if conf >= confidence_threshold and margin >= margin_threshold:
        return {"label": label, "confidence": conf, "margin": margin}
    reason = "low_confidence" if conf < confidence_threshold else "ambiguous_margin"
    return {
        "label": "Unknown",
        "confidence": conf,
        "margin": margin,
        "reject_reason": reason,
    }


def run_monument_recognition(
    frames_dir: str,
    model_dir: str,
    device: Optional[str] = None,
    confidence_threshold: float = 0.75,
    margin_threshold: float = 0.15,
    detections_by_frame: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    max_person_count: int = 3,
    max_person_area_ratio: float = 0.25,
    backend: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run monument recognition on frames. Returns { frame_filename: { label, confidence } }.

    Frames dominated by people are skipped (Unknown). Predictions below confidence_threshold
    or with top-1/top-2 margin below margin_threshold become Unknown.

    backend: 'yolo_cls' (default when weights exist) or 'resnet' (legacy ResNet18+LR).
    """
    be = resolve_monument_backend(backend)
    if be == "yolo_cls":
        return _run_monument_yolo_cls(
            frames_dir=frames_dir,
            device=device or _get_device(),
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            detections_by_frame=detections_by_frame,
            max_person_count=max_person_count,
            max_person_area_ratio=max_person_area_ratio,
        )
    return _run_monument_resnet(
        frames_dir=frames_dir,
        model_dir=model_dir,
        device=device or _get_device(),
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        detections_by_frame=detections_by_frame,
        max_person_count=max_person_count,
        max_person_area_ratio=max_person_area_ratio,
    )


def _run_monument_yolo_cls(
    frames_dir: str,
    device: str,
    confidence_threshold: float,
    margin_threshold: float,
    detections_by_frame: Optional[Dict[str, List[Dict[str, Any]]]],
    max_person_count: int,
    max_person_area_ratio: float,
) -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(MONUMENT_YOLO_CLS_WEIGHTS):
        logger.warning("YOLO-cls weights missing at %s", MONUMENT_YOLO_CLS_WEIGHTS)
        return {}
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        logger.warning("ultralytics import failed: %s", exc)
        return {}

    model = YOLO(MONUMENT_YOLO_CLS_WEIGHTS)
    results: Dict[str, Dict[str, Any]] = {}
    frame_files = [
        f for f in sorted(os.listdir(frames_dir))
        if f.lower().endswith(_ALLOWED_EXT)
    ]
    for name in frame_files:
        path = os.path.join(frames_dir, name)
        dets = None if detections_by_frame is None else detections_by_frame.get(name)
        skip, reason = _frame_should_skip_monument(
            dets, path, max_person_count, max_person_area_ratio
        )
        if skip:
            results[name] = {"label": "Unknown", "confidence": 0.0, "reject_reason": reason}
            continue
        out = _predict_yolo_cls_one(
            path, device, confidence_threshold, margin_threshold, model=model
        )
        results[name] = out or {
            "label": "Unknown",
            "confidence": 0.0,
            "reject_reason": "yolo_predict_failed",
        }
    return results


def _run_monument_resnet(
    frames_dir: str,
    model_dir: str,
    device: str,
    confidence_threshold: float,
    margin_threshold: float,
    detections_by_frame: Optional[Dict[str, List[Dict[str, Any]]]],
    max_person_count: int,
    max_person_area_ratio: float,
) -> Dict[str, Dict[str, Any]]:
    """Legacy ResNet18 + LogisticRegression path."""
    import numpy as np

    model = load_monument_model(model_dir)
    if model is None:
        return {}

    preprocess = str(model.get("preprocess", "none"))
    results: Dict[str, Dict[str, Any]] = {}
    frame_files = [
        f for f in sorted(os.listdir(frames_dir))
        if f.lower().endswith(_ALLOWED_EXT)
    ]
    if not frame_files:
        return results

    predict_m = model.get("predict_with_margin_fn")
    paths = [os.path.join(frames_dir, f) for f in frame_files]
    batch_size = 16
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_names = frame_files[i : i + batch_size]

        to_extract_paths: List[str] = []
        to_extract_names: List[str] = []
        for p, name in zip(batch_paths, batch_names):
            dets = None
            if detections_by_frame is not None:
                dets = detections_by_frame.get(name)
            skip, reason = _frame_should_skip_monument(
                dets, p, max_person_count, max_person_area_ratio
            )
            if skip:
                results[name] = {
                    "label": "Unknown",
                    "confidence": 0.0,
                    "reject_reason": reason,
                }
            else:
                to_extract_paths.append(p)
                to_extract_names.append(name)

        if not to_extract_paths:
            continue

        feats = _extract_features_batch(to_extract_paths, device, preprocess=preprocess)
        valid = []
        valid_names = []
        for j, f in enumerate(feats):
            if f is not None:
                valid.append(f)
                valid_names.append(to_extract_names[j])
            else:
                results[to_extract_names[j]] = {
                    "label": "Unknown",
                    "confidence": 0.0,
                    "reject_reason": "feature_extract_failed",
                }
        if not valid:
            continue
        X = np.array(valid, dtype=np.float32)
        if predict_m is not None:
            labels, confs, margins = predict_m(X)
        else:
            labels, confs = model["predict_fn"](X)
            margins = [1.0] * len(labels)
        for name, label, conf, margin in zip(valid_names, labels, confs, margins):
            if conf >= confidence_threshold and margin >= margin_threshold:
                results[name] = {
                    "label": label,
                    "confidence": float(conf),
                    "margin": float(margin),
                }
            else:
                reason = "low_confidence" if conf < confidence_threshold else "ambiguous_margin"
                results[name] = {
                    "label": "Unknown",
                    "confidence": float(conf),
                    "margin": float(margin),
                    "reject_reason": reason,
                }

    return results
