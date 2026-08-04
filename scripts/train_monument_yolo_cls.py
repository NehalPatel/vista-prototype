#!/usr/bin/env python3
"""Train / evaluate Ultralytics YOLO classification for monuments (experiment).

Builds a YOLO-cls folder layout from the persisted train/val split, trains yolov8n-cls
(or another --model), evaluates on val, and writes a comparison-friendly summary JSON.

Example:
  python scripts/train_monument_yolo_cls.py --epochs 30 --imgsz 224
  python scripts/train_monument_yolo_cls.py --eval-only --weights experiments/monument_v2/yolo_cls/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.monuments import collect_monument_images  # noqa: E402
from pipeline.paths import (  # noqa: E402
    MONUMENT_POLICY_PATH,
    MONUMENT_SPLITS_DIR,
    TRAINING_DATASET_DIR,
    TRAINING_DATA_DIR,
    TRAINING_MONUMENTS_DIR,
    VISTA_DIR,
)

EXPERIMENT_ROOT = os.path.join(REPO_ROOT, "experiments", "monument_v2")
YOLO_DATA_ROOT = os.path.join(EXPERIMENT_ROOT, "yolo_cls_dataset")
YOLO_RUN_ROOT = os.path.join(EXPERIMENT_ROOT, "yolo_cls")


def _norm(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def _default_split() -> str:
    return os.path.join(MONUMENT_SPLITS_DIR, "train_val_split.json")


def _link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_yolo_dataset(split_path: str, out_root: str) -> dict:
    pairs = collect_monument_images(
        TRAINING_DATASET_DIR, TRAINING_MONUMENTS_DIR, policy_path=MONUMENT_POLICY_PATH
    )
    path_to_label = {_norm(p): lab for p, lab in pairs}
    with open(split_path, "r", encoding="utf-8") as f:
        sp = json.load(f)
    train_paths = [_norm(p) for p in sp.get("train", []) if _norm(p) in path_to_label]
    val_paths = [_norm(p) for p in sp.get("val", []) if _norm(p) in path_to_label]
    # New images not in split → train
    known = set(train_paths) | set(val_paths)
    for p, lab in path_to_label.items():
        if p not in known:
            train_paths.append(p)

    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
    counts = {"train": defaultdict(int), "val": defaultdict(int)}
    for split_name, paths in (("train", train_paths), ("val", val_paths)):
        for p in paths:
            lab = path_to_label[p]
            ext = os.path.splitext(p)[1].lower() or ".jpg"
            base = os.path.splitext(os.path.basename(p))[0]
            # Avoid collisions across sources
            digest = abs(hash(p)) % (10**8)
            dst = os.path.join(out_root, split_name, lab, f"{base}_{digest}{ext}")
            _link_or_copy(p, dst)
            counts[split_name][lab] += 1

    meta = {
        "split_path": split_path,
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "classes": sorted(set(path_to_label.values())),
        "counts": {k: dict(v) for k, v in counts.items()},
    }
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def eval_yolo_weights(weights: str, data_root: str, device: str) -> dict:
    from ultralytics import YOLO
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    model = YOLO(weights)
    val_root = os.path.join(data_root, "val")
    y_true: list[str] = []
    y_pred: list[str] = []
    confs: list[float] = []
    for cls_name in sorted(os.listdir(val_root)):
        cls_dir = os.path.join(val_root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            path = os.path.join(cls_dir, fn)
            if not os.path.isfile(path):
                continue
            res = model.predict(path, verbose=False, device=device)
            if not res:
                continue
            r0 = res[0]
            # Classification result
            probs = getattr(r0, "probs", None)
            if probs is None:
                continue
            top1 = int(probs.top1)
            names = r0.names if isinstance(r0.names, dict) else {i: n for i, n in enumerate(r0.names)}
            pred = names.get(top1, str(top1))
            conf = float(probs.top1conf) if hasattr(probs, "top1conf") else float(probs.data[top1])
            y_true.append(cls_name)
            y_pred.append(pred)
            confs.append(conf)

    labels = sorted(set(y_true) | set(y_pred))
    label_to_idx = {c: i for i, c in enumerate(labels)}
    yt = np.array([label_to_idx[y] for y in y_true])
    yp = np.array([label_to_idx[y] for y in y_pred])
    acc = float(accuracy_score(yt, yp)) if len(yt) else 0.0
    report = classification_report(
        yt, yp, labels=list(range(len(labels))), target_names=labels, zero_division=0
    )
    cm = confusion_matrix(yt, yp, labels=list(range(len(labels))))
    return {
        "accuracy": acc,
        "n_val_scored": len(y_true),
        "mean_top1_conf": float(sum(confs) / max(1, len(confs))),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Ultralytics cls checkpoint")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--weights", default=None, help="For --eval-only or resume")
    parser.add_argument("--skip-rebuild-dataset", action="store_true")
    args = parser.parse_args()

    split_path = os.path.abspath(args.split_file or _default_split())
    if not os.path.isfile(split_path):
        print(f"Split not found: {split_path}. Run evaluate_monument_classifier.py --create-split", file=sys.stderr)
        return 1

    device = args.device
    if device is None:
        try:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
    if not args.skip_rebuild_dataset and not args.eval_only:
        print("Building YOLO-cls dataset from split...", flush=True)
        meta = build_yolo_dataset(split_path, YOLO_DATA_ROOT)
        print(f"Dataset: train={meta['n_train']} val={meta['n_val']} classes={len(meta['classes'])}", flush=True)
    elif not os.path.isdir(os.path.join(YOLO_DATA_ROOT, "train")):
        print("Building YOLO-cls dataset (required)...", flush=True)
        build_yolo_dataset(split_path, YOLO_DATA_ROOT)

    weights_path = args.weights
    if not args.eval_only:
        from ultralytics import YOLO

        print(f"Training {args.model} epochs={args.epochs} device={device}", flush=True)
        model = YOLO(args.model)
        model.train(
            data=YOLO_DATA_ROOT,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=EXPERIMENT_ROOT,
            name="yolo_cls",
            exist_ok=True,
            pretrained=True,
        )
        weights_path = os.path.join(YOLO_RUN_ROOT, "weights", "best.pt")
        if not os.path.isfile(weights_path):
            # Ultralytics may nest runs
            candidates = []
            for root, _dirs, files in os.walk(EXPERIMENT_ROOT):
                if "best.pt" in files and "weights" in root.replace("\\", "/"):
                    candidates.append(os.path.join(root, "best.pt"))
            if candidates:
                weights_path = max(candidates, key=os.path.getmtime)

    if not weights_path or not os.path.isfile(weights_path):
        print(f"Weights not found: {weights_path}", file=sys.stderr)
        return 1

    print(f"Evaluating {weights_path}", flush=True)
    metrics = eval_yolo_weights(weights_path, YOLO_DATA_ROOT, device=device)
    out_dir = os.path.join(TRAINING_DATA_DIR, "monument_eval", "yolo_cls_compare")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    summary_path = os.path.join(out_dir, f"summary_{stamp}.json")
    report_path = os.path.join(out_dir, f"classification_report_{stamp}.txt")
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "yolov8-cls",
        "weights": os.path.abspath(weights_path),
        "data_root": os.path.abspath(YOLO_DATA_ROOT),
        "split_file": split_path,
        "accuracy": metrics["accuracy"],
        "n_val_scored": metrics["n_val_scored"],
        "mean_top1_conf": metrics["mean_top1_conf"],
        "labels": metrics["labels"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(metrics["classification_report"])
    print(f"YOLO-cls val accuracy: {metrics['accuracy']:.4f} (n={metrics['n_val_scored']})")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {report_path}")
    # Stable pointer for integration
    latest = os.path.join(EXPERIMENT_ROOT, "latest_yolo_cls.json")
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
