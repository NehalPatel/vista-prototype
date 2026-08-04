#!/usr/bin/env python3
"""
Evaluate monument classifier with a persisted train/val split and confusion matrix.

Workflow (baseline vs after dataset / preprocessing changes):
  1. Create or refresh the split:  python scripts/evaluate_monument_classifier.py --create-split
  2. Artifacts go to vista-prototype/training_data/monument_eval/ (timestamped run dir
     unless you pass --out-dir).
  3. After curation, background removal, or policy changes, run again *without*
     --create-split so the same image paths stay in validation when they still exist;
     images added after the split are treated as train-only for this eval fit.
  4. Compare confusion_matrix.csv, classification_report.txt, and optional heatmap PNG
     across runs.

The default mode fits LogisticRegression on the training split only (no leakage), then
evaluates on the validation split. This does not load vista-prototype/monument_model/.

Optional: --saved-model-dir PATH evaluates the saved weights on the val split; metrics
are biased if that model was trained on those val images (typical full-data training).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.monuments import (  # noqa: E402
    _extract_features_batch,
    collect_monument_images,
    load_monument_model,
)
from pipeline.paths import (  # noqa: E402
    MONUMENT_POLICY_PATH,
    MONUMENT_SPLITS_DIR,
    TRAINING_DATASET_DIR,
    TRAINING_DATA_DIR,
    TRAINING_MONUMENTS_DIR,
)


def _norm(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def _default_split_path() -> str:
    os.makedirs(MONUMENT_SPLITS_DIR, exist_ok=True)
    return os.path.join(MONUMENT_SPLITS_DIR, "train_val_split.json")


def _pairs_to_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    return {_norm(p): lab for p, lab in pairs}


def _stratified_split_per_class(
    pairs: list[tuple[str, str]],
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    from sklearn.model_selection import train_test_split

    by_label: dict[str, list[str]] = defaultdict(list)
    for path, lab in pairs:
        by_label[lab].append(_norm(path))

    train_paths: list[str] = []
    val_paths: list[str] = []
    for lab, paths in sorted(by_label.items()):
        paths_u = sorted(set(paths))
        if len(paths_u) < 2:
            train_paths.extend(paths_u)
            continue
        n_test = max(1, int(round(len(paths_u) * test_ratio)))
        n_test = min(n_test, len(paths_u) - 1)
        tr, va = train_test_split(
            paths_u,
            test_size=n_test,
            random_state=seed,
            shuffle=True,
        )
        train_paths.extend(tr)
        val_paths.extend(va)
    return sorted(set(train_paths)), sorted(set(val_paths))


def _merge_split_with_current(
    split_train: list[str],
    split_val: list[str],
    current_paths: set[str],
) -> tuple[list[str], list[str]]:
    st = set(_norm(p) for p in split_train)
    sv = set(_norm(p) for p in split_val)
    known = st | sv
    orphans = sorted(current_paths - known)

    train_final = sorted(p for p in st if p in current_paths)
    val_final = sorted(p for p in sv if p in current_paths)
    train_final = sorted(set(train_final + orphans))
    return train_final, val_final


def _write_confusion_csv(path: str, matrix: Any, labels: list[str]) -> None:
    import numpy as np

    m = np.asarray(matrix)
    with open(path, "w", encoding="utf-8") as f:
        f.write("true\\pred," + ",".join(labels) + "\n")
        for i, row_name in enumerate(labels):
            f.write(row_name + "," + ",".join(str(int(x)) for x in m[i]) + "\n")


def _plot_confusion_png(path: str, matrix: Any, labels: list[str]) -> None:
    import numpy as np

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    m = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), max(6, len(labels) * 0.45)))
    im = ax.imshow(m, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Monument classifier confusion matrix (holdout)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = m.max() / 2.0 if m.size else 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j,
                i,
                format(int(m[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if m[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monument classifier holdout evaluation and confusion matrix."
    )
    parser.add_argument(
        "--create-split",
        action="store_true",
        help="Build a new stratified split from current dataset and write split JSON.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help=f"Train/val split JSON (default: under monument_splits/)",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for this run's report artifacts (default: monument_eval/<timestamp>/)",
    )
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Override monument_policy.json path (default: training_data/monument_policy.json)",
    )
    parser.add_argument(
        "--preprocess",
        choices=["none", "rembg"],
        default="none",
        help="Must match training if comparing to rembg-trained models.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Feature extraction device (default: auto)",
    )
    parser.add_argument(
        "--saved-model-dir",
        default=None,
        help="If set, load saved monument_model and predict val set (may be optimistically biased).",
    )
    parser.add_argument(
        "--class-weight-balanced",
        action="store_true",
        help="Use class_weight=balanced when fitting the eval classifier.",
    )
    args = parser.parse_args()

    split_path = os.path.abspath(args.split_file or _default_split_path())
    policy_path = args.policy_path or MONUMENT_POLICY_PATH

    pairs = collect_monument_images(
        TRAINING_DATASET_DIR,
        TRAINING_MONUMENTS_DIR,
        policy_path=policy_path,
    )
    if not pairs:
        print("No images found.", file=sys.stderr)
        return 1

    path_to_label = _pairs_to_map(pairs)
    current_paths = set(path_to_label.keys())

    if args.create_split:
        tr, va = _stratified_split_per_class(pairs, args.test_ratio, args.seed)
        payload = {
            "version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "policy_path": policy_path,
            "train": tr,
            "val": va,
        }
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote split: {split_path} (train={len(tr)} val={len(va)})")

    if not os.path.isfile(split_path):
        print(f"No split file at {split_path}; run with --create-split first.", file=sys.stderr)
        return 1

    with open(split_path, "r", encoding="utf-8") as f:
        sp = json.load(f)
    split_train = [_norm(p) for p in sp.get("train", [])]
    split_val = [_norm(p) for p in sp.get("val", [])]

    train_paths, val_paths = _merge_split_with_current(split_train, split_val, current_paths)
    if not val_paths:
        print("Validation split is empty after intersecting with current files.", file=sys.stderr)
        return 1

    eval_root = os.path.join(TRAINING_DATA_DIR, "monument_eval")
    out_dir = args.out_dir or os.path.join(
        eval_root,
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc"),
    )
    os.makedirs(out_dir, exist_ok=True)

    device = args.device
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.preprocessing import StandardScaler

    y_val_true = np.array([path_to_label[p] for p in val_paths])
    used_preprocess = args.preprocess

    if args.saved_model_dir:
        model_dir = os.path.abspath(args.saved_model_dir)
        print(
            "WARNING: saved-model metrics are only unbiased if this model never trained on val paths.",
            file=sys.stderr,
        )
        meta = load_monument_model(model_dir)
        if meta is None:
            print(f"Could not load model from {model_dir}", file=sys.stderr)
            return 1
        used_preprocess = str(meta.get("preprocess", args.preprocess))
        feats_val = _extract_features_batch(val_paths, device, preprocess=used_preprocess)
        Xv_list = []
        yv_list = []
        for p, f, y in zip(val_paths, feats_val, y_val_true):
            if f is not None:
                Xv_list.append(f)
                yv_list.append(y)
        if not Xv_list:
            print("No valid val features.", file=sys.stderr)
            return 1
        Xv = np.array(Xv_list, dtype=np.float32)
        yv = np.array(yv_list)
        class_names = list(meta["class_names"])
        label_to_idx = {c: i for i, c in enumerate(class_names)}
        y_idx = np.array([label_to_idx[yy] for yy in yv])
        pred_labels, _ = meta["predict_fn"](Xv)
        y_pred_idx = np.array([label_to_idx[pl] for pl in pred_labels])
    else:
        y_train = np.array([path_to_label[p] for p in train_paths])
        feats_train = _extract_features_batch(train_paths, device, preprocess=args.preprocess)
        X_tr = []
        y_tr = []
        for p, f, y in zip(train_paths, feats_train, y_train):
            if f is not None:
                X_tr.append(f)
                y_tr.append(y)
        if not X_tr:
            print("Training split has no valid images/features.", file=sys.stderr)
            return 1
        if len(set(y_tr)) < 2:
            print("Need at least 2 classes in training split for classification.", file=sys.stderr)
            return 1
        X_train = np.array(X_tr, dtype=np.float32)
        y_train_arr = np.array(y_tr)

        feats_val = _extract_features_batch(val_paths, device, preprocess=args.preprocess)
        Xv_list = []
        yv_list = []
        for p, f, y in zip(val_paths, feats_val, y_val_true):
            if f is not None:
                Xv_list.append(f)
                yv_list.append(y)
        if not Xv_list:
            print("No valid val features.", file=sys.stderr)
            return 1
        Xv = np.array(Xv_list, dtype=np.float32)
        yv = np.array(yv_list)

        class_names = sorted(set(y_train_arr.tolist()) | set(yv.tolist()))
        label_to_idx = {c: i for i, c in enumerate(class_names)}
        y_train_idx = np.array([label_to_idx[yy] for yy in y_train_arr])
        y_idx = np.array([label_to_idx[yy] for yy in yv])

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        Xv_s = scaler.transform(Xv)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*multi_class.*")
            lr_kw: dict[str, Any] = {"max_iter": 1000, "random_state": args.seed}
            if args.class_weight_balanced:
                lr_kw["class_weight"] = "balanced"
            clf = LogisticRegression(**lr_kw)
            clf.fit(X_train_s, y_train_idx)
        y_pred_idx = clf.predict(Xv_s)

    labels_sorted = class_names
    cm = confusion_matrix(y_idx, y_pred_idx, labels=list(range(len(labels_sorted))))
    acc = accuracy_score(y_idx, y_pred_idx)
    report = classification_report(
        y_idx,
        y_pred_idx,
        labels=list(range(len(labels_sorted))),
        target_names=labels_sorted,
        zero_division=0,
    )

    cm_csv = os.path.join(out_dir, "confusion_matrix.csv")
    report_txt = os.path.join(out_dir, "classification_report.txt")
    summary_json = os.path.join(out_dir, "summary.json")

    _write_confusion_csv(cm_csv, cm, labels_sorted)
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(report)
    _plot_confusion_png(os.path.join(out_dir, "confusion_matrix.png"), cm, labels_sorted)

    summary = {
        "accuracy": float(acc),
        "n_train_fit": len(train_paths) if not args.saved_model_dir else None,
        "n_val": len(val_paths),
        "n_val_scored": int(len(y_idx)),
        "labels": labels_sorted,
        "split_file": split_path,
        "preprocess": used_preprocess,
        "saved_model_dir": args.saved_model_dir,
        "policy_path": policy_path,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Accuracy (holdout): {acc:.4f}")
    print(f"Wrote: {cm_csv}")
    print(f"Wrote: {report_txt}")
    print(f"Wrote: {summary_json}")
    png_path = os.path.join(out_dir, "confusion_matrix.png")
    if os.path.isfile(png_path):
        print(f"Wrote: {png_path}")
    else:
        print("(Install matplotlib for confusion_matrix.png: pip install -r requirements-monuments-extra.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
