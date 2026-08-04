#!/usr/bin/env python3
"""Curate monument training images with strict quality filters.

This script helps clean monument datasets by flagging/moving images that are likely
incorrect for a class (class-consistency outliers), person-dominant, or low quality.

Default workflow:
1) Scan a class first (dry run style report only)
2) Quarantine flagged files
3) Review quarantine manually
4) Restore mistakes or purge quarantine later
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Resolve repo root for imports when run as `python scripts/curate_monuments.py`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.paths import TRAINING_DATA_DIR, TRAINING_MONUMENTS_DIR

try:
    import torch
    from torchvision import models
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing torch/torchvision required for curation embeddings. "
        "Install dependencies from requirements.txt."
    ) from exc

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing ultralytics required for person filtering. "
        "Install dependencies from requirements.txt."
    ) from exc


ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageMetrics:
    class_name: str
    rel_path: str
    abs_path: str
    blur_laplacian_var: float
    brightness_mean: float
    person_count: int
    person_area_ratio: float
    embedding_distance: float
    outlier_z: float
    duplicate_of: str | None
    reasons: list[str]
    rembg_fg_ratio: float | None = None
    rembg_structure_count: int | None = None
    rembg_largest_share: float | None = None


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in ALLOWED_EXTS


def _iter_images(root_dir: str, class_filter: str | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if not os.path.isdir(root_dir):
        return rows
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_filter and class_name.lower() != class_filter.lower():
            continue
        for file_name in sorted(os.listdir(class_dir)):
            abs_path = os.path.join(class_dir, file_name)
            if os.path.isfile(abs_path) and _is_image_file(abs_path):
                rows.append((class_name, abs_path))
    return rows


def _load_image_rgb(path: str) -> np.ndarray | None:
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def _compute_blur_brightness(rgb: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    return blur_var, brightness


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _dhash64(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = tiny[:, 1:] > tiny[:, :-1]
    out = 0
    bit = 0
    for y in range(8):
        for x in range(8):
            if diff[y, x]:
                out |= (1 << bit)
            bit += 1
    return out


def _hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _load_person_model() -> YOLO:
    # Prefer local model file to avoid downloading.
    local_model = os.path.join(REPO_ROOT, "yolov8n.pt")
    model_path = local_model if os.path.isfile(local_model) else "yolov8n.pt"
    return YOLO(model_path)


def _person_stats(
    model: YOLO,
    rgb: np.ndarray,
    conf_threshold: float,
) -> tuple[int, float]:
    h, w = rgb.shape[:2]
    img_area = float(max(1, h * w))
    results = model.predict(rgb, conf=conf_threshold, verbose=False)
    person_count = 0
    person_area = 0.0
    if not results:
        return 0, 0.0
    r0 = results[0]
    if r0.boxes is None or r0.boxes.cls is None or r0.boxes.xyxy is None:
        return 0, 0.0
    classes = r0.boxes.cls.cpu().numpy().astype(int)
    boxes = r0.boxes.xyxy.cpu().numpy()
    for cls_id, (x1, y1, x2, y2) in zip(classes, boxes):
        if cls_id != 0:  # COCO class 0 = person
            continue
        person_count += 1
        bw = max(0.0, float(x2 - x1))
        bh = max(0.0, float(y2 - y1))
        person_area += bw * bh
    return person_count, float(person_area / img_area)


class Embedder:
    def __init__(self, device: str) -> None:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
        # Use penultimate features for image similarity.
        self.model = torch.nn.Sequential(*(list(backbone.children())[:-1])).to(device)
        self.model.eval()
        self.preprocess = weights.transforms()
        self.device = device

    @torch.no_grad()
    def encode(self, rgb: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(rgb)
        pil_like = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        feat = self.model(pil_like).flatten(1)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy()[0].astype(np.float32)


def _robust_outlier_scores(distances: list[float]) -> list[float]:
    if not distances:
        return []
    arr = np.asarray(distances, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    denom = max(mad * 1.4826, 1e-6)
    return [float((x - med) / denom) for x in arr]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_report_path() -> str:
    report_dir = os.path.join(TRAINING_DATA_DIR, "curation_reports")
    os.makedirs(report_dir, exist_ok=True)
    return os.path.join(report_dir, f"monuments_curation_{_timestamp()}.json")


def _default_quarantine_dir() -> str:
    qdir = os.path.join(TRAINING_DATA_DIR, "monuments_quarantine")
    os.makedirs(qdir, exist_ok=True)
    return qdir


def _manifest_path(quarantine_dir: str) -> str:
    return os.path.join(quarantine_dir, "_manifest.jsonl")


def _append_manifest(quarantine_dir: str, row: dict[str, Any]) -> None:
    path = _manifest_path(quarantine_dir)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _classify_reasons(
    blur_var: float,
    brightness: float,
    person_count: int,
    person_area_ratio: float,
    outlier_z: float,
    blur_threshold: float,
    min_brightness: float,
    max_brightness: float,
    max_person_count: int,
    max_person_area_ratio: float,
    outlier_z_threshold: float,
    rembg_fg_ratio: float | None = None,
    rembg_structure_count: int | None = None,
    rembg_largest_share: float | None = None,
    min_fg_ratio: float = 0.0,
    max_fg_ratio: float = 1.0,
    max_structure_blobs: int = 999,
    min_largest_share: float = 0.0,
) -> list[str]:
    reasons: list[str] = []
    if blur_var < blur_threshold:
        reasons.append("low_detail_blurry")
    if brightness < min_brightness:
        reasons.append("too_dark")
    if brightness > max_brightness:
        reasons.append("too_bright")
    if person_count > max_person_count:
        reasons.append("too_many_people")
    if person_area_ratio > max_person_area_ratio:
        reasons.append("person_dominant")
    if outlier_z > outlier_z_threshold:
        reasons.append("class_consistency_outlier")
    if rembg_fg_ratio is not None:
        if rembg_fg_ratio < min_fg_ratio:
            reasons.append("rembg_subject_too_small")
        if rembg_fg_ratio > max_fg_ratio:
            reasons.append("rembg_foreground_full_frame")
    if rembg_structure_count is not None and rembg_structure_count > max_structure_blobs:
        reasons.append("multi_structure_candidate")
    if rembg_largest_share is not None and rembg_largest_share < min_largest_share:
        reasons.append("mask_fragmented")
    return reasons


def _rembg_mask_metrics(
    rgb: np.ndarray,
    min_blob_area_ratio: float,
) -> tuple[float | None, int | None, float | None]:
    """Foreground ratio, count of large connected components, largest share of FG."""
    try:
        from rembg import remove  # type: ignore
    except ImportError:
        return None, None, None
    h, w = rgb.shape[:2]
    img_area = max(1, h * w)
    min_area = max(32, int(min_blob_area_ratio * img_area))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return None, None, None
    try:
        out_bytes = remove(buf.tobytes())
    except Exception:
        return None, None, None
    arr = np.frombuffer(out_bytes, dtype=np.uint8)
    dec = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if dec is None or dec.shape[2] < 4:
        return None, None, None
    alpha = dec[:, :, 3]
    binary = (alpha > 127).astype(np.uint8)
    fg_ratio = float(binary.mean())
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas: list[int] = []
    for i in range(1, n_lab):
        areas.append(int(stats[i, cv2.CC_STAT_AREA]))
    total_fg = sum(areas)
    if total_fg <= 0:
        return fg_ratio, 0, 0.0
    large = [a for a in areas if a >= min_area]
    largest = max(areas) if areas else 0
    largest_share = float(largest / total_fg)
    return fg_ratio, len(large), largest_share


def run_scan_or_quarantine(args: argparse.Namespace) -> dict[str, Any]:
    monuments_dir = os.path.abspath(args.monuments_dir)
    quarantine_dir = os.path.abspath(args.quarantine_dir)
    report_path = os.path.abspath(args.report_path or _default_report_path())
    run_id = _timestamp()

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs(quarantine_dir, exist_ok=True)

    entries = _iter_images(monuments_dir, class_filter=args.class_name)
    if not entries:
        raise SystemExit(f"No images found in monuments dir: {monuments_dir}")

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    embedder = Embedder(device=device)
    person_model = _load_person_model()

    # First pass: compute per-image metrics and embeddings.
    base_rows: list[dict[str, Any]] = []
    class_vectors: dict[str, list[np.ndarray]] = {}
    for class_name, abs_path in entries:
        rgb = _load_image_rgb(abs_path)
        if rgb is None:
            continue
        rel_path = os.path.relpath(abs_path, monuments_dir).replace("\\", "/")
        # Duplicate signatures
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        file_sha1 = _sha1_bytes(encoded.tobytes()) if ok else ""
        dhash64 = _dhash64(rgb)

        blur_var, brightness = _compute_blur_brightness(rgb)
        person_count, person_area_ratio = _person_stats(
            person_model, rgb, conf_threshold=args.person_conf
        )
        vec = embedder.encode(rgb)
        class_vectors.setdefault(class_name, []).append(vec)
        rembg_fg: float | None = None
        rembg_n: int | None = None
        rembg_share: float | None = None
        if args.rembg_check:
            rembg_fg, rembg_n, rembg_share = _rembg_mask_metrics(
                rgb, float(args.rembg_min_blob_area_ratio)
            )
        base_rows.append(
            {
                "class_name": class_name,
                "abs_path": abs_path,
                "rel_path": rel_path,
                "blur_var": blur_var,
                "brightness": brightness,
                "person_count": person_count,
                "person_area_ratio": person_area_ratio,
                "file_sha1": file_sha1,
                "dhash64": dhash64,
                "vec": vec,
                "rembg_fg_ratio": rembg_fg,
                "rembg_structure_count": rembg_n,
                "rembg_largest_share": rembg_share,
            }
        )

    # Compute class centroids.
    class_centroids: dict[str, np.ndarray] = {}
    for class_name, vecs in class_vectors.items():
        mat = np.stack(vecs, axis=0)
        centroid = mat.mean(axis=0)
        norm = np.linalg.norm(centroid) + 1e-8
        class_centroids[class_name] = (centroid / norm).astype(np.float32)

    # Compute robust outlier z-scores per class.
    by_class_rows: dict[str, list[dict[str, Any]]] = {}
    for row in base_rows:
        by_class_rows.setdefault(row["class_name"], []).append(row)
    for class_name, rows in by_class_rows.items():
        centroid = class_centroids[class_name]
        dists: list[float] = []
        for row in rows:
            vec = row["vec"]
            cos_sim = float(np.dot(vec, centroid) / (np.linalg.norm(vec) + 1e-8))
            dist = 1.0 - cos_sim
            row["embedding_distance"] = dist
            dists.append(dist)
        zscores = _robust_outlier_scores(dists)
        for row, z in zip(rows, zscores):
            row["outlier_z"] = z

    # Duplicate detection within each class.
    # Keep the first seen unique image and flag subsequent exact/near duplicates.
    for class_name, rows in by_class_rows.items():
        rows_sorted = sorted(rows, key=lambda r: r["rel_path"])
        exact_seen: dict[str, str] = {}
        kept_hashes: list[tuple[int, str]] = []
        for row in rows_sorted:
            row["duplicate_of"] = None
            sha1 = str(row.get("file_sha1", ""))
            if sha1 and sha1 in exact_seen:
                row["duplicate_of"] = exact_seen[sha1]
                continue
            if sha1:
                exact_seen[sha1] = row["rel_path"]

            if not args.enable_duplicate_check:
                continue
            cur_hash = int(row.get("dhash64", 0))
            found: str | None = None
            for prior_hash, prior_rel in kept_hashes:
                if _hamming64(cur_hash, prior_hash) <= int(args.dup_hamming_threshold):
                    found = prior_rel
                    break
            if found is not None:
                row["duplicate_of"] = found
            else:
                kept_hashes.append((cur_hash, row["rel_path"]))

    # Final classify and optional quarantine move.
    all_metrics: list[ImageMetrics] = []
    moved_count = 0
    kept_count = 0
    for row in base_rows:
        outlier_z = float(row.get("outlier_z", 0.0))
        rembg_fg = row.get("rembg_fg_ratio")
        rembg_n = row.get("rembg_structure_count")
        rembg_share = row.get("rembg_largest_share")
        reasons = _classify_reasons(
            blur_var=float(row["blur_var"]),
            brightness=float(row["brightness"]),
            person_count=int(row["person_count"]),
            person_area_ratio=float(row["person_area_ratio"]),
            outlier_z=outlier_z,
            blur_threshold=float(args.blur_threshold),
            min_brightness=float(args.min_brightness),
            max_brightness=float(args.max_brightness),
            max_person_count=int(args.max_person_count),
            max_person_area_ratio=float(args.max_person_area_ratio),
            outlier_z_threshold=float(args.outlier_z_threshold),
            rembg_fg_ratio=rembg_fg if args.rembg_check else None,
            rembg_structure_count=rembg_n if args.rembg_check else None,
            rembg_largest_share=rembg_share if args.rembg_check else None,
            min_fg_ratio=float(args.rembg_min_fg_ratio),
            max_fg_ratio=float(args.rembg_max_fg_ratio),
            max_structure_blobs=int(args.rembg_max_structure_blobs),
            min_largest_share=float(args.rembg_min_largest_share),
        )
        duplicate_of = row.get("duplicate_of")
        if duplicate_of:
            reasons.append("duplicate_image")
        if (
            args.urban_review_candidates
            and rembg_fg is not None
            and 0.12 < rembg_fg < 0.48
            and int(row["person_count"]) == 0
        ):
            reasons.append("urban_context_review")
        metric = ImageMetrics(
            class_name=row["class_name"],
            rel_path=row["rel_path"],
            abs_path=row["abs_path"],
            blur_laplacian_var=float(row["blur_var"]),
            brightness_mean=float(row["brightness"]),
            person_count=int(row["person_count"]),
            person_area_ratio=float(row["person_area_ratio"]),
            embedding_distance=float(row.get("embedding_distance", 0.0)),
            outlier_z=outlier_z,
            duplicate_of=duplicate_of,
            reasons=reasons,
            rembg_fg_ratio=rembg_fg,
            rembg_structure_count=rembg_n,
            rembg_largest_share=rembg_share,
        )
        all_metrics.append(metric)

        if not reasons:
            kept_count += 1
            continue

        if args.mode == "quarantine":
            src = metric.abs_path
            rel = metric.rel_path
            dst = os.path.join(quarantine_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved_count += 1
            _append_manifest(
                quarantine_dir,
                {
                    "run_id": run_id,
                    "timestamp": int(time.time()),
                    "class_name": metric.class_name,
                    "source_rel": rel,
                    "source_abs": src,
                    "quarantine_abs": dst,
                    "reasons": reasons,
                    "duplicate_of": duplicate_of,
                },
            )

    # Build report.
    class_summary: dict[str, dict[str, int]] = {}
    for m in all_metrics:
        summary = class_summary.setdefault(
            m.class_name, {"total": 0, "flagged": 0, "kept": 0}
        )
        summary["total"] += 1
        if m.reasons:
            summary["flagged"] += 1
        else:
            summary["kept"] += 1

    report = {
        "run_id": run_id,
        "mode": args.mode,
        "strict_profile": "very_strict",
        "settings": {
            "blur_threshold": args.blur_threshold,
            "min_brightness": args.min_brightness,
            "max_brightness": args.max_brightness,
            "max_person_count": args.max_person_count,
            "max_person_area_ratio": args.max_person_area_ratio,
            "outlier_z_threshold": args.outlier_z_threshold,
            "person_conf": args.person_conf,
            "device": device,
            "rembg_check": args.rembg_check,
            "rembg_min_fg_ratio": args.rembg_min_fg_ratio,
            "rembg_max_fg_ratio": args.rembg_max_fg_ratio,
            "rembg_max_structure_blobs": args.rembg_max_structure_blobs,
            "rembg_min_largest_share": args.rembg_min_largest_share,
            "rembg_min_blob_area_ratio": args.rembg_min_blob_area_ratio,
            "urban_review_candidates": args.urban_review_candidates,
        },
        "paths": {
            "monuments_dir": monuments_dir,
            "quarantine_dir": quarantine_dir,
            "report_path": report_path,
        },
        "totals": {
            "images_scanned": len(all_metrics),
            "flagged": sum(1 for m in all_metrics if m.reasons),
            "kept": kept_count,
            "quarantined_moved": moved_count,
        },
        "class_summary": class_summary,
        "files": [
            {
                "class_name": m.class_name,
                "relative_path": m.rel_path,
                "reasons": m.reasons,
                "metrics": {
                    "blur_laplacian_var": m.blur_laplacian_var,
                    "brightness_mean": m.brightness_mean,
                    "person_count": m.person_count,
                    "person_area_ratio": m.person_area_ratio,
                    "embedding_distance": m.embedding_distance,
                    "outlier_z": m.outlier_z,
                    "rembg_fg_ratio": m.rembg_fg_ratio,
                    "rembg_structure_count": m.rembg_structure_count,
                    "rembg_largest_share": m.rembg_largest_share,
                },
                "duplicate_of": m.duplicate_of,
            }
            for m in all_metrics
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report written: {report_path}")
    print(
        f"Scanned={report['totals']['images_scanned']} "
        f"Flagged={report['totals']['flagged']} "
        f"Kept={report['totals']['kept']} "
        f"Moved={report['totals']['quarantined_moved']}"
    )
    return report


def run_restore(args: argparse.Namespace) -> None:
    monuments_dir = os.path.abspath(args.monuments_dir)
    quarantine_dir = os.path.abspath(args.quarantine_dir)
    if not os.path.isdir(quarantine_dir):
        print("No quarantine directory found.")
        return

    restored = 0
    for class_name in sorted(os.listdir(quarantine_dir)):
        class_dir = os.path.join(quarantine_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name.startswith("_"):
            continue
        if args.class_name and class_name.lower() != args.class_name.lower():
            continue
        for file_name in sorted(os.listdir(class_dir)):
            src = os.path.join(class_dir, file_name)
            if not os.path.isfile(src) or not _is_image_file(src):
                continue
            dst = os.path.join(monuments_dir, class_name, file_name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                stem, ext = os.path.splitext(file_name)
                dst = os.path.join(
                    monuments_dir, class_name, f"{stem}_restored_{int(time.time())}{ext}"
                )
            shutil.move(src, dst)
            restored += 1
    print(f"Restored {restored} file(s).")


def run_purge(args: argparse.Namespace) -> None:
    quarantine_dir = os.path.abspath(args.quarantine_dir)
    if not os.path.isdir(quarantine_dir):
        print("No quarantine directory found.")
        return
    if not args.yes:
        raise SystemExit("Refusing purge without --yes.")

    removed = 0
    for class_name in sorted(os.listdir(quarantine_dir)):
        class_dir = os.path.join(quarantine_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name.startswith("_"):
            continue
        if args.class_name and class_name.lower() != args.class_name.lower():
            continue
        for file_name in os.listdir(class_dir):
            path = os.path.join(class_dir, file_name)
            if os.path.isfile(path) and _is_image_file(path):
                os.remove(path)
                removed += 1
    print(f"Purged {removed} quarantined image(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict monument dataset curation with quarantine/restore/purge."
    )
    parser.add_argument(
        "--mode",
        choices=["scan", "quarantine", "restore", "purge"],
        default="scan",
        help="scan=report only, quarantine=move flagged files, restore=move back, purge=delete quarantine images",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Same as --mode scan (report only). Cannot be combined with another --mode.",
    )
    parser.add_argument(
        "--monuments-dir",
        default=TRAINING_MONUMENTS_DIR,
        help="Path to monument training root.",
    )
    parser.add_argument(
        "--quarantine-dir",
        default=_default_quarantine_dir(),
        help="Path to quarantine root.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="JSON report output path (scan/quarantine only).",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="Limit operation to one monument class (e.g., tajmahal).",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Disable CUDA usage.")

    # Very strict defaults.
    parser.add_argument("--blur-threshold", type=float, default=70.0)
    parser.add_argument("--min-brightness", type=float, default=40.0)
    parser.add_argument("--max-brightness", type=float, default=220.0)
    parser.add_argument("--max-person-count", type=int, default=1)
    parser.add_argument("--max-person-area-ratio", type=float, default=0.12)
    parser.add_argument("--outlier-z-threshold", type=float, default=2.2)
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument(
        "--enable-duplicate-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable near-duplicate detection inside each class.",
    )
    parser.add_argument(
        "--dup-hamming-threshold",
        type=int,
        default=4,
        help="dHash hamming distance threshold for near-duplicates (lower=stricter).",
    )
    parser.add_argument(
        "--no-people",
        action="store_true",
        help="Reject any visible person (sets max person count to 0 and area ratio to 0).",
    )
    parser.add_argument(
        "--rembg-check",
        action="store_true",
        help="Run rembg per image and apply foreground / multi-blob heuristics (slow; needs rembg).",
    )
    parser.add_argument(
        "--rembg-min-fg-ratio",
        type=float,
        default=0.06,
        help="With --rembg-check: flag if foreground share of pixels is below this.",
    )
    parser.add_argument(
        "--rembg-max-fg-ratio",
        type=float,
        default=0.93,
        help="With --rembg-check: flag if foreground covers more than this (likely rembg failure).",
    )
    parser.add_argument(
        "--rembg-max-structure-blobs",
        type=int,
        default=1,
        help="With --rembg-check: flag if more than this many large connected components.",
    )
    parser.add_argument(
        "--rembg-min-largest-share",
        type=float,
        default=0.52,
        help="With --rembg-check: flag if largest blob is smaller than this fraction of total FG.",
    )
    parser.add_argument(
        "--rembg-min-blob-area-ratio",
        type=float,
        default=0.04,
        help="Min area (as fraction of image) for a blob to count as a structure.",
    )
    parser.add_argument(
        "--urban-review-candidates",
        action="store_true",
        help="Tag images with mid-range rembg FG and no people for manual urban/clutter review.",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive purge operation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scan and args.mode != "scan":
        raise SystemExit(
            f"error: --scan cannot be used with --mode {args.mode}; use only --scan or --mode scan"
        )
    if args.no_people:
        args.max_person_count = 0
        args.max_person_area_ratio = 0.0
    if args.mode in ("scan", "quarantine"):
        run_scan_or_quarantine(args)
        return 0
    if args.mode == "restore":
        run_restore(args)
        return 0
    if args.mode == "purge":
        run_purge(args)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

