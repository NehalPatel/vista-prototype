#!/usr/bin/env python3
"""Batch-remove backgrounds from monument training images (rembg -> RGBA PNG).

Default: write parallel tree under training_data/monuments_nobg/<class>/...
Use --in-place only when you intend to replace originals (destructive).

Requires: pip install -r requirements-monuments-extra.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.paths import TRAINING_DATASET_DIR, TRAINING_MONUMENTS_DIR  # noqa: E402

ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}


def _iter_class_images(root: str) -> Iterable[tuple[str, str, str]]:
    """Yield (class_name, src_abs, rel_under_root)."""
    if not os.path.isdir(root):
        return
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fn in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in ALLOWED:
                continue
            abs_path = os.path.join(class_dir, fn)
            if os.path.isfile(abs_path):
                rel = f"{class_name}/{fn}"
                yield class_name, abs_path, rel


def _remove_one(src: bytes) -> bytes | None:
    try:
        from rembg import remove  # type: ignore
    except ImportError:
        print(
            "rembg not installed. Run: pip install -r requirements-monuments-extra.txt",
            file=sys.stderr,
        )
        return None
    try:
        return remove(src)
    except Exception as exc:
        print(f"rembg failed: {exc}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["monuments", "dataset", "both"],
        default="monuments",
        help="Which training root to scan (default: monuments only).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Destination root for parallel tree (default: training_data/monuments_nobg).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write PNG next to source file and remove original non-PNG (destructive).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if destination PNG already exists.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="If >0, downscale longest image side before rembg (faster, less precise).",
    )
    args = parser.parse_args()

    roots: list[str] = []
    if args.source in ("monuments", "both"):
        roots.append(TRAINING_MONUMENTS_DIR)
    if args.source in ("dataset", "both"):
        roots.append(TRAINING_DATASET_DIR)

    if args.in_place and args.output_root:
        print("Do not combine --in-place with --output-root.", file=sys.stderr)
        return 1

    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError:
        print("opencv-python required.", file=sys.stderr)
        return 1

    out_default = os.path.join(os.path.dirname(TRAINING_MONUMENTS_DIR), "monuments_nobg")
    output_root = os.path.abspath(args.output_root or out_default)

    processed = 0
    skipped = 0
    failed = 0

    for base in roots:
        if not os.path.isdir(base):
            print(f"Skip missing directory: {base}")
            continue
        for class_name, abs_path, rel in _iter_class_images(base):
            if args.in_place:
                stem = os.path.splitext(abs_path)[0]
                dst = stem + ".png"
            else:
                dst = os.path.join(output_root, rel)
                stem, _ = os.path.splitext(dst)
                dst = stem + ".png"

            if args.skip_existing and os.path.isfile(dst):
                skipped += 1
                continue

            img = cv2.imread(abs_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                failed += 1
                continue

            if args.max_side > 0:
                h, w = img.shape[:2]
                m = max(h, w)
                if m > args.max_side:
                    scale = args.max_side / float(m)
                    img = cv2.resize(
                        img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

            ok, buf = cv2.imencode(".png", img)
            if not ok:
                failed += 1
                continue
            out = _remove_one(buf.tobytes())
            if out is None:
                failed += 1
                continue

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(out)
            processed += 1

            if args.in_place and not abs_path.lower().endswith(".png"):
                try:
                    os.remove(abs_path)
                except OSError:
                    pass

    print(f"Done. processed={processed} skipped={skipped} failed={failed}")
    if not args.in_place:
        print(f"Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
