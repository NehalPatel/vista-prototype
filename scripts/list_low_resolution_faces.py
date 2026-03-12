#!/usr/bin/env python3
"""List person names whose face folder has any low-resolution image (one low-res = whole folder).

Output is a minimal JSON with just the names so you can look up better images. Stops at the
first low-res image per person (no need to scan the rest).

Run from repo root:
  python scripts/list_low_resolution_faces.py
  python scripts/list_low_resolution_faces.py -o report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.paths import TRAINING_FACES_DIR

DEFAULT_SIZE_KB = 20
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_images(folder: str) -> list[str]:
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob(os.path.join(folder, "**", "*" + ext), recursive=True))
    return sorted(paths)


def scan_faces_dir(faces_dir: str, size_kb: float) -> list[str]:
    """Return list of person names that have at least one low-res image (by file size).
    Stops at first low-res image per person (one low-res => treat folder as low-res).
    """
    size_threshold = int(size_kb * 1024)
    low_resolution_persons = []

    if not os.path.isdir(faces_dir):
        return []

    for person_name in sorted(os.listdir(faces_dir)):
        person_dir = os.path.join(faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for path in find_images(person_dir):
            try:
                if os.path.getsize(path) < size_threshold:
                    low_resolution_persons.append(person_name)
                    break
            except OSError:
                pass
    return low_resolution_persons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List person names with low-resolution face images (output: names only, for looking up better images)."
    )
    parser.add_argument(
        "-o", "--output",
        default="low_resolution_faces.json",
        help="Output JSON path (default: low_resolution_faces.json)",
    )
    parser.add_argument(
        "--size-kb",
        type=float,
        default=DEFAULT_SIZE_KB,
        help=f"Treat image as low-res if file size < this KB (default: {DEFAULT_SIZE_KB})",
    )
    parser.add_argument("--faces-dir", default=None, help="Override faces directory")
    args = parser.parse_args()

    faces_dir = args.faces_dir or TRAINING_FACES_DIR
    names = scan_faces_dir(faces_dir, args.size_kb)
    out = {"low_resolution_persons": names}

    out_path = args.output if os.path.isabs(args.output) else os.path.join(REPO_ROOT, args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Found {len(names)} person(s) with low-resolution images → {out_path}")
    if names:
        print("Names:", ", ".join(names[:30]) + (" ..." if len(names) > 30 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
