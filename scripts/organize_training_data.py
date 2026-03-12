#!/usr/bin/env python3
"""Organize images from inbox into training_data/faces/ and training_data/monuments/.

Moves from inbox_faces/ and inbox_monuments/ (one subfolder per person/monument) into
faces/ and monuments/. Files are moved (not copied); empty inbox subfolders are removed.
Only .jpg, .jpeg, .png are processed. Names are sanitized for filesystem safety.

Run from repo root.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

# Run from repo root so pipeline is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.paths import (
    TRAINING_DATA_DIR,
    TRAINING_FACES_DIR,
    TRAINING_MONUMENTS_DIR,
    INBOX_FACES_DIR,
    INBOX_MONUMENTS_DIR,
)
from pipeline.utils import sanitize_dataset_name

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def move_inbox_to_target(inbox_dir: str, target_dir: str, kind: str) -> tuple[int, int]:
    """Move images from each subfolder of inbox_dir to target_dir/<sanitized_name>/.
    Empty inbox subfolders are removed after moving. Returns (num_folders_processed, num_images_moved).
    """
    if not os.path.isdir(inbox_dir):
        return 0, 0
    total_images = 0
    folders_done = 0
    for name in sorted(os.listdir(inbox_dir)):
        sub = os.path.join(inbox_dir, name)
        if not os.path.isdir(sub):
            continue
        safe_name = sanitize_dataset_name(name)
        if not safe_name:
            print(f"  [skip] {kind} folder {name!r} -> invalid name after sanitize")
            continue
        dest = os.path.join(target_dir, safe_name)
        os.makedirs(dest, exist_ok=True)
        moved = 0
        for fname in os.listdir(sub):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            src_path = os.path.join(sub, fname)
            if not os.path.isfile(src_path):
                continue
            dest_path = os.path.join(dest, fname)
            base, ext = os.path.splitext(fname)
            n = 0
            while os.path.exists(dest_path):
                n += 1
                dest_path = os.path.join(dest, f"{base}_{n}{ext}")
            shutil.move(src_path, dest_path)
            moved += 1
        if moved:
            folders_done += 1
            total_images += moved
            print(f"  {kind}: {name!r} -> {safe_name}/ ({moved} images moved)")
            # Remove inbox subfolder if empty (no remaining files)
            try:
                if not os.listdir(sub):
                    os.rmdir(sub)
            except OSError:
                pass
    return folders_done, total_images


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Organize images from inbox_faces and inbox_monuments into faces/ and monuments/."
    )
    parser.add_argument(
        "--faces-only",
        action="store_true",
        help="Only organize inbox_faces -> faces/",
    )
    parser.add_argument(
        "--monuments-only",
        action="store_true",
        help="Only organize inbox_monuments -> monuments/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without moving files.",
    )
    args = parser.parse_args()

    do_faces = args.faces_only or (not args.monuments_only)
    do_monuments = args.monuments_only or (not args.faces_only)

    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(TRAINING_FACES_DIR, exist_ok=True)
    os.makedirs(TRAINING_MONUMENTS_DIR, exist_ok=True)
    os.makedirs(INBOX_FACES_DIR, exist_ok=True)
    os.makedirs(INBOX_MONUMENTS_DIR, exist_ok=True)

    # Inbox mode
    if args.dry_run:
        print("Dry run – no files will be moved.")
        if do_faces and os.path.isdir(INBOX_FACES_DIR):
            for name in os.listdir(INBOX_FACES_DIR):
                sub = os.path.join(INBOX_FACES_DIR, name)
                if os.path.isdir(sub):
                    n = sum(1 for f in os.listdir(sub) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
                    print(f"  faces: {name!r} -> {sanitize_dataset_name(name) or name}/ ({n} images)")
        if do_monuments and os.path.isdir(INBOX_MONUMENTS_DIR):
            for name in os.listdir(INBOX_MONUMENTS_DIR):
                sub = os.path.join(INBOX_MONUMENTS_DIR, name)
                if os.path.isdir(sub):
                    n = sum(1 for f in os.listdir(sub) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
                    print(f"  monuments: {name!r} -> {sanitize_dataset_name(name) or name}/ ({n} images)")
        return 0

    total_folders = 0
    total_images = 0

    if do_faces:
        print("Organizing inbox_faces -> faces/ (moving files)")
        f, i = move_inbox_to_target(INBOX_FACES_DIR, TRAINING_FACES_DIR, "faces")
        total_folders += f
        total_images += i

    if do_monuments:
        print("Organizing inbox_monuments -> monuments/ (moving files)")
        f, i = move_inbox_to_target(INBOX_MONUMENTS_DIR, TRAINING_MONUMENTS_DIR, "monuments")
        total_folders += f
        total_images += i

    print(f"Done: {total_folders} folders, {total_images} images moved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
