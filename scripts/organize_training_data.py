#!/usr/bin/env python3
"""Organize unorganized images into training_data/faces/ and training_data/monuments/.

Two modes:
1. --from-datasets: Copy from training_data/datasets/ into faces/ and monuments/.
   - Faces: expects datasets/faces/.../PersonName/ (any nesting); each leaf folder = one person.
   - Monuments: expects datasets/mounuments/.../images/train/Name/ and .../test/Name/; merges into monuments/Name/.

2. Default (inbox): Copy from inbox_faces/ and inbox_monuments/ (one subfolder per name) into faces/ and monuments/.

Run from repo root. Only .jpg, .jpeg, .png are copied. Names are sanitized for filesystem safety.
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
    DATASETS_DIR,
    INBOX_FACES_DIR,
    INBOX_MONUMENTS_DIR,
)
from pipeline.utils import sanitize_dataset_name

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def copy_inbox_to_target(inbox_dir: str, target_dir: str, kind: str) -> tuple[int, int]:
    """Copy images from each subfolder of inbox_dir to target_dir/<sanitized_name>/.
    Returns (num_folders_processed, num_images_copied).
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
        copied = 0
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
            shutil.copy2(src_path, dest_path)
            copied += 1
        if copied:
            folders_done += 1
            total_images += copied
            print(f"  {kind}: {name!r} -> {safe_name}/ ({copied} images)")
    return folders_done, total_images


def _dir_has_images(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for f in os.listdir(path):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS and os.path.isfile(os.path.join(path, f)):
            return True
    return False


def _copy_images(src_dir: str, dest_dir: str) -> int:
    """Copy all images from src_dir into dest_dir (unique filenames). Returns count copied."""
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for fname in os.listdir(src_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        src_path = os.path.join(src_dir, fname)
        if not os.path.isfile(src_path):
            continue
        dest_path = os.path.join(dest_dir, fname)
        base, ext = os.path.splitext(fname)
        n = 0
        while os.path.exists(dest_path):
            n += 1
            dest_path = os.path.join(dest_dir, f"{base}_{n}{ext}")
        shutil.copy2(src_path, dest_path)
        copied += 1
    return copied


def copy_faces_from_datasets(datasets_dir: str, target_faces: str, dry_run: bool) -> tuple[int, int]:
    """Find all leaf dirs under datasets/faces/ that contain images; copy each to target_faces/<sanitized_name>/."""
    faces_root = os.path.join(datasets_dir, "faces")
    if not os.path.isdir(faces_root):
        return 0, 0
    # Collect (dir_path, label) for every dir that contains images (leaf = use its basename as label)
    person_dirs: list[tuple[str, str]] = []
    for root, dirs, _ in os.walk(faces_root):
        for d in dirs:
            sub = os.path.join(root, d)
            if _dir_has_images(sub):
                person_dirs.append((sub, d))
    if dry_run:
        for sub, name in sorted(person_dirs, key=lambda x: x[1]):
            n = sum(1 for f in os.listdir(sub) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
            print(f"  faces: {name!r} -> {sanitize_dataset_name(name) or name}/ ({n} images)")
        return len(person_dirs), sum(sum(1 for f in os.listdir(s) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS) for s, _ in person_dirs)
    folders_done = 0
    total_images = 0
    for sub, name in sorted(person_dirs, key=lambda x: x[1]):
        safe = sanitize_dataset_name(name)
        if not safe:
            continue
        dest = os.path.join(target_faces, safe)
        c = _copy_images(sub, dest)
        if c:
            folders_done += 1
            total_images += c
            print(f"  faces: {name!r} -> {safe}/ ({c} images)")
    return folders_done, total_images


def copy_monuments_from_datasets(datasets_dir: str, target_monuments: str, dry_run: bool) -> tuple[int, int]:
    """Find datasets/mounuments/.../images/train and .../test; merge each Name into target_monuments/Name/."""
    def monument_folders(parent: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if not os.path.isdir(parent):
            return out
        for name in os.listdir(parent):
            sub = os.path.join(parent, name)
            if os.path.isdir(sub) and _dir_has_images(sub):
                out.append((sub, name))
        return out

    # Find any train/ and test/ dirs under datasets (e.g. datasets/mounuments/Indian-monuments/images/train)
    train_bases: list[str] = []
    test_bases: list[str] = []
    for root, dirs, _ in os.walk(datasets_dir):
        for d in dirs:
            sub = os.path.join(root, d)
            if d == "train":
                train_bases.append(sub)
            elif d == "test":
                test_bases.append(sub)
    if not train_bases and not test_bases:
        return 0, 0

    names_to_sources: dict[str, list[str]] = {}
    for train_base in train_bases:
        for sub, name in monument_folders(train_base):
            names_to_sources.setdefault(name, []).append(sub)
    for test_base in test_bases:
        for sub, name in monument_folders(test_base):
            names_to_sources.setdefault(name, []).append(sub)

    if dry_run:
        total = 0
        for name in sorted(names_to_sources):
            n = 0
            for src in names_to_sources[name]:
                n += sum(1 for f in os.listdir(src) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
            total += n
            print(f"  monuments: {name!r} -> {sanitize_dataset_name(name) or name}/ ({n} images)")
        return len(names_to_sources), total
    folders_done = 0
    total_images = 0
    for name in sorted(names_to_sources):
        safe = sanitize_dataset_name(name)
        if not safe:
            continue
        dest = os.path.join(target_monuments, safe)
        merged = 0
        for src in names_to_sources[name]:
            merged += _copy_images(src, dest)
        if merged:
            folders_done += 1
            total_images += merged
            print(f"  monuments: {name!r} -> {safe}/ ({merged} images)")
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
        help="Only organize monuments (inbox or --from-datasets).",
    )
    parser.add_argument(
        "--from-datasets",
        action="store_true",
        help="Copy from training_data/datasets/ into faces/ and monuments/ (faces: any nesting; monuments: train+test).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying.",
    )
    args = parser.parse_args()

    do_faces = args.faces_only or (not args.monuments_only)
    do_monuments = args.monuments_only or (not args.faces_only)

    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    os.makedirs(TRAINING_FACES_DIR, exist_ok=True)
    os.makedirs(TRAINING_MONUMENTS_DIR, exist_ok=True)
    os.makedirs(INBOX_FACES_DIR, exist_ok=True)
    os.makedirs(INBOX_MONUMENTS_DIR, exist_ok=True)

    if args.from_datasets:
        total_folders = 0
        total_images = 0
        if args.dry_run:
            print("Dry run – organizing from training_data/datasets/ (no files copied).")
        if do_faces:
            print("Organizing from datasets/faces/ -> faces/")
            f, i = copy_faces_from_datasets(DATASETS_DIR, TRAINING_FACES_DIR, args.dry_run)
            total_folders += f
            total_images += i
        if do_monuments:
            print("Organizing from datasets/.../train and test/ -> monuments/")
            f, i = copy_monuments_from_datasets(DATASETS_DIR, TRAINING_MONUMENTS_DIR, args.dry_run)
            total_folders += f
            total_images += i
        if args.dry_run:
            return 0
        print(f"Done: {total_folders} folders, {total_images} images copied.")
        return 0

    # Inbox mode
    if args.dry_run:
        print("Dry run – no files will be copied.")
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
        print("Organizing inbox_faces -> faces/")
        f, i = copy_inbox_to_target(INBOX_FACES_DIR, TRAINING_FACES_DIR, "faces")
        total_folders += f
        total_images += i

    if do_monuments:
        print("Organizing inbox_monuments -> monuments/")
        f, i = copy_inbox_to_target(INBOX_MONUMENTS_DIR, TRAINING_MONUMENTS_DIR, "monuments")
        total_folders += f
        total_images += i

    print(f"Done: {total_folders} folders, {total_images} images copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
