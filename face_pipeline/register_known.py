import argparse
import os
from glob import glob
from typing import Dict, List, Tuple

import cv2
import numpy as np

from pipeline.utils import canonical_display_name

from .detection import load_detector, detect_faces, FACE_MODEL_CHOICES
from .embeddings import get_embedding
from .face_database import (
    FACE_DB_FILENAME,
    load_face_database,
    merge_embeddings_for_person,
    save_face_database,
)
from .paths import KNOWN_FACES_DIR


def find_images(folder: str) -> List[str]:
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    files: List[str] = []
    for p in patterns:
        files.extend(glob(os.path.join(folder, p), recursive=True))
    return sorted(files)


def get_embeddings_for_paths(
    detector: object,
    image_paths: List[str],
    conf_thresh: float = 0.8,
) -> List[Tuple[str, np.ndarray]]:
    """Get embeddings for image paths without saving any .npy files.

    Returns list of (image_path, embedding) for each image that yielded a face and embedding.
    """
    result: List[Tuple[str, np.ndarray]] = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        dets = detect_faces(detector, img, conf_thresh=conf_thresh)
        if not dets:
            continue
        dets.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        emb = get_embedding(dets[0].get("face_obj"))
        if emb is None:
            continue
        result.append((os.path.normpath(os.path.abspath(img_path)), emb))
    return result


def register_faces_from_folder(
    images_dir: str,
    label: str,
    device: str = "cpu",
    model_name: str = "buffalo_l",
    conf_thresh: float = 0.8,
    known_faces_dir: str | None = None,
    silent: bool = False,
) -> tuple[int, str]:
    """Register faces from a directory into known_faces/face_database.npy (mean per person).

    Merges new embeddings with any existing mean for the same display name.
    Returns (count_of_new_embeddings, error_message).
    """
    kf = known_faces_dir or str(KNOWN_FACES_DIR)
    os.makedirs(kf, exist_ok=True)

    detector = load_detector(device=device, model_name=model_name, silent=silent)
    images = find_images(images_dir)
    if not images:
        return 0, "No images found in directory"

    pairs = get_embeddings_for_paths(detector, images, conf_thresh=conf_thresh)
    if not pairs:
        return 0, "No faces detected in images"

    display_key = canonical_display_name(label)
    db = load_face_database(kf)
    # Drop stale alternate key if folder label differs from display form
    if label != display_key and label in db:
        db.pop(label, None)

    new_embs = [p[1] for p in pairs]
    n = merge_embeddings_for_person(db, display_key, new_embs)
    save_face_database(kf, db)
    return n, ""


def register_faces_from_paths(
    image_paths: List[str],
    label: str,
    device: str = "cpu",
    model_name: str = "buffalo_l",
    conf_thresh: float = 0.8,
    known_faces_dir: str | None = None,
    silent: bool = False,
) -> tuple[int, str, Dict[str, str]]:
    """Register embeddings for explicit image paths into face_database.npy.

    Returns (count, error_message, path_to_embedding). The third value is always {}
    (kept for API compatibility with older callers).
    """
    kf = known_faces_dir or str(KNOWN_FACES_DIR)
    os.makedirs(kf, exist_ok=True)

    detector = load_detector(device=device, model_name=model_name, silent=silent)
    pairs = get_embeddings_for_paths(detector, image_paths, conf_thresh=conf_thresh)
    if not pairs:
        return 0, "No faces detected in images", {}

    display_key = canonical_display_name(label)
    db = load_face_database(kf)
    if label != display_key and label in db:
        db.pop(label, None)

    new_embs = [p[1] for p in pairs]
    merge_embeddings_for_person(db, display_key, new_embs)
    save_face_database(kf, db)
    return len(new_embs), "", {}


def main():
    parser = argparse.ArgumentParser(
        description="Register known faces into face_database.npy (mean embedding per person)."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing face images (one person per run)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Person label (default: folder name of --images-dir)",
    )
    parser.add_argument(
        "--known-faces-dir",
        default=str(KNOWN_FACES_DIR),
        help=f"Directory for {FACE_DB_FILENAME}",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda", help="Device for InsightFace"
    )
    parser.add_argument(
        "--model",
        choices=list(FACE_MODEL_CHOICES),
        default="buffalo_l",
        help="Face model: buffalo_l, buffalo_s, buffalo_sc",
    )
    parser.add_argument(
        "--conf", type=float, default=0.8, help="Face detection confidence threshold"
    )
    args = parser.parse_args()

    label = args.label or os.path.basename(os.path.normpath(args.images_dir))
    count, err = register_faces_from_folder(
        args.images_dir,
        label,
        device=args.device,
        model_name=args.model,
        conf_thresh=args.conf,
        known_faces_dir=args.known_faces_dir,
        silent=False,
    )
    if err:
        print(err)
        return 1
    print(f"Registered {count} face embedding(s) into {args.known_faces_dir}/{FACE_DB_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
