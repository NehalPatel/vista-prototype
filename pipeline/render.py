"""Render utilities to assemble annotated frames into a final video."""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2

from .utils import safe_print


def _list_images_sorted(images_dir: str) -> List[str]:
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    # Prefer frame_XXXX ordering if present
    def sort_key(name: str) -> Tuple[int, str]:
        # Extract trailing integer from 'frame_XXXX.jpg' if available
        try:
            base = os.path.splitext(name)[0]
            if base.startswith("frame_"):
                return (int(base.split("_")[-1]), name)
        except Exception:
            pass
        return (10**9, name)

    return sorted(files, key=sort_key)


def make_video_from_images(images_dir: str, output_path: str, fps: int = 1) -> bool:
    """Create a video from images in `images_dir`.

    - Expects annotated frames (e.g., from detection step).
    - Writes MP4 by default using 'mp4v' codec.
    - Falls back to AVI ('XVID') if MP4 writer cannot be opened.
    """
    try:
        images = _list_images_sorted(images_dir)
        if not images:
            safe_print("No images found for video rendering.")
            return False

        first_img_path = os.path.join(images_dir, images[0])
        first_img = cv2.imread(first_img_path)
        if first_img is None:
            safe_print("Failed to read first image; cannot determine video size.")
            return False

        height, width = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            # Fallback to AVI
            safe_print("MP4 writer not available; falling back to AVI.")
            avi_path = os.path.splitext(output_path)[0] + ".avi"
            fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(avi_path, fourcc_avi, fps, (width, height))
            if not writer.isOpened():
                safe_print("Failed to open any video writer.")
                return False
            output_path = avi_path

        for img_name in images:
            img_path = os.path.join(images_dir, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                safe_print(f"Warning: could not read {img_name}; skipping.")
                continue
            # Ensure size consistency
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)

        writer.release()
        safe_print(f"Final video saved to: {output_path}")
        return True
    except Exception as exc:
        safe_print(f"Video rendering failed: {exc}")
        return False