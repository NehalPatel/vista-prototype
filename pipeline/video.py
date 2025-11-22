"""Video download and frame extraction logic."""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
from pytube import YouTube

from .utils import safe_print

# Optional yt-dlp fallback for robust downloads
try:
    import yt_dlp  # type: ignore
    HAS_YTDLP = True
except Exception:
    HAS_YTDLP = False


def download_video(url: str, output_dir: str) -> Optional[str]:
    """Download the highest-resolution MP4 for a given YouTube URL.

    Returns the path to the downloaded file, or None on failure.
    """
    safe_print("Starting video download...")
    try:
        yt = YouTube(url)
        stream = (
            yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()
            or yt.streams.get_highest_resolution()
        )
        if stream is None:
            safe_print("Error: No suitable MP4 stream found.")
            raise RuntimeError("No stream")

        file_path = stream.download(output_path=output_dir)
        safe_print(f"Download complete: {file_path}")
        return file_path
    except Exception as exc:
        safe_print(f"pytube failed: {exc}")

        # Fallback to yt-dlp if available
        if not HAS_YTDLP:
            safe_print("yt-dlp not installed; unable to fallback. You can install it with: pip install yt-dlp")
            return None

        try:
            safe_print("Attempting download via yt-dlp fallback...")
            os.makedirs(output_dir, exist_ok=True)
            ydl_opts = {
                # Prefer mp4 progressive; fallback to best available
                "format": "best[ext=mp4]/best",
                "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
                "quiet": True,
                "noplaylist": True,
                # Avoid post-processing requirements (ffmpeg) when possible
                "merge_output_format": "mp4",
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info)
            safe_print(f"Download complete (yt-dlp): {file_path}")
            return file_path
        except Exception as exc2:
            safe_print(f"yt-dlp download failed: {exc2}")
            return None


def extract_frames(video_path: str, frames_dir: str) -> List[str]:
    """Extract one frame per second from the video and save as JPEG files.

    Returns a list of saved frame filenames (basename only).
    """
    safe_print("Extracting frames (1 per second)...")
    
    # Clear existing frames to prevent merging with previous runs
    if os.path.exists(frames_dir):
        for f in os.listdir(frames_dir):
            fp = os.path.join(frames_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    saved_frames: List[str] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        safe_print("Error: Could not open video for frame extraction.")
        return saved_frames

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_int = max(1, int(round(fps or 1)))

        frame_index = 0
        save_index = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % fps_int == 0:
                filename = f"frame_{save_index:04d}.jpg"
                out_path = os.path.join(frames_dir, filename)
                ok = cv2.imwrite(out_path, frame)
                if ok:
                    saved_frames.append(filename)
                else:
                    safe_print(f"Warning: Failed to write frame {filename}")
                save_index += 1

            frame_index += 1

        safe_print(f"Frame extraction complete. Saved {len(saved_frames)} frames.")
        return saved_frames
    except Exception as exc:
        safe_print(f"Frame extraction error: {exc}")
        return saved_frames
    finally:
        cap.release()