"""Utility helpers for printing, progress bars, GPU checks, and ID parsing."""

from __future__ import annotations

from typing import Iterable, Optional
import re
from urllib.parse import urlparse, parse_qs


def safe_print(message: str) -> None:
    """Print with flush for better user experience."""
    print(message, flush=True)


# Optional tqdm
try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


def progress_iter(iterable: Iterable, desc: str):
    """Wrap iterable with tqdm when available."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc)
    return iterable


# Optional torch
try:
    import torch  # type: ignore
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


_YT_ID_RE = re.compile(r"(?:v=|\/)([0-9A-Za-z_-]{6,})")


def extract_video_id_from_url(url: str) -> Optional[str]:
    """Best-effort extraction of a YouTube video ID from a URL.

    For youtube.com/watch URLs, uses the 'v' query parameter so playlist IDs
    are never used. Also supports youtu.be/VIDEOID.
    Returns None when a plausible ID cannot be found.
    """
    if not url:
        return None
    parsed = urlparse(url)
    netloc = (parsed.netloc or "").lower()
    if "youtube" in netloc and parsed.path in ("/watch", "/watch/"):
        q = parse_qs(parsed.query)
        v_list = q.get("v")
        if v_list and len(v_list) > 0 and re.match(r"^[0-9A-Za-z_-]{6,128}$", v_list[0]):
            return v_list[0]
    if "youtu.be" in netloc:
        path = (parsed.path or "").strip("/")
        if path and re.match(r"^[0-9A-Za-z_-]{6,128}$", path):
            return path
    m = _YT_ID_RE.search(url)
    if not m:
        return None
    return m.group(1)


def sanitize_id(name: str) -> str:
    """Sanitize a string to be used as a directory-friendly ID.

    Allows alphanumerics, dash and underscore; strips other chars.
    """
    cleaned = re.sub(r"[^0-9A-Za-z_-]", "", name)
    return cleaned[:128] if cleaned else ""


def validate_video_id(video_id: str) -> bool:
    """Return True if video_id is a valid identifier for folder naming."""
    if not video_id:
        return False
    return re.fullmatch(r"[0-9A-Za-z_-]{3,128}", video_id) is not None