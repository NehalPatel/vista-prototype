from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISTA_DIR = PROJECT_ROOT / "vista-prototype"

# Inputs
FRAMES_DIR = VISTA_DIR / "frames"

# Outputs for face pipeline
FACES_DIR = VISTA_DIR / "faces"
CROPS_DIR = FACES_DIR / "crops"
EMBED_DIR = FACES_DIR / "embeddings"
FACE_RESULTS_DIR = VISTA_DIR / "face_results"

# Known faces for recognition (optional)
KNOWN_FACES_DIR = VISTA_DIR / "known_faces"


def ensure_dirs() -> None:
    """Ensure all required face pipeline directories exist."""
    for d in [VISTA_DIR, FRAMES_DIR, FACES_DIR, CROPS_DIR, EMBED_DIR, FACE_RESULTS_DIR, KNOWN_FACES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "VISTA_DIR",
    "FRAMES_DIR",
    "FACES_DIR",
    "CROPS_DIR",
    "EMBED_DIR",
    "FACE_RESULTS_DIR",
    "KNOWN_FACES_DIR",
    "ensure_dirs",
]