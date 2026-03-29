#!/usr/bin/env python3
"""Inspect vista-prototype/known_faces/face_database.npy (mean embedding per person).

Run from repo root:
  python scripts/read_face_database.py
  python scripts/read_face_database.py --json
  python scripts/read_face_database.py --path vista-prototype/known_faces/face_database.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from face_pipeline.face_database import FACE_DB_FILENAME, load_face_database
from face_pipeline.paths import KNOWN_FACES_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Read and summarize face_database.npy")
    parser.add_argument(
        "--path",
        default=os.path.join(str(KNOWN_FACES_DIR), FACE_DB_FILENAME),
        help="Path to face_database.npy",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary as JSON",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.path)
    if not os.path.isfile(db_path):
        print(f"File not found: {db_path}", file=sys.stderr)
        return 1

    db = load_face_database(os.path.dirname(db_path))
    if not db:
        print("Database is empty or could not be loaded.")
        return 0

    rows = []
    for name, val in sorted(db.items(), key=lambda x: str(x[0]).lower()):
        mean = val["mean"] if isinstance(val, dict) and "mean" in val else val
        arr = np.asarray(mean, dtype=np.float32)
        count = int(val.get("count", 1)) if isinstance(val, dict) else 1
        rows.append(
            {
                "name": str(name),
                "count": count,
                "embedding_dim": int(arr.size),
                "l2_norm": float(np.linalg.norm(arr)),
            }
        )

    if args.json:
        print(json.dumps({"path": db_path, "persons": rows}, indent=2))
        return 0

    print(f"face_database.npy: {db_path}")
    print(f"Persons: {len(rows)}")
    print()
    for r in rows:
        print(
            f"  {r['name']!r}: count={r['count']}, dim={r['embedding_dim']}, ||mean||={r['l2_norm']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
