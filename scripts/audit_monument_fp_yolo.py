#!/usr/bin/env python3
"""Quick FP audit using YOLO-cls weights (no rembg)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.paths import TRAINING_DATA_DIR  # noqa: E402

_ALLOWED = (".jpg", ".jpeg", ".png")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--conf-threshold", type=float, default=0.75)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=30)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from ultralytics import YOLO

    frames_dir = os.path.abspath(args.frames_dir)
    names = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(_ALLOWED))
    if args.stride > 1:
        names = names[:: args.stride]
    if args.max_frames > 0:
        names = names[: args.max_frames]

    model = YOLO(args.weights)
    accepted = []
    rejected = []
    for name in names:
        path = os.path.join(frames_dir, name)
        res = model.predict(path, verbose=False, device=args.device)
        r0 = res[0]
        probs = r0.probs
        top1 = int(probs.top1)
        names_map = r0.names if isinstance(r0.names, dict) else {i: n for i, n in enumerate(r0.names)}
        label = names_map.get(top1, str(top1))
        conf = float(probs.top1conf)
        # margin approx: top1 - top2
        data = probs.data.cpu().numpy() if hasattr(probs.data, "cpu") else probs.data
        import numpy as np

        arr = np.asarray(data).reshape(-1)
        order = np.argsort(-arr)
        margin = float(arr[order[0]] - arr[order[1]]) if len(order) > 1 else float(arr[order[0]])
        row = {"frame": name, "label": label, "confidence": conf, "margin": margin}
        if conf >= args.conf_threshold and margin >= 0.15:
            accepted.append(row)
        else:
            row["reject_reason"] = "low_confidence" if conf < args.conf_threshold else "ambiguous_margin"
            rejected.append(row)

    by_label: dict[str, int] = {}
    for row in accepted:
        by_label[row["label"]] = by_label.get(row["label"], 0) + 1

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "yolo_cls",
        "frames_dir": frames_dir,
        "weights": os.path.abspath(args.weights),
        "n_frames": len(names),
        "n_accepted_fp": len(accepted),
        "n_rejected": len(rejected),
        "false_positive_rate": float(len(accepted) / max(1, len(names))),
        "accepted_by_label": by_label,
        "accepted": accepted,
    }
    out = args.out or os.path.join(
        TRAINING_DATA_DIR,
        "monument_eval",
        "fp_audit",
        f"yolo_cls_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"YOLO FP rate: {summary['false_positive_rate']:.3f} ({len(accepted)}/{len(names)})")
    print(f"Accepted by label: {by_label}")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
