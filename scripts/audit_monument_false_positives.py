#!/usr/bin/env python3
"""Audit monument false positives on video frames that should not contain monuments.

Samples frames from a frames directory (e.g. a street/crowd clip), runs the current
monument recognizer with the same confidence/margin/person gates used in production,
and writes a JSON summary of accepted labels vs rejects.

Examples:
  python scripts/audit_monument_false_positives.py --frames-dir vista-prototype/frames/VzLG6OqOcn8
  python scripts/audit_monument_false_positives.py --frames-dir vista-prototype/frames/VzLG6OqOcn8 --max-frames 40 --stride 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.monuments import load_monument_model, run_monument_recognition  # noqa: E402
from pipeline.paths import MONUMENT_MODEL_DIR, TRAINING_DATA_DIR  # noqa: E402

_ALLOWED = (".jpg", ".jpeg", ".png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "False-positive audit for monument recognition on non-landmark video frames. "
            "Reports how often gates reject vs how often a landmark label is accepted."
        )
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Directory of extracted video frames (jpg/png).",
    )
    parser.add_argument("--model-dir", default=MONUMENT_MODEL_DIR)
    parser.add_argument("--conf-threshold", type=float, default=0.75)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--max-person-count", type=int, default=3)
    parser.add_argument("--max-person-area-ratio", type=float, default=0.25)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth frame after sort (default 1 = all).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If >0, cap number of frames after stride.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default under training_data/monument_eval/).",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    frames_dir = os.path.abspath(args.frames_dir)
    if not os.path.isdir(frames_dir):
        print(f"Frames dir not found: {frames_dir}", file=sys.stderr)
        return 1
    if load_monument_model(args.model_dir) is None:
        print(f"No monument model at {args.model_dir}", file=sys.stderr)
        return 1

    names = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith(_ALLOWED)
    )
    if args.stride > 1:
        names = names[:: args.stride]
    if args.max_frames > 0:
        names = names[: args.max_frames]
    if not names:
        print("No frames to audit.", file=sys.stderr)
        return 1

    # Temporary subset directory so we do not mutate the full video folder listing logic.
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory(prefix="monument_fp_audit_") as tmp:
        for n in names:
            src = os.path.join(frames_dir, n)
            dst = os.path.join(tmp, n)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

        device = args.device
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        print(
            f"Auditing {len(names)} frames from {frames_dir} "
            f"(conf>={args.conf_threshold}, margin>={args.margin_threshold}, device={device})",
            flush=True,
        )
        results = run_monument_recognition(
            tmp,
            args.model_dir,
            device=device,
            confidence_threshold=args.conf_threshold,
            margin_threshold=args.margin_threshold,
            detections_by_frame=None,  # no YOLO dets: measures classifier FP rate alone
            max_person_count=args.max_person_count,
            max_person_area_ratio=args.max_person_area_ratio,
        )

    accepted = []
    rejected = []
    for name in names:
        info = results.get(name) or {"label": "Unknown", "confidence": 0.0}
        row = {
            "frame": name,
            "label": info.get("label"),
            "confidence": info.get("confidence"),
            "margin": info.get("margin"),
            "reject_reason": info.get("reject_reason"),
        }
        if info.get("label") and info.get("label") != "Unknown":
            accepted.append(row)
        else:
            rejected.append(row)

    by_label: dict[str, int] = {}
    for row in accepted:
        lab = str(row["label"])
        by_label[lab] = by_label.get(lab, 0) + 1

    reason_counts: dict[str, int] = {}
    for row in rejected:
        r = str(row.get("reject_reason") or "unknown_reject")
        reason_counts[r] = reason_counts.get(r, 0) + 1

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frames_dir": frames_dir,
        "model_dir": os.path.abspath(args.model_dir),
        "n_frames": len(names),
        "n_accepted_fp": len(accepted),
        "n_rejected": len(rejected),
        "false_positive_rate": float(len(accepted) / max(1, len(names))),
        "accepted_by_label": by_label,
        "reject_reasons": reason_counts,
        "settings": {
            "conf_threshold": args.conf_threshold,
            "margin_threshold": args.margin_threshold,
            "stride": args.stride,
            "max_frames": args.max_frames,
            "note": "detections_by_frame=None so person gates did not fire; measures classifier-only FPs",
        },
        "accepted": accepted,
        "rejected_sample": rejected[:50],
    }

    out = args.out
    if not out:
        out_dir = os.path.join(TRAINING_DATA_DIR, "monument_eval", "fp_audit")
        os.makedirs(out_dir, exist_ok=True)
        vid = os.path.basename(frames_dir.rstrip("\\/"))
        out = os.path.join(
            out_dir,
            f"{vid}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"FP rate: {summary['false_positive_rate']:.3f} ({len(accepted)}/{len(names)})")
    print(f"Accepted by label: {by_label}")
    print(f"Reject reasons: {reason_counts}")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
