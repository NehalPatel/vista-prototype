import os
import sys
import json
import csv
import math
import time
import argparse
from typing import List, Dict, Any, Iterable, Tuple


def _safe_imports():
    # Optional utils
    safe_print = print
    progress_iter = None
    try:
        from pipeline.utils import safe_print as sp, progress_iter as pi  # type: ignore
        safe_print = sp
        progress_iter = pi
    except Exception:
        pass

    # Face pipeline modules
    try:
        from face_pipeline.detection import load_detector, detect_faces  # type: ignore
        from face_pipeline.embeddings import get_embedding  # type: ignore
        from face_pipeline.recognition import load_known_embeddings, match  # type: ignore
        from face_pipeline.paths import KNOWN_FACES_DIR  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Face pipeline imports unavailable: {e}. Install 'insightface', 'onnxruntime[-gpu]', 'opencv-python'."
        )

    return safe_print, progress_iter, load_detector, detect_faces, get_embedding, load_known_embeddings, match, KNOWN_FACES_DIR


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _iter_video_frames(video_path: str, target_fps: float) -> Iterable[Tuple[int, float, Any]]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_fps = float(src_fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / max(0.1, target_fps))))
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                ts = index / src_fps
                yield index, ts, frame
            index += 1
    finally:
        cap.release()


def _process_frame(frame_bgr: Any, detector_app: Any, detect_faces, get_embedding, known_embeddings: List[Dict[str, Any]], thresholds: Dict[str, float], det_conf: float) -> List[Dict[str, Any]]:
    faces_raw = detect_faces(detector_app, frame_bgr, conf_thresh=det_conf)
    results: List[Dict[str, Any]] = []
    for fr in faces_raw:
        bbox = fr.get('bbox')
        conf = float(fr.get('confidence', 0.0))
        info: Dict[str, Any] = {
            'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            'detection_confidence': conf,
            'label': 'Unknown',
            'match_confidence': 0.0,
        }
        emb = get_embedding(fr.get('face_obj'))
        if emb is not None and known_embeddings:
            m = match(emb, known_embeddings, thresholds)
            info['label'] = m.get('label', 'Unknown')
            info['match_confidence'] = float(m.get('confidence', 0.0))
            if 'distance' in m:
                info['distance'] = float(m['distance'])
        results.append(info)
    return results


def _write_per_video_outputs(outdir: str, video_name: str, events: List[Dict[str, Any]]) -> Tuple[str, str]:
    _ensure_dir(outdir)
    base = os.path.splitext(os.path.basename(video_name))[0]
    json_path = os.path.join(outdir, f"{base}.faces.json")
    csv_path = os.path.join(outdir, f"{base}.faces.csv")

    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump({'video': video_name, 'events': events}, jf, indent=2)

    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['video', 'timestamp_s', 'label', 'match_confidence', 'detection_confidence', 'x1', 'y1', 'x2', 'y2'])
        for ev in events:
            for face in ev['faces']:
                x1, y1, x2, y2 = face['bbox']
                writer.writerow([
                    video_name,
                    f"{ev['timestamp']:.3f}",
                    face['label'],
                    f"{face.get('match_confidence', 0.0):.4f}",
                    f"{face.get('detection_confidence', 0.0):.4f}",
                    x1, y1, x2, y2,
                ])
    return json_path, csv_path


def _write_aggregate_report(
    outdir: str,
    per_video_events: Dict[str, List[Dict[str, Any]]],
    face_model: str = "buffalo_l",
) -> Tuple[str, str]:
    agg_json = os.path.join(outdir, 'aggregate.faces.json')
    agg_csv = os.path.join(outdir, 'aggregate.faces.csv')

    # Flatten
    flat_rows: List[Dict[str, Any]] = []
    for v, events in per_video_events.items():
        for ev in events:
            ts = ev['timestamp']
            for face in ev['faces']:
                row = {
                    'video': v,
                    'timestamp_s': ts,
                    'label': face['label'],
                    'match_confidence': face.get('match_confidence', 0.0),
                    'detection_confidence': face.get('detection_confidence', 0.0),
                    'bbox': face['bbox'],
                }
                flat_rows.append(row)

    with open(agg_json, 'w', encoding='utf-8') as jf:
        json.dump({
            'videos': list(per_video_events.keys()),
            'face_model': face_model,
            'detections': flat_rows,
        }, jf, indent=2)

    with open(agg_csv, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['video', 'timestamp_s', 'label', 'match_confidence', 'detection_confidence', 'x1', 'y1', 'x2', 'y2'])
        for r in flat_rows:
            x1, y1, x2, y2 = r['bbox']
            writer.writerow([
                r['video'], f"{r['timestamp_s']:.3f}", r['label'], f"{r['match_confidence']:.4f}", f"{r['detection_confidence']:.4f}", x1, y1, x2, y2
            ])
    return agg_json, agg_csv


def _glob_inputs(inputs: List[str]) -> List[str]:
    from glob import glob
    files: List[str] = []
    for pat in inputs:
        if os.path.isdir(pat):
            for root, _, fns in os.walk(pat):
                for fn in fns:
                    if fn.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        files.append(os.path.join(root, fn))
        else:
            files.extend(glob(pat))
    # Filter to known video formats
    files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
    return sorted(set(files))


def run(
    inputs: List[str],
    outdir: str,
    fps: float,
    det_conf: float,
    thresholds: Dict[str, float],
    device: str,
    model_name: str = "buffalo_l",
) -> Dict[str, List[Dict[str, Any]]]:
    safe_print, progress_iter, load_detector, detect_faces, get_embedding, load_known_embeddings, match, KNOWN_FACES_DIR = _safe_imports()

    # Device selection
    dev = device
    if dev == 'auto':
        try:
            import torch  # type: ignore
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            dev = 'cpu'
    detector_app = load_detector(device=dev, model_name=model_name)

    # Load known embeddings
    known_dir = KNOWN_FACES_DIR if isinstance(KNOWN_FACES_DIR, str) else str(KNOWN_FACES_DIR)
    known_embeddings = load_known_embeddings(known_dir)
    if not known_embeddings:
        safe_print(f"Warning: No known embeddings found in {known_dir}. All faces will be 'Unknown'.")

    video_files = _glob_inputs(inputs)
    if not video_files:
        raise RuntimeError('No input video files found for provided patterns/paths.')

    _ensure_dir(outdir)
    per_video_events: Dict[str, List[Dict[str, Any]]] = {}

    iterable = progress_iter(video_files, desc='Processing videos') if progress_iter else video_files
    for v in iterable:
        safe_print(f"Processing: {v}")
        events: List[Dict[str, Any]] = []
        for idx, ts, frame in _iter_video_frames(v, target_fps=fps):
            faces = _process_frame(frame, detector_app, detect_faces, get_embedding, known_embeddings, thresholds, det_conf)
            if faces:
                events.append({'frame_index': idx, 'timestamp': ts, 'faces': faces})
        _write_per_video_outputs(outdir, v, events)
        per_video_events[v] = events
    _write_aggregate_report(outdir, per_video_events, face_model=model_name)
    return per_video_events


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Batch video face detection and identification')
    p.add_argument('--inputs', nargs='+', required=True, help='Video paths, directories, or glob patterns')
    p.add_argument('--outdir', default=os.path.join('reports', 'video_faces'), help='Output directory for reports')
    p.add_argument('--fps', type=float, default=1.0, help='Sampling FPS for processing frames')
    p.add_argument('--det-conf', type=float, default=0.6, help='Detection confidence threshold')
    p.add_argument('--same', type=float, default=0.6, help='Recognition threshold for confident match')
    p.add_argument('--maybe', type=float, default=0.8, help='Recognition threshold for possible match')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device selection for detector')
    p.add_argument('--model', default='buffalo_l', choices=['buffalo_l', 'buffalo_s', 'buffalo_sc'], help='Face model: buffalo_l (best), buffalo_s, buffalo_sc')
    return p.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    thresholds = {'same': float(args.same), 'maybe': float(args.maybe)}
    try:
        run(
            inputs=args.inputs,
            outdir=args.outdir,
            fps=float(args.fps),
            det_conf=float(args.det_conf),
            thresholds=thresholds,
            device=args.device,
            model_name=args.model,
        )
        print(f"Done. Reports written to: {args.outdir}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())