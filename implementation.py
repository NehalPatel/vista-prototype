import argparse
import os
import sys

from pipeline.paths import (
    ensure_directories,
    RESULTS_DIR,
    FRAMES_DIR,
    VIDEOS_DIR,
    get_video_results_paths,
    ensure_video_results_dirs,
)
from pipeline.utils import (
    extract_video_id_from_url,
    sanitize_id,
    validate_video_id,
    HAS_TORCH,
)
from pipeline.video import download_video, extract_frames
from pipeline.detection import (
    run_yolo,
    generate_summary,
    save_detection_results,
    write_metadata,
)
from pipeline.render import make_video_from_images


def parse_args():
    parser = argparse.ArgumentParser(description="Run object detection pipeline.")
    parser.add_argument("--url", type=str, default=None, help="YouTube URL to download and process")
    parser.add_argument("--video", type=str, default=None, help="Local video file path to process")
    parser.add_argument("--out-video", type=str, default=None, help="Output path for the rendered annotated video")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for the output video")
    parser.add_argument("--conf-threshold", type=float, default=0.7, help="Confidence threshold for detections")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directories()

    # Determine source and download if needed
    source_desc = ""
    video_path = None
    video_id = None

    if args.video:
        video_path = os.path.abspath(args.video)
        source_desc = f"local:{video_path}"
        base = os.path.splitext(os.path.basename(video_path))[0]
        video_id = sanitize_id(base)
    elif args.url:
        source_desc = args.url
        # Try pytube, fallback to yt-dlp via download_video
        video_path = download_video(args.url, VIDEOS_DIR)  # type: ignore[name-defined]
        video_id = extract_video_id_from_url(args.url) or sanitize_id(os.path.splitext(os.path.basename(video_path))[0])
    else:
        print("Error: Provide either --url or --video.", file=sys.stderr)
        sys.exit(1)

    if not video_path or not os.path.exists(video_path):
        print("Error: Video file not available after download.", file=sys.stderr)
        sys.exit(1)

    # Validate video_id
    if not validate_video_id(video_id or ""):
        print("Error: Invalid video ID derived from source.", file=sys.stderr)
        sys.exit(1)

    vid_id = video_id or "unknown"
    paths = get_video_results_paths(vid_id)

    # Ensure per-video results directories and prevent overwrite
    if not ensure_video_results_dirs(vid_id):
        print("Error: Failed to create per-video results directories.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(paths["detection_json"]) or (
        os.path.isdir(paths["processed_frames"]) and any(os.scandir(paths["processed_frames"]))
    ):
        print(
            f"Error: Existing results found for video_id '{vid_id}'. To prevent overwriting, please remove old files or use a different video ID.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract frames
    extract_frames(video_path, FRAMES_DIR)

    # Run detection with confidence threshold, save annotated frames into processed_frames
    results_by_frame = run_yolo(
        frames_dir=FRAMES_DIR,
        detections_dir=paths["processed_frames"],
        model_path="yolov8n.pt",
        conf_threshold=args.conf_threshold,
    )

    total_dets, by_class = generate_summary(results_by_frame)
    total_frames = len(results_by_frame)

    # Write single detection_results.json and metadata.txt
    save_detection_results(
        results_by_frame=results_by_frame,
        output_json_path=paths["detection_json"],
        video_id=vid_id,
        conf_threshold=args.conf_threshold,
    )

    device = "cpu"
    try:
        if HAS_TORCH:
            import torch  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    write_metadata(
        metadata_path=paths["metadata_txt"],
        video_id=vid_id,
        source=source_desc,
        total_frames=total_frames,
        total_detections=total_dets,
        by_class=by_class,
        model_name="yolov8n.pt",
        device=device,
        conf_threshold=args.conf_threshold,
    )

    # Render final annotated video
    out_video = args.out_video or os.path.join(paths["base"], "detections_video.mp4")
    make_video_from_images(
        images_dir=paths["processed_frames"],
        output_path=out_video,
        fps=args.fps,
    )

    print(f"Completed. Results in '{paths['base']}'.")


if __name__ == "__main__":
    main()