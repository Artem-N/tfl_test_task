"""
extract_frames.py

Cut a source drone video into individual frame images.

By default, EVERY frame of the video is extracted (--fps not given). Pass
--fps to instead sample at an approximate target rate (e.g. --fps 1 for the
1-2 FPS sampling mentioned in the task for building the training/eval dataset).

Frames are named "<video_stem>_frame_%06d.jpg", where the number is the
*original* frame index from the source video (not a sequential save counter).
This means the timestamp of any extracted frame can always be recovered as
`frame_idx / source_fps`, without needing to know the sampling rate used.

A `frames_index.csv` sidecar file is also written into `--out-dir`, mapping
filename -> video -> frame_idx -> timestamp_s. This is needed later for
metrics like "time to first detection" (see scripts/evaluate.py).

Since multiple source videos are typically extracted into the *same*
`--out-dir` (e.g. all 4 training clips into `data/frames/train/`), this index
file is accumulated across runs: rows for the video just processed are
replaced (so re-running is safe / not duplicated), while rows belonging to
other videos already in the file are preserved.

Usage:
    # Extract every frame (default)
    python scripts/extract_frames.py --video data/videos/train/clip1.mp4 \
        --out-dir data/frames/train/clip1

    # Sample at ~1 FPS instead
    python scripts/extract_frames.py --video data/videos/train/clip1.mp4 \
        --out-dir data/frames/train/clip1 --fps 1
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import cv2

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

INDEX_FIELDS = ["filename", "video", "frame_idx", "timestamp_s"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for frame extraction."""
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to source video file.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write extracted frames to.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target sampling rate in frames per second. Omit to extract every frame (default).",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Image file format for extracted frames.",
    )
    return parser.parse_args()


def _derive_video_stem(filename: str) -> str:
    """Best-effort recovery of the source video stem from a frame filename,
    for backward compatibility with index files written before the `video`
    column existed.
    """
    marker = "_frame_"
    idx = filename.rfind(marker)
    return filename[:idx] if idx != -1 else Path(filename).stem


def _load_existing_index_rows(index_path: Path) -> list[tuple[str, str, int, str]]:
    """Load previously written index rows, if any."""
    if not index_path.exists():
        return []

    with open(index_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            video = row.get("video") or _derive_video_stem(row["filename"])
            rows.append((row["filename"], video, int(row["frame_idx"]), row["timestamp_s"]))
        return rows


def _write_frames_index(
    out_dir: Path,
    video_stem: str,
    new_rows: list[tuple[str, str, int, str]],
) -> Path:
    """Merge `new_rows` (for `video_stem`) into `out_dir/frames_index.csv`,
    replacing any prior rows for the same video and preserving rows for other
    videos already extracted into `out_dir`.
    """
    index_path = out_dir / "frames_index.csv"
    existing_rows = _load_existing_index_rows(index_path)
    kept_rows = [row for row in existing_rows if row[1] != video_stem]
    all_rows = sorted(kept_rows + new_rows, key=lambda row: (row[1], row[2]))

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(INDEX_FIELDS)
        writer.writerows(all_rows)

    return index_path


def extract_frames(
    video_path: Path,
    out_dir: Path,
    target_fps: Optional[float] = None,
    image_ext: str = "jpg",
) -> int:
    """Extract frames from `video_path` into `out_dir`.

    If `target_fps` is None (default), every frame is extracted. Otherwise,
    frames are sampled at approximately `target_fps` frames per second by
    keeping every Nth source frame, where N = round(source_fps / target_fps).

    Returns the number of frames written.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if source_fps <= 0:
        logger.warning("Could not read source FPS from %s; assuming 30.0", video_path)
        source_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / source_fps if source_fps else 0.0

    if target_fps is None or target_fps <= 0:
        frame_interval = 1
    else:
        frame_interval = max(1, round(source_fps / target_fps))

    logger.info(
        "Video: %s | %dx%d | source_fps=%.2f | total_frames=%d | duration=%.1fs | frame_interval=%d",
        video_path.name, width, height, source_fps, total_frames, duration_s, frame_interval,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    video_stem = video_path.stem

    new_rows: list[tuple[str, str, int, str]] = []
    frame_idx = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{video_stem}_frame_{frame_idx:06d}.{image_ext}"
            cv2.imwrite(str(out_dir / filename), frame)

            timestamp_s = frame_idx / source_fps
            new_rows.append((filename, video_stem, frame_idx, f"{timestamp_s:.3f}"))
            saved_count += 1

        frame_idx += 1

    cap.release()

    index_path = _write_frames_index(out_dir, video_stem, new_rows)

    logger.info("Extracted %d frames to %s (index: %s)", saved_count, out_dir, index_path)
    return saved_count


def main() -> None:
    args = parse_args()
    extract_frames(args.video, args.out_dir, args.fps, args.image_ext)


if __name__ == "__main__":
    main()
