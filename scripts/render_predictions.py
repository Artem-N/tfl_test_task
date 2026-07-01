"""
render_predictions.py

Purpose (per task section 10):
    Produce visual artifacts for the repo, and let you actually *watch* the
    trained detector work:

    1. Video mode (--video): run the detector over an arbitrary input video,
       draw boxes (colored by estimated distance band) live in a window
       (press 'q' to stop early) and/or save the annotated result as an mp4.
    2. Example-frames mode (--eval-frames): render a handful of annotated
       still frames from the eval GT set, with both predictions (green/orange)
       and GT boxes (blue) overlaid, saved to examples/*.jpg.

    Either mode can be used alone, or both together in one invocation.

Usage:
    # Watch the detector run on a video, live, and save the annotated result
    python scripts/render_predictions.py --weights weights/best.pt \
        --video data/videos/eval/eval.mp4 --out-video examples/eval_prediction.mp4

    # Just save (no live window), e.g. for headless/CI runs
    python scripts/render_predictions.py --weights weights/best.pt \
        --video data/videos/eval/eval.mp4 --no-show --out-video examples/eval_prediction.mp4

    # Render a few annotated example images (predictions + GT) from the eval set
    python scripts/render_predictions.py --weights weights/best.pt \
        --eval-frames data/labels/eval_gt/images --eval-labels data/labels/eval_gt/labels \
        --out-images examples/
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from distance_utils import (
    FAR_BAND,
    HORIZONTAL_FOV_DEG,
    IGNORE_BAND,
    NEAR_BAND,
    REAL_VEHICLE_LENGTH_M,
    compute_focal_length_px,
    distance_to_band,
    estimate_distance_m,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# BGR colors
COLOR_NEAR = (0, 200, 0)      # green
COLOR_FAR = (0, 140, 255)     # orange
COLOR_IGNORE = (128, 128, 128)  # gray
COLOR_GT = (255, 80, 0)       # blue
BAND_COLORS = {NEAR_BAND: COLOR_NEAR, FAR_BAND: COLOR_FAR, IGNORE_BAND: COLOR_IGNORE}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for rendering prediction overlays."""
    parser = argparse.ArgumentParser(description="Watch/render the detector's predictions.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained model weights.")

    parser.add_argument("--video", type=Path, default=None, help="Input video to run the detector on (video mode).")
    parser.add_argument(
        "--out-video",
        type=Path,
        default=None,
        help="Where to save the annotated video (default: examples/<video_stem>_prediction.mp4).",
    )
    parser.add_argument("--no-save-video", action="store_true", help="Don't save the annotated video in video mode.")
    parser.add_argument(
        "--show", action=argparse.BooleanOptionalAction, default=True,
        help="Show a live preview window while processing video (press 'q' to stop early). Use --no-show for headless runs.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame of the video (1 = every frame).")

    parser.add_argument("--eval-frames", type=Path, default=None, help="Directory of images to render examples from.")
    parser.add_argument("--eval-labels", type=Path, default=None, help="Optional GT labels dir (YOLO format) for overlay.")
    parser.add_argument("--out-images", type=Path, default=Path("examples"), help="Directory for example images.")
    parser.add_argument("--num-examples", type=int, default=6, help="How many example frames to render.")

    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions.")
    parser.add_argument("--device", type=str, default=None, help="Inference device (e.g. 'cpu' or '0').")
    parser.add_argument("--fov", type=float, default=HORIZONTAL_FOV_DEG, help="Camera horizontal FOV in degrees.")
    parser.add_argument(
        "--vehicle-length", type=float, default=REAL_VEHICLE_LENGTH_M, help="Assumed average vehicle length in meters."
    )
    return parser.parse_args()


def _band_and_distance(box_xyxy: np.ndarray, focal_px: float, vehicle_length_m: float) -> tuple[str, float]:
    w, h = box_xyxy[2] - box_xyxy[0], box_xyxy[3] - box_xyxy[1]
    distance_m = estimate_distance_m(w, h, focal_px, vehicle_length_m)
    return distance_to_band(distance_m), distance_m


def _draw_box(frame: np.ndarray, box_xyxy: np.ndarray, color: tuple, label: str, thickness: int = 2) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(y1 - 4, text_h + 2)
        cv2.rectangle(frame, (x1, text_y - text_h - 4), (x1 + text_w + 4, text_y + 2), color, -1)
        cv2.putText(frame, label, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_legend(frame: np.ndarray) -> None:
    entries = [("pred near (0-200m)", COLOR_NEAR), ("pred far (200-400m)", COLOR_FAR), ("GT", COLOR_GT)]
    x, y = 10, 24
    for text, color in entries:
        cv2.rectangle(frame, (x, y - 12), (x + 18, y), color, -1)
        cv2.putText(frame, text, (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def draw_detections(
    frame: np.ndarray,
    pred_boxes: np.ndarray,
    pred_confs: np.ndarray,
    focal_px: float,
    vehicle_length_m: float,
    gt_boxes: Optional[np.ndarray] = None,
    show_legend: bool = True,
) -> np.ndarray:
    """Draw predicted boxes (colored by distance band) and, optionally, GT
    boxes onto a copy of `frame`. Returns the annotated frame.
    """
    annotated = frame.copy()

    if gt_boxes is not None:
        for box in gt_boxes:
            _draw_box(annotated, box, COLOR_GT, "GT", thickness=1)

    for box, conf in zip(pred_boxes, pred_confs):
        band, distance_m = _band_and_distance(box, focal_px, vehicle_length_m)
        color = BAND_COLORS[band]
        label = f"vehicle {conf:.2f} | {distance_m:.0f}m {band}"
        _draw_box(annotated, box, color, label)

    if show_legend:
        _draw_legend(annotated)

    return annotated


def _predict_boxes(model: YOLO, source, conf: float, device: Optional[str]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Run inference and return (boxes_xyxy, confs, (orig_h, orig_w))."""
    result = model.predict(source, conf=conf, device=device, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
    confs = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.zeros((0,))
    return boxes, confs, result.orig_shape


def _load_gt_boxes_xyxy(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO-format GT boxes and convert to absolute pixel (x1, y1, x2, y2)."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float64)
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = (float(x) for x in parts)
        cx_px, cy_px, w_px, h_px = cx * img_w, cy * img_h, w * img_w, h * img_h
        boxes.append((cx_px - w_px / 2, cy_px - h_px / 2, cx_px + w_px / 2, cy_px + h_px / 2))
    return np.array(boxes, dtype=np.float64) if boxes else np.zeros((0, 4), dtype=np.float64)


def run_on_video(model: YOLO, args: argparse.Namespace) -> None:
    """Run the detector over `args.video`, showing a live preview and/or
    saving the annotated result, per `args.show` / `args.out_video`.
    """
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    focal_px = compute_focal_length_px(width, args.fov)

    out_video_path = args.out_video
    if out_video_path is None and not args.no_save_video:
        out_video_path = Path("examples") / f"{args.video.stem}_prediction.mp4"

    writer = None
    if out_video_path is not None:
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps / max(args.stride, 1), (width, height))
        print(f"Saving annotated video to {out_video_path}")

    window_name = f"Vehicle detector - {args.video.name} (press 'q' to quit)"
    frame_idx = 0
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % max(args.stride, 1) == 0:
                boxes, confs, _ = _predict_boxes(model, frame, args.conf, args.device)
                annotated = draw_detections(frame, boxes, confs, focal_px, args.vehicle_length)

                if writer is not None:
                    writer.write(annotated)
                if args.show:
                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Stopped early by user ('q').")
                        break
                processed += 1

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed {processed} frames from {args.video}" + (f" -> {out_video_path}" if writer is not None else ""))


def render_examples(model: YOLO, args: argparse.Namespace) -> None:
    """Render `args.num_examples` annotated example images (predictions + GT,
    if available) from `args.eval_frames` into `args.out_images`.
    """
    image_paths = sorted(p for p in args.eval_frames.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.eval_frames}")

    n = min(args.num_examples, len(image_paths))
    step = max(len(image_paths) // n, 1)
    selected = image_paths[::step][:n]

    args.out_images.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(selected, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: could not read {image_path}, skipping.")
            continue

        boxes, confs, (orig_h, orig_w) = _predict_boxes(model, str(image_path), args.conf, args.device)
        focal_px = compute_focal_length_px(orig_w, args.fov)

        gt_boxes = None
        if args.eval_labels is not None:
            label_path = args.eval_labels / f"{image_path.stem}.txt"
            gt_boxes = _load_gt_boxes_xyxy(label_path, orig_w, orig_h)

        annotated = draw_detections(frame, boxes, confs, focal_px, args.vehicle_length, gt_boxes=gt_boxes)

        out_path = args.out_images / f"eval_prediction_{i:02d}.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Wrote {out_path} (source: {image_path.name})")


def main() -> None:
    args = parse_args()
    if args.video is None and args.eval_frames is None:
        raise SystemExit("Provide --video (watch/save a video) and/or --eval-frames (render example images).")

    model = YOLO(str(args.weights))

    if args.video is not None:
        run_on_video(model, args)

    if args.eval_frames is not None:
        render_examples(model, args)


if __name__ == "__main__":
    main()
