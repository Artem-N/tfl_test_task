"""
evaluate.py

Purpose (per task sections 7-9):
    Run the trained model on the held-out eval clip(s), match predictions
    against GT boxes (IoU >= 0.5 => TP), bucket TP/FP/FN by distance band
    (near = 0-200 m, far = 200-400 m; via distance_utils.py), and compute:

        Metric                  | 0-200 m | 200-400 m
        Detection rate          | TP/(TP+FN) | TP/(TP+FN)
        Precision               | TP/(TP+FP) | TP/(TP+FP)
        False alarms / min      | FP*60/N_frames | FP*60/N_frames
        Time to first detection | seconds | seconds

    Bonus: mAP@0.5 across both bands combined (task section 9).

    IMPORTANT: the eval clip(s) were never used for training, threshold
    tuning, or model selection (task section 5) - the GT here is the final,
    manually-checked ground truth, evaluated exactly once with the config
    given on the command line.

Eval set:
    GT lives in --eval-images / --eval-labels (default: data/labels/eval_gt/),
    which currently contains manually-finalized frames from *two* source
    clips (eval.mp4 and eval_1.mp4, distinguished by filename prefix). Both
    are treated as a single combined eval set for TP/FP/FN/precision/
    detection-rate/mAP (i.e. counts are pooled across both). "Time to first
    detection" is inherently a single-timeline notion, so it is computed
    separately per source video (each has its own t=0) and also reported as
    an average across videos - see README "Evaluation Methodology".

Usage:
    python scripts/evaluate.py --weights weights/best.pt \
        --eval-images data/labels/eval_gt/images \
        --eval-labels data/labels/eval_gt/labels \
        --videos-dir data/videos/eval \
        --conf 0.25 --iou 0.5 \
        --out runs/metrics.json --out-table runs/metrics_table.md
"""

import argparse
import json
import logging
import re
from collections import defaultdict
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
BANDS = (NEAR_BAND, FAR_BAND)
FRAME_NAME_RE = re.compile(r"^(?P<video>.+)_frame_(?P<idx>\d+)$")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the trained detector on the held-out eval clip(s).")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained model weights.")
    parser.add_argument(
        "--eval-images", type=Path, default=Path("data/labels/eval_gt/images"), help="Directory of eval GT images."
    )
    parser.add_argument(
        "--eval-labels",
        type=Path,
        default=Path("data/labels/eval_gt/labels"),
        help="Directory of eval GT labels (YOLO format).",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/videos/eval"),
        help="Directory with the original source eval videos, used only to recover each frame's "
        "timestamp via that video's FPS (frame_idx / source_fps).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP matching.")
    parser.add_argument(
        "--device", type=str, default=None, help="Inference device (e.g. 'cpu' or '0'). Default: let ultralytics choose."
    )
    parser.add_argument("--fov", type=float, default=HORIZONTAL_FOV_DEG, help="Camera horizontal FOV in degrees.")
    parser.add_argument(
        "--vehicle-length", type=float, default=REAL_VEHICLE_LENGTH_M, help="Assumed average vehicle length in meters."
    )
    parser.add_argument("--out", type=Path, default=Path("runs/metrics.json"), help="Where to write metrics JSON.")
    parser.add_argument(
        "--out-table", type=Path, default=Path("runs/metrics_table.md"), help="Where to write the metrics markdown table."
    )
    return parser.parse_args()


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two axis-aligned boxes given as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_predictions_to_gt(
    pred_boxes: np.ndarray, pred_confs: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Greedily match predictions to GT boxes for a single frame.

    Predictions are considered in descending confidence order; each is
    matched to its highest-IoU still-unmatched GT box, if that IoU clears
    `iou_threshold`.

    Returns (matches, unmatched_pred_indices, unmatched_gt_indices), where
    matches is a list of (pred_idx, gt_idx, iou).
    """
    unmatched_preds = set(range(len(pred_boxes)))
    unmatched_gts = set(range(len(gt_boxes)))
    matches: list[tuple[int, int, float]] = []

    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_confs[i])
    for i in order:
        best_j: Optional[int] = None
        best_iou = 0.0
        for j in unmatched_gts:
            iou = compute_iou(pred_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None and best_iou >= iou_threshold:
            matches.append((i, best_j, best_iou))
            unmatched_preds.discard(i)
            unmatched_gts.discard(best_j)

    return matches, unmatched_preds, unmatched_gts


def _parse_video_and_frame_idx(image_path: Path) -> tuple[str, int]:
    """Recover (video_stem, frame_idx) from a `<video>_frame_<idx>` filename."""
    match = FRAME_NAME_RE.match(image_path.stem)
    if not match:
        return image_path.stem, 0
    return match.group("video"), int(match.group("idx"))


def _load_gt_boxes_xyxy(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO-format GT boxes and convert to absolute pixel (x1, y1, x2, y2).

    Returns an (N, 4) array; empty (0, 4) if the label file is missing/empty
    (i.e. no vehicles in that frame - standard YOLO convention).
    """
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


class VideoFpsLookup:
    """Resolves a video's source FPS by stem, caching results, for turning a
    frame's `<video>_frame_<idx>` naming into an absolute timestamp.
    """

    def __init__(self, videos_dir: Path, fallback_fps: float = 30.0) -> None:
        self.videos_dir = videos_dir
        self.fallback_fps = fallback_fps
        self._cache: dict[str, float] = {}

    def get_fps(self, video_stem: str) -> float:
        if video_stem in self._cache:
            return self._cache[video_stem]

        fps = self.fallback_fps
        for ext in VIDEO_EXTS:
            candidate = self.videos_dir / f"{video_stem}{ext}"
            if candidate.exists():
                cap = cv2.VideoCapture(str(candidate))
                found_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if found_fps and found_fps > 0:
                    fps = found_fps
                break
        else:
            logger.warning(
                "Could not find source video for '%s' in %s; assuming %.1f FPS for timestamps.",
                video_stem, self.videos_dir, self.fallback_fps,
            )

        self._cache[video_stem] = fps
        return fps


def compute_average_precision(records: list[dict], total_positives: int) -> float:
    """Compute AP@0.5 (across `BANDS` only) from a flat list of prediction
    records, each `{"confidence": float, "is_tp": bool, "band": str}`.

    Uses the standard monotonic-envelope precision/recall integration
    (VOC-2012-style all-point interpolation).
    """
    records = [r for r in records if r["band"] in BANDS]
    if total_positives == 0 or not records:
        return 0.0

    records = sorted(records, key=lambda r: -r["confidence"])
    tp_cum = fp_cum = 0
    precisions, recalls = [], []
    for r in records:
        if r["is_tp"]:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_positives)

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap, prev_recall = 0.0, 0.0
    for p, r in zip(precisions, recalls):
        ap += (r - prev_recall) * p
        prev_recall = r
    return ap


def evaluate(args: argparse.Namespace) -> dict:
    """Run the full evaluation pipeline and return the metrics dict."""
    model = YOLO(str(args.weights))
    fps_lookup = VideoFpsLookup(args.videos_dir)

    image_paths = sorted(p for p in args.eval_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"No eval images found in {args.eval_images}")

    counts = {band: {"tp": 0, "fp": 0, "fn": 0} for band in BANDS}
    ap_records: list[dict] = []
    total_positives_for_ap = 0

    first_gt_time: dict[str, dict[str, Optional[float]]] = defaultdict(lambda: {b: None for b in BANDS})
    first_tp_time: dict[str, dict[str, Optional[float]]] = defaultdict(lambda: {b: None for b in BANDS})
    per_frame_records: list[dict] = []

    # Group+sort per video so "first appearance"/"first detection" timestamps are
    # discovered in chronological order within each video's own timeline.
    by_video: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_paths:
        video, _ = _parse_video_and_frame_idx(image_path)
        by_video[video].append(image_path)

    n_frames = len(image_paths)
    logger.info("Evaluating %d frames across %d source video(s): %s", n_frames, len(by_video), sorted(by_video))

    for video in sorted(by_video):
        frames = sorted(by_video[video], key=lambda p: _parse_video_and_frame_idx(p)[1])
        source_fps = fps_lookup.get_fps(video)

        for image_path in frames:
            _, frame_idx = _parse_video_and_frame_idx(image_path)
            timestamp_s = frame_idx / source_fps

            result = model.predict(str(image_path), conf=args.conf, device=args.device, verbose=False)[0]
            orig_h, orig_w = result.orig_shape
            focal_px = compute_focal_length_px(orig_w, args.fov)

            pred_boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
            pred_confs = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.zeros((0,))

            label_path = args.eval_labels / f"{image_path.stem}.txt"
            gt_boxes = _load_gt_boxes_xyxy(label_path, orig_w, orig_h)

            def _band_of(box: np.ndarray) -> str:
                w, h = box[2] - box[0], box[3] - box[1]
                distance_m = estimate_distance_m(w, h, focal_px, args.vehicle_length)
                return distance_to_band(distance_m)

            gt_bands = [_band_of(box) for box in gt_boxes]
            pred_bands = [_band_of(box) for box in pred_boxes]

            matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
                pred_boxes, pred_confs, gt_boxes, args.iou
            )

            frame_tp = frame_fp = frame_fn = 0

            for pred_idx, gt_idx, _iou in matches:
                band = gt_bands[gt_idx]
                ap_records.append({"confidence": float(pred_confs[pred_idx]), "is_tp": True, "band": band})
                if band in BANDS:
                    counts[band]["tp"] += 1
                    frame_tp += 1
                    if first_tp_time[video][band] is None:
                        first_tp_time[video][band] = timestamp_s

            for pred_idx in unmatched_preds:
                band = pred_bands[pred_idx]
                ap_records.append({"confidence": float(pred_confs[pred_idx]), "is_tp": False, "band": band})
                if band in BANDS:
                    counts[band]["fp"] += 1
                    frame_fp += 1

            for gt_idx in unmatched_gts:
                band = gt_bands[gt_idx]
                if band in BANDS:
                    counts[band]["fn"] += 1
                    frame_fn += 1

            for band in BANDS:
                if band in gt_bands and first_gt_time[video][band] is None:
                    first_gt_time[video][band] = timestamp_s

            total_positives_for_ap += sum(1 for b in gt_bands if b in BANDS)

            per_frame_records.append(
                {
                    "image": image_path.name,
                    "video": video,
                    "timestamp_s": round(timestamp_s, 3),
                    "n_gt": len(gt_boxes),
                    "n_pred": len(pred_boxes),
                    "tp": frame_tp,
                    "fp": frame_fp,
                    "fn": frame_fn,
                }
            )

    # --- Aggregate metrics per band ---
    band_metrics = {}
    for band in BANDS:
        tp, fp, fn = counts[band]["tp"], counts[band]["fp"], counts[band]["fn"]
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        false_alarms_per_min = fp * 60 / n_frames if n_frames > 0 else None

        per_video_ttfd = {}
        for video in sorted(by_video):
            gt_t = first_gt_time[video][band]
            tp_t = first_tp_time[video][band]
            per_video_ttfd[video] = round(tp_t - gt_t, 3) if (gt_t is not None and tp_t is not None) else None

        valid_ttfd = [v for v in per_video_ttfd.values() if v is not None]
        overall_ttfd = round(sum(valid_ttfd) / len(valid_ttfd), 3) if valid_ttfd else None

        band_metrics[band] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "detection_rate": detection_rate,
            "precision": precision,
            "false_alarms_per_min": false_alarms_per_min,
            "time_to_first_detection_s": overall_ttfd,
            "time_to_first_detection_by_video_s": per_video_ttfd,
        }

    map50 = compute_average_precision(ap_records, total_positives_for_ap)

    metrics = {
        "weights": str(args.weights),
        "iou_threshold": args.iou,
        "conf_threshold": args.conf,
        "horizontal_fov_deg": args.fov,
        "vehicle_length_m": args.vehicle_length,
        "n_eval_frames": n_frames,
        "eval_videos": sorted(by_video),
        "bands": band_metrics,
        "map50_combined": map50,
        "frames": per_frame_records,
    }
    return metrics


def write_outputs(metrics: dict, out_json: Path, out_table: Path) -> None:
    """Write the metrics dict to JSON and a human-readable markdown table."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_json)

    def _fmt(value, pct: bool = False) -> str:
        if value is None:
            return "N/A"
        return f"{value * 100:.1f}%" if pct else f"{value:.2f}"

    near, far = metrics["bands"][NEAR_BAND], metrics["bands"][FAR_BAND]
    lines = [
        "# Evaluation Metrics",
        "",
        f"Weights: `{metrics['weights']}`  |  IoU threshold: {metrics['iou_threshold']}  |  "
        f"Confidence threshold: {metrics['conf_threshold']}",
        "",
        f"Eval videos: {', '.join(metrics['eval_videos'])}  |  N eval frames: {metrics['n_eval_frames']}  "
        f"(sampled at ~1 FPS per video)",
        "",
        f"Distance assumptions: vehicle length = {metrics['vehicle_length_m']} m, "
        f"horizontal FOV = {metrics['horizontal_fov_deg']} deg",
        "",
        "| Metric | 0-200 m | 200-400 m |",
        "|---|---|---|",
        f"| Detection rate | {_fmt(near['detection_rate'], pct=True)} | {_fmt(far['detection_rate'], pct=True)} |",
        f"| Precision | {_fmt(near['precision'], pct=True)} | {_fmt(far['precision'], pct=True)} |",
        f"| False alarms / min | {_fmt(near['false_alarms_per_min'])} | {_fmt(far['false_alarms_per_min'])} |",
        f"| Time to first detection (s) | {_fmt(near['time_to_first_detection_s'])} | "
        f"{_fmt(far['time_to_first_detection_s'])} |",
        "",
        f"**mAP@0.5 (both bands combined, bonus metric):** {metrics['map50_combined']:.3f}",
        "",
        "## Raw counts",
        "",
        "| Band | TP | FP | FN |",
        "|---|---|---|---|",
        f"| 0-200 m | {near['tp']} | {near['fp']} | {near['fn']} |",
        f"| 200-400 m | {far['tp']} | {far['fp']} | {far['fn']} |",
        "",
        "## Time to first detection by video (s)",
        "",
        "| Video | 0-200 m | 200-400 m |",
        "|---|---|---|",
    ]
    for video in metrics["eval_videos"]:
        near_v = near["time_to_first_detection_by_video_s"].get(video)
        far_v = far["time_to_first_detection_by_video_s"].get(video)
        lines.append(f"| {video} | {_fmt(near_v)} | {_fmt(far_v)} |")

    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_table)


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    write_outputs(metrics, args.out, args.out_table)


if __name__ == "__main__":
    main()
