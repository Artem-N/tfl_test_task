"""
train.py

Purpose (per task section 5, "Train videos" step 5):
    Train a single-class ("vehicle") object detection model (YOLO11) on the
    cleaned pseudo-labeled dataset built from the 4 training videos.

Before training, this script (re)builds `train.txt` / `val.txt` file lists
for the dataset from whatever images currently exist in
`data/labels/train/images/`, using a per-video temporal holdout for
validation (see `build_train_val_split` docstring). This means you can just
drop/update cleaned images+labels in that folder and re-run training; the
split is regenerated automatically.

Usage:
    python scripts/train.py --config configs/train_config.yaml

The best checkpoint from the run is copied to weights/best.pt.
"""

import argparse
import logging
import random
import re
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Matches the "<video>_frame_<idx>" naming produced by scripts/extract_frames.py
FRAME_NAME_RE = re.compile(r"^(?P<video>.+)_frame_(?P<idx>\d+)$")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser(description="Train the vehicle detector.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train_config.yaml"), help="Path to training config YAML."
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load training config from YAML.

    The `data` field is resolved relative to the config file's own directory
    (not the current working directory), so the script behaves the same
    regardless of where it's invoked from.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = (config_path.parent / data_path).resolve()
    config["data"] = str(data_path)
    return config


def _parse_video_and_frame_idx(image_path: Path) -> tuple[str, int]:
    """Recover (video_stem, frame_idx) from a `<video>_frame_<idx>` filename.

    Falls back to treating the file as its own single-frame "video" if it
    doesn't follow the expected naming convention (e.g. manually added images).
    """
    match = FRAME_NAME_RE.match(image_path.stem)
    if not match:
        return image_path.stem, 0
    return match.group("video"), int(match.group("idx"))


def build_train_val_split(dataset_dir: Path, val_split: float, seed: int = 0) -> tuple[Path, Path]:
    """Build `train.txt` / `val.txt` file lists inside `dataset_dir` from the
    images in `dataset_dir/images`.

    Images are grouped by source video (parsed from the filename). For each
    video, the *last* `val_split` fraction of frames (by frame index, i.e.
    the most recent in time) is held out for validation; the rest is used
    for training. Consecutive drone-video frames are highly correlated, so a
    random frame-level split would leak near-duplicate frames between train
    and val and give an overly optimistic validation score. A temporal
    holdout per video is a more honest (if imperfect) approximation of
    "unseen" data while still using every training video.

    Note: this is a *training-time* val split for model selection/early
    stopping only. It is unrelated to, and much smaller in stakes than, the
    fully held-out eval clip, which is never used here.

    Returns (train_txt_path, val_txt_path).
    """
    images_dir = dataset_dir / "images"
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    by_video: dict[str, list[tuple[int, Path]]] = {}
    for image_path in image_paths:
        video, frame_idx = _parse_video_and_frame_idx(image_path)
        by_video.setdefault(video, []).append((frame_idx, image_path))

    train_paths: list[Path] = []
    val_paths: list[Path] = []

    for video, frames in sorted(by_video.items()):
        frames.sort(key=lambda item: item[0])
        ordered_paths = [p for _, p in frames]
        n_val = round(len(ordered_paths) * val_split) if len(ordered_paths) > 1 else 0
        n_val = min(max(n_val, 1 if len(ordered_paths) > 1 else 0), len(ordered_paths) - 1)
        split_idx = len(ordered_paths) - n_val
        train_paths.extend(ordered_paths[:split_idx])
        val_paths.extend(ordered_paths[split_idx:])
        logger.info("Video '%s': %d frames -> %d train / %d val", video, len(ordered_paths), split_idx, n_val)

    random.Random(seed).shuffle(train_paths)  # shuffle order only; does not change the split itself

    train_txt = dataset_dir / "train.txt"
    val_txt = dataset_dir / "val.txt"
    # Absolute paths avoid any ambiguity about what the list is relative to.
    train_txt.write_text("\n".join(str(p.resolve()) for p in train_paths) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(str(p.resolve()) for p in val_paths) + "\n", encoding="utf-8")

    logger.info(
        "Split written: %d train / %d val images (val_split=%.2f) -> %s / %s",
        len(train_paths), len(val_paths), val_split, train_txt, val_txt,
    )
    return train_txt, val_txt


def _ensure_split_files_exist(data_yaml_path: Path, val_split: float, seed: int) -> None:
    """If `data.yaml`'s train/val fields point to .txt file lists, (re)build
    them from the current contents of the corresponding images/ folder.
    """
    with open(data_yaml_path, encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    dataset_root = data_yaml_path.parent
    train_field = data_config.get("train")
    if not train_field or not str(train_field).endswith(".txt"):
        return  # dataset already points directly at a directory; nothing to build

    dataset_dir = (dataset_root / train_field).resolve().parent
    build_train_val_split(dataset_dir, val_split, seed)


def train_model(config: dict) -> Path:
    """Train the detection model per `config` and return the path to the
    best checkpoint copied to weights/best.pt.
    """
    data_yaml_path = Path(config["data"])
    _ensure_split_files_exist(data_yaml_path, config["val_split"], config.get("seed", 0))

    project_path = Path(config.get("project", "runs/detect"))
    if not project_path.is_absolute():
        project_path = (REPO_ROOT / project_path).resolve()

    # Augmentation knobs (see configs/train_config.yaml for the full rationale).
    # Only forwarded if present in the config, so omitting them falls back to
    # Ultralytics' own defaults rather than silently disabling augmentation.
    augment_keys = (
        "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "scale", "shear", "perspective",
        "mosaic", "mixup", "copy_paste",
    )
    augment_kwargs = {k: config[k] for k in augment_keys if k in config}

    model = YOLO(config["model"])
    train_kwargs = dict(
        data=str(data_yaml_path),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        patience=config.get("patience"),
        optimizer=config.get("optimizer", "auto"),
        lr0=config.get("lr0"),
        seed=config.get("seed", 0),
        project=str(project_path),
        name=config.get("name", "train"),
        **augment_kwargs,
    )
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}

    logger.info("Starting training with: %s", train_kwargs)
    results = model.train(**train_kwargs)

    best_ckpt = Path(results.save_dir) / "weights" / "best.pt"
    weights_out = REPO_ROOT / "weights" / "best.pt"
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, weights_out)
    logger.info("Copied best checkpoint %s -> %s", best_ckpt, weights_out)

    return weights_out


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_model(config)


if __name__ == "__main__":
    main()
