"""Utilities for preparing the COCO weed-detection dataset.

The module deliberately does not download data or fabricate labels.  It fails
with actionable errors when the configured dataset is absent or malformed.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require_directory(path: Path, description: str) -> Path:
    """Return *path* when it exists as a directory, otherwise fail clearly."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"{description} was not found: {path}. "
            "Set WEED_DATA_ROOT or provide the required dataset before running."
        )
    return path


def set_seed(seed: int) -> None:
    """Set the Python-level reproducibility controls used by this project."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)


def _safe_source(root: Path, relative_name: str) -> Path:
    """Resolve a COCO file name while rejecting path traversal."""
    candidate = (root / relative_name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"COCO file name escapes the image directory: {relative_name!r}") from exc
    return candidate


def _validate_box(bbox: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, box_width, box_height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        return None
    x0 = max(0.0, min(float(width), x))
    y0 = max(0.0, min(float(height), y))
    x1 = max(0.0, min(float(width), x + box_width))
    y1 = max(0.0, min(float(height), y + box_height))
    clipped_width, clipped_height = x1 - x0, y1 - y0
    if clipped_width <= 0 or clipped_height <= 0:
        return None
    return (
        (x0 + x1) / 2 / width,
        (y0 + y1) / 2 / height,
        clipped_width / width,
        clipped_height / height,
    )


def coco_to_yolo(annotation_path: Path, images_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Convert a COCO annotation file into a YOLO directory without inventing data."""
    annotation_path = annotation_path.expanduser().resolve()
    images_dir = require_directory(images_dir, "COCO image directory")
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"COCO annotation file was not found: {annotation_path}. "
            "Expected _annotations.coco.json in the configured train directory."
        )
    try:
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON annotation file: {annotation_path}") from exc

    if not isinstance(coco, dict):
        raise TypeError("COCO JSON root must be an object")
    categories = coco.get("categories")
    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(categories, list) or not isinstance(images, list) or not isinstance(annotations, list):
        raise TypeError("COCO JSON must contain list-valued categories, images, and annotations")
    if not all(isinstance(item, dict) for item in categories + images + annotations):
        raise TypeError("COCO categories, images, and annotations must contain objects")
    category_ids = [item.get("id") for item in categories]
    if any(value is None for value in category_ids) or len(set(category_ids)) != len(category_ids):
        raise ValueError("COCO categories must have unique ids")
    class_names = [str(item.get("name", "")).strip() for item in categories]
    if not class_names or any(not name for name in class_names):
        raise ValueError("COCO categories must have non-empty names")
    category_to_index = {category_id: index for index, category_id in enumerate(category_ids)}

    image_by_id: dict[Any, dict[str, Any]] = {}
    for item in images:
        image_id = item.get("id")
        filename = item.get("file_name")
        width, height = item.get("width"), item.get("height")
        if image_id is None or not isinstance(filename, str) or not filename:
            raise ValueError("Every COCO image needs an id and file_name")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions for COCO image {filename!r}")
        if image_id in image_by_id:
            raise ValueError(f"Duplicate COCO image id: {image_id}")
        image_by_id[image_id] = item

    output_dir = output_dir.expanduser().resolve()
    image_output = output_dir / "images"
    label_output = output_dir / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    boxes_by_image: dict[Any, list[str]] = {image_id: [] for image_id in image_by_id}
    skipped_annotations = 0
    for annotation in annotations:
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        image = image_by_id.get(image_id)
        if image is None or category_id not in category_to_index:
            skipped_annotations += 1
            continue
        box = _validate_box(annotation.get("bbox"), image["width"], image["height"])
        if box is None:
            skipped_annotations += 1
            continue
        boxes_by_image[image_id].append(
            f"{category_to_index[category_id]} " + " ".join(f"{value:.8f}" for value in box)
        )

    copied_images = 0
    for image_id, image in sorted(image_by_id.items(), key=lambda pair: str(pair[1]["file_name"])):
        source = _safe_source(images_dir, image["file_name"])
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            raise FileNotFoundError(f"COCO image is missing or unsupported: {source}")
        destination = image_output / source.name
        shutil.copy2(source, destination)
        (label_output / f"{source.stem}.txt").write_text(
            "\n".join(boxes_by_image[image_id]) + ("\n" if boxes_by_image[image_id] else ""),
            encoding="utf-8",
        )
        copied_images += 1
    return {
        "class_names": class_names,
        "images": copied_images,
        "annotations": sum(len(items) for items in boxes_by_image.values()),
        "skipped_annotations": skipped_annotations,
        "output_dir": str(output_dir),
    }


def _group_key(image_path: Path) -> str:
    """Group Roboflow-style derivatives by their pre-hash stem."""
    return image_path.stem.split(".rf.", 1)[0]


def split_dataset(source_dir: Path, output_dir: Path, validation_fraction: float = 0.2, seed: int = 42) -> dict[str, int]:
    """Split original images by group before any augmentation is introduced."""
    source_dir = require_directory(source_dir, "converted image directory")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    images = sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) < 2:
        raise ValueError("At least two images are required for a train/validation split")
    groups: dict[str, list[Path]] = {}
    for image in images:
        groups.setdefault(_group_key(image), []).append(image)
    if len(groups) < 2:
        raise ValueError("At least two independent image groups are required for a split")
    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    val_group_count = max(1, round(len(group_names) * validation_fraction))
    val_groups = set(group_names[:val_group_count])
    selected = {"train": [image for key, items in groups.items() if key not in val_groups for image in items],
                "val": [image for key, items in groups.items() if key in val_groups for image in items]}
    if not selected["train"] or not selected["val"]:
        raise ValueError("Split produced an empty train or validation set")

    output_dir = output_dir.expanduser().resolve()
    for split_name, split_images in selected.items():
        image_destination = output_dir / "images" / split_name
        label_destination = output_dir / "labels" / split_name
        image_destination.mkdir(parents=True, exist_ok=True)
        label_destination.mkdir(parents=True, exist_ok=True)
        for old_file in (*image_destination.iterdir(), *label_destination.iterdir()):
            if old_file.is_file() or old_file.is_symlink():
                old_file.unlink()
        for image in sorted(split_images):
            shutil.copy2(image, image_destination / image.name)
            label = source_dir.parent / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"Missing label for image {image.name}: {label}")
            shutil.copy2(label, label_destination / label.name)
    return {name: len(items) for name, items in selected.items()}


def write_dataset_yaml(dataset_root: Path, class_names: list[str], destination: Path) -> Path:
    """Write an Ultralytics dataset config with paths relative to its root."""
    if not class_names:
        raise ValueError("class_names must not be empty")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write the dataset configuration; install requirements.txt") from exc
    dataset_root = dataset_root.expanduser().resolve()
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"path": str(dataset_root), "train": "images/train", "val": "images/val", "nc": len(class_names), "names": class_names}
    destination.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return destination
