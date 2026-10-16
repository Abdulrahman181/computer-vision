"""Portable malaria-cell image classification pipeline.

The TensorFlow import is intentionally lazy so repository validation and tests can
run without the heavy ML dependency or the image dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})


@dataclass(frozen=True)
class DatasetPaths:
    """Validated paths for the two immutable dataset splits."""

    root: Path
    train: Path
    test: Path


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _class_directories(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_dir())


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in image_size
        )
    ):
        raise ValueError("image_size must be a tuple of two positive integers")
    return image_size


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _class_image_counts(directory: Path) -> dict[str, int]:
    return {
        class_dir.name: len(_image_files(class_dir))
        for class_dir in _class_directories(directory)
    }


def resolve_dataset_dir(root: str | Path) -> DatasetPaths:
    """Validate the expected directory layout and return absolute paths.

    The function does not download, generate, or silently substitute data. It
    raises an actionable error when the dataset is absent or malformed.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root_path}. Set MALARIA_DATA_DIR or pass --data-dir "
            "to a directory containing training_set/ and testing_set/."
        )

    train_dir = root_path / "training_set"
    test_dir = root_path / "testing_set"
    missing = [str(path) for path in (train_dir, test_dir) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Dataset is incomplete; missing directory/directories: "
            + ", ".join(missing)
            + ". Expected training_set/<class>/ and testing_set/<class>/."
        )

    train_classes = _class_directories(train_dir)
    test_classes = _class_directories(test_dir)
    train_names = [path.name for path in train_classes]
    test_names = [path.name for path in test_classes]
    if len(train_names) != 2 or train_names != test_names:
        raise ValueError(
            "training_set and testing_set must contain the same exactly two class directories; "
            f"found training={train_names!r}, testing={test_names!r}."
        )

    counts = {
        "training_set": _class_image_counts(train_dir),
        "testing_set": _class_image_counts(test_dir),
    }
    empty = [
        f"{split}/{label}"
        for split, class_counts in counts.items()
        for label, count in class_counts.items()
        if count == 0
    ]
    if empty:
        raise ValueError(f"No supported image files found in: {', '.join(empty)}.")

    return DatasetPaths(root=root_path, train=train_dir, test=test_dir)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_cross_split_duplicates(paths: DatasetPaths) -> list[tuple[Path, Path]]:
    """Find exact byte-identical files shared by train and test.

    This catches a common form of image leakage. It is deliberately based on
    file content rather than file names; semantic near-duplicates still need a
    dataset-level review.
    """

    train_by_hash: dict[str, Path] = {}
    for path in _image_files(paths.train):
        train_by_hash.setdefault(_sha256(path), path)

    duplicates: list[tuple[Path, Path]] = []
    for path in _image_files(paths.test):
        train_path = train_by_hash.get(_sha256(path))
        if train_path is not None:
            duplicates.append((train_path, path))
    return duplicates


def find_within_split_label_conflicts(paths: DatasetPaths) -> list[tuple[Path, Path]]:
    """Find identical bytes assigned to different classes within one split."""

    conflicts: list[tuple[Path, Path]] = []
    for split in (paths.train, paths.test):
        by_hash: dict[str, tuple[str, Path]] = {}
        for path in _image_files(split):
            digest = _sha256(path)
            relative_parts = path.relative_to(split).parts
            if not relative_parts:
                continue
            label = relative_parts[0]
            previous = by_hash.get(digest)
            if previous is not None and previous[0] != label:
                conflicts.append((previous[1], path))
            else:
                by_hash.setdefault(digest, (label, path))
    return conflicts


def _validate_image_contents(paths: DatasetPaths) -> None:
    """Reject files with image extensions that Pillow cannot decode."""

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "Pillow is required to validate dataset images; install requirements.txt first."
        ) from exc

    invalid: list[Path] = []
    for path in _image_files(paths.train) + _image_files(paths.test):
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError):
            invalid.append(path)
    if invalid:
        examples = ", ".join(str(path) for path in invalid[:3])
        raise ValueError(f"Found {len(invalid)} unreadable image file(s); examples: {examples}")


def validate_dataset_integrity(paths: DatasetPaths) -> None:
    """Fail closed on duplicates, contradictory labels, or unreadable images."""

    duplicates = find_cross_split_duplicates(paths)
    if duplicates:
        examples = "; ".join(f"{train} == {test}" for train, test in duplicates[:3])
        raise ValueError(
            f"Detected {len(duplicates)} exact train/test duplicate image(s), "
            "indicating data leakage. "
            f"Examples: {examples}"
        )
    conflicts = find_within_split_label_conflicts(paths)
    if conflicts:
        examples = "; ".join(f"{first} == {second}" for first, second in conflicts[:3])
        raise ValueError(
            f"Detected {len(conflicts)} exact duplicate image(s) assigned to different classes "
            f"within a split. Examples: {examples}"
        )
    _validate_image_contents(paths)


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds when the ML stack is available."""

    _validate_seed(seed)
    import random

    random.seed(seed)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "NumPy is required for training; install requirements.txt first."
        ) from exc
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "TensorFlow is required for training; install requirements.txt first."
        ) from exc
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        # Older TensorFlow builds may not expose this switch or may reject it
        # after device initialization. The seed is still applied above.
        pass


def build_model(image_size: tuple[int, int] = (64, 64)) -> Any:
    """Build the binary CNN using current TensorFlow/Keras layer signatures."""

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "TensorFlow is required to build the model; install requirements.txt first."
        ) from exc

    height, width = _validate_image_size(image_size)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(height, width, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def create_generators(
    paths: DatasetPaths,
    image_size: tuple[int, int] = (64, 64),
    batch_size: int = 32,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Any, Any, Any]:
    """Create train/validation generators from training data and a test generator.

    The test split is never used for augmentation, model selection, or early
    stopping. Validation is carved out of training_set with a fixed seed.
    """

    _validate_image_size(image_size)
    _validate_seed(seed)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "TensorFlow is required for image generators; install requirements.txt first."
        ) from exc

    augmentation = ImageDataGenerator(
        rescale=1.0 / 255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_fraction,
    )
    evaluation = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=validation_fraction)
    common = {
        "target_size": image_size,
        "batch_size": batch_size,
        "class_mode": "binary",
        "classes": sorted(path.name for path in _class_directories(paths.train)),
    }
    train = augmentation.flow_from_directory(
        paths.train, subset="training", seed=seed, shuffle=True, **common
    )
    validation = evaluation.flow_from_directory(
        paths.train, subset="validation", seed=seed, shuffle=False, **common
    )
    test = ImageDataGenerator(rescale=1.0 / 255.0).flow_from_directory(
        paths.test,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        classes=sorted(path.name for path in _class_directories(paths.test)),
        shuffle=False,
    )
    if train.class_indices != validation.class_indices or train.class_indices != test.class_indices:
        raise ValueError("Class-to-index mappings differ between data splits; refusing to train.")
    expected_labels = set(range(len(common["classes"])))
    for name, generator in (("training", train), ("validation", validation), ("testing", test)):
        if generator.samples == 0 or set(generator.classes) != expected_labels:
            raise ValueError(
                f"The {name} split must contain at least one image for every class; "
                f"found {generator.samples} samples with labels {sorted(set(generator.classes))}."
            )
    return train, validation, test


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def classification_metrics(
    y_true: Iterable[float], y_probability: Iterable[float], threshold: float = 0.5
) -> dict[str, float | None]:
    """Return thresholded binary metrics without claiming undefined values."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "NumPy is required to calculate metrics; install requirements.txt first."
        ) from exc

    truth = np.asarray(list(y_true), dtype=float).reshape(-1)
    probability = np.asarray(list(y_probability), dtype=float).reshape(-1)
    if isinstance(threshold, bool):
        raise ValueError("threshold must be a finite number between 0 and 1.")
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite number between 0 and 1.") from exc
    if truth.size == 0 or truth.size != probability.size:
        raise ValueError("y_true and y_probability must be non-empty and have equal length.")
    if not np.isfinite(truth).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    probabilities_in_range = (0 <= probability).all() and (probability <= 1).all()
    if (
        not np.isfinite(probability).all()
        or not probabilities_in_range
        or not np.isfinite(threshold_value)
        or not 0 <= threshold_value <= 1
    ):
        raise ValueError(
            "Probabilities must be finite values between 0 and 1, and threshold must be "
            "between 0 and 1."
        )
    truth = truth.astype(np.int64)
    predicted = (probability >= threshold_value).astype(np.int64)
    tp = int(((truth == 1) & (predicted == 1)).sum())
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "threshold": threshold_value,
        "accuracy": _safe_ratio(tp + tn, truth.size),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "true_positives": float(tp),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
    }


def run_experiment(
    data_dir: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    validation_fraction: float = 0.2,
    image_size: tuple[int, int] = (64, 64),
    seed: int = 42,
) -> dict[str, Any]:
    """Train on training_set, select on its validation subset, and evaluate once on testing_set."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    paths = resolve_dataset_dir(data_dir)
    validate_dataset_integrity(paths)
    set_global_seed(seed)
    train, validation, test = create_generators(
        paths,
        image_size=image_size,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    model = build_model(image_size)
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "TensorFlow is required for training; install requirements.txt first."
        ) from exc
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    history = model.fit(train, epochs=epochs, validation_data=validation, callbacks=[early_stop])
    test_results = model.evaluate(test, verbose=0, return_dict=True)
    test.reset()
    probabilities = model.predict(test, verbose=0).reshape(-1)
    metrics = classification_metrics(test.classes, probabilities)
    return {
        "seed": seed,
        "image_size": list(image_size),
        "class_indices": test.class_indices,
        "samples": {"train": train.samples, "validation": validation.samples, "test": test.samples},
        "test_loss": float(test_results["loss"]),
        "test_metrics": {key: float(value) for key, value in test_results.items() if key != "loss"},
        "threshold_metrics": metrics,
        "history": {
            key: [float(value) for value in values] for key, values in history.history.items()
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=os.environ.get("MALARIA_DATA_DIR", "data/Malaria Cells")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, help="Optional JSON output path for measured results."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        results = run_experiment(
            args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    serialized = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
