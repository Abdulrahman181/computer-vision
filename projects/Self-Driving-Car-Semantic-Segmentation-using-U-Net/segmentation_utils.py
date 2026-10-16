"""Pure NumPy contracts shared by the inference UI and tests.

These helpers intentionally do not import TensorFlow, Streamlit, OpenCV, or
Pandas so static and contract tests can run without the heavyweight runtime.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_model_output_shape(output_shape: object, num_classes: int) -> tuple[object, ...]:
    """Validate a single segmentation output shape and return it as a tuple."""
    if isinstance(output_shape, list):
        raise ValueError("The model must expose one tensor output, not multiple outputs.")
    try:
        shape = tuple(output_shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"Model output shape is not a sequence: {output_shape!r}") from exc
    if len(shape) != 4:
        raise ValueError(
            "Expected model output shape [batch, height, width, classes], "
            f"received {output_shape!r}."
        )
    if shape[-1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} output classes, received shape {output_shape!r}."
        )
    for dimension_name, dimension in zip(("height", "width"), shape[1:3]):
        if dimension is not None and (not isinstance(dimension, (int, np.integer)) or dimension <= 0):
            raise ValueError(f"Model output {dimension_name} must be positive or None; received {dimension!r}.")
    return shape


def validate_model_input_shape(input_shape: object) -> tuple[object, ...]:
    """Require a single image input with three channels."""
    if isinstance(input_shape, list):
        raise ValueError("The model must expose one image input, not multiple inputs.")
    try:
        shape = tuple(input_shape)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"Model input shape is not a sequence: {input_shape!r}") from exc
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError(f"Expected model input shape [batch, height, width, 3], received {input_shape!r}.")
    for dimension_name, dimension in zip(("height", "width"), shape[1:3]):
        if dimension is not None and (not isinstance(dimension, (int, np.integer)) or dimension <= 0):
            raise ValueError(f"Model input {dimension_name} must be positive or None; received {dimension!r}.")
    return shape


def prediction_to_probabilities(prediction: object, num_classes: int) -> np.ndarray:
    """Return finite per-pixel probabilities, applying softmax to logits if needed."""
    array = np.asarray(prediction, dtype=np.float32)
    if array.ndim != 4 or array.shape[-1] != num_classes:
        raise ValueError(f"Unexpected prediction shape: {array.shape}; expected [batch, height, width, {num_classes}].")
    if not np.isfinite(array).all():
        raise ValueError("Model prediction contains NaN or infinite values.")
    sums = array.sum(axis=-1, keepdims=True)
    is_probability = np.all(array >= 0) and np.allclose(sums, 1.0, atol=1e-4)
    if is_probability:
        return array
    shifted = array - np.max(array, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def validate_class_mask(pred_mask: object, num_classes: int) -> np.ndarray:
    """Validate a non-empty integer class mask and return int32 values."""
    array = np.asarray(pred_mask)
    if array.size == 0:
        raise ValueError("Class mask must not be empty.")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError("Class mask values must be finite integers.")
    array = array.astype(np.int32, copy=False)
    if np.any(array < 0) or np.any(array >= num_classes):
        raise ValueError(f"Class mask contains values outside [0, {num_classes - 1}].")
    return array


def class_coverage(pred_mask: object, class_names: Sequence[str]) -> dict[str, float]:
    """Return percentages for present classes; percentages sum to 100."""
    validated = validate_class_mask(pred_mask, len(class_names))
    counts = np.bincount(validated.reshape(-1), minlength=len(class_names))
    total = validated.size
    return {
        class_names[class_id]: round(float(count / total * 100.0), 2)
        for class_id, count in enumerate(counts)
        if count
    }
