"""Streamlit inference UI for a trained eight-class U-Net model."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from segmentation_utils import (
    class_coverage,
    prediction_to_probabilities,
    validate_class_mask,
    validate_model_input_shape,
    validate_model_output_shape,
)

NUM_CLASSES = 8
CLASS_NAMES = [
    "Void/Background",
    "Road",
    "Construction",
    "Object",
    "Nature",
    "Sky",
    "Human",
    "Vehicle",
]
COLORS = np.array(
    [
        [0, 0, 0],
        [128, 64, 128],
        [70, 70, 70],
        [250, 170, 30],
        [107, 142, 35],
        [70, 130, 180],
        [220, 20, 60],
        [0, 0, 142],
    ],
    dtype=np.uint8,
)


@st.cache_resource
def load_model(model_path: str):
    """Load and validate one image-to-eight-class segmentation model."""
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Set CITYSCAPES_MODEL to an existing "
            "trained .keras file; no model is bundled with this repository."
        )
    model = tf.keras.models.load_model(path)
    validate_model_input_shape(model.input_shape)
    validate_model_output_shape(model.output_shape, NUM_CLASSES)
    return model


def _target_size(model_instance) -> tuple[int, int]:
    shape = validate_model_input_shape(model_instance.input_shape)
    if shape[1] is None or shape[2] is None:
        return (256, 256)
    return (int(shape[2]), int(shape[1]))


def process_image(image_pil: Image.Image, model_instance=None):
    """Return colored mask, overlay, confidence percentage, and class mask."""
    active_model = model_instance if model_instance is not None else globals().get("model")
    if active_model is None:
        raise RuntimeError("No model is available; pass model_instance or start the Streamlit app.")
    target_size = _target_size(active_model)
    image_rgb = image_pil.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    image_array = np.asarray(image_rgb, dtype=np.float32) / 255.0
    raw_prediction = active_model.predict(image_array[None, ...], verbose=0)
    prediction = prediction_to_probabilities(raw_prediction, NUM_CLASSES)
    if prediction.shape[0] != 1 or prediction.shape[1:3] != image_array.shape[:2]:
        raise ValueError(
            "Model output spatial shape must match the resized input: "
            f"prediction={prediction.shape}, input={image_array.shape}."
        )

    confidence_map = np.max(prediction[0], axis=-1)
    mean_confidence = float(np.mean(confidence_map) * 100.0)
    pred_mask = validate_class_mask(np.argmax(prediction[0], axis=-1), NUM_CLASSES)
    colored_mask = COLORS[pred_mask]
    original_array = np.asarray(image_rgb, dtype=np.uint8)
    overlay = cv2.addWeighted(original_array, 0.6, colored_mask, 0.4, 0)
    return colored_mask, overlay, mean_confidence, pred_mask


def get_class_statistics(pred_mask: np.ndarray) -> dict[str, float]:
    """Return coverage percentages for valid predicted classes."""
    return class_coverage(pred_mask, CLASS_NAMES)


def analyze_video(uploaded_video, model_instance=None) -> None:
    """Process at most 300 frames from a video upload and always release resources."""
    suffix = Path(uploaded_video.name).suffix.lower() or ".mp4"
    temp_path: Path | None = None
    cap = None
    try:
        payload = uploaded_video.getvalue()
        if not payload:
            st.error("The uploaded video is empty.")
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(payload)
            temp_path = Path(temp_file.name)
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            st.error("The uploaded video could not be opened by OpenCV.")
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = min(frame_count if frame_count > 0 else 300, 300)
        frame_slot, metric_slot = st.empty(), st.empty()
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, overlay, confidence, _ = process_image(Image.fromarray(frame_rgb), model_instance)
            frame_slot.image(overlay, channels="RGB", use_container_width=True)
            metric_slot.metric("Mean pixel confidence", f"{confidence:.2f}%")
    finally:
        if cap is not None:
            cap.release()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Cityscapes Analytics", layout="wide")
    st.title("Autonomous Driving Scene Analytics System")
    st.write("Semantic segmentation with confidence and class-coverage summaries.")

    model_path = os.environ.get("CITYSCAPES_MODEL", "cityscapes_unet_model.keras")
    try:
        active_model = load_model(model_path)
    except Exception as exc:  # Streamlit boundary: backend errors vary by TensorFlow version.
        st.error(f"Unable to load or validate the model artifact: {exc}")
        st.stop()
    st.sidebar.success("System Ready")
    st.sidebar.title("Control Panel")
    mode = st.sidebar.selectbox("Select Input Mode", ["Image Analysis", "Video Analysis", "Live Camera"])

    if mode == "Image Analysis":
        st.subheader("Single Image Analytics")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Input", use_container_width=True)
            if st.button("Analyze Scene"):
                with st.spinner("Analyzing..."):
                    _, overlay, confidence, pred_mask = process_image(image, active_model)
                with col2:
                    st.image(overlay, caption=f"Output (Confidence: {confidence:.2f}%)", use_container_width=True)
                stats = get_class_statistics(pred_mask)
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean pixel confidence", f"{confidence:.2f}%")
                m2.metric("Road coverage", f"{stats.get('Road', 0.0):.2f}%")
                m3.metric("Vehicle coverage", f"{stats.get('Vehicle', 0.0):.2f}%")
                st.markdown("### Scene Statistics")
                st.table(pd.DataFrame(list(stats.items()), columns=["Object Class", "Coverage (%)"]))
    elif mode == "Video Analysis":
        st.subheader("Video File Analytics (first 300 frames)")
        uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
        if uploaded_video and st.button("Analyze Video"):
            with st.spinner("Analyzing video..."):
                analyze_video(uploaded_video, active_model)
    else:
        st.subheader("Camera Snapshot Analytics")
        st.caption("The app processes one snapshot per run to avoid an uninterruptible server-side camera loop.")
        camera_file = st.camera_input("Capture a frame")
        if camera_file:
            camera_image = Image.open(camera_file).convert("RGB")
            _, overlay, confidence, pred_mask = process_image(camera_image, active_model)
            st.image(overlay, caption=f"Output (Confidence: {confidence:.2f}%)", use_container_width=True)
            st.table(pd.DataFrame(list(get_class_statistics(pred_mask).items()), columns=["Object Class", "Coverage (%)"]))


if __name__ == "__main__":
    main()
