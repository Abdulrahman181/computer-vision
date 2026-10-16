# Self-Driving Car Semantic Segmentation with U-Net

This repository contains a TensorFlow/Keras U-Net experiment for semantic segmentation of urban driving scenes and a Streamlit inference app. The training notebook is designed for the **Cityscapes image-pairs** layout; the dataset and trained model are intentionally not committed.

## Repository status

- Training and evaluation live in `autonomous-driving-scene-segmentation.ipynb`.
- `stream_segmentation.py` serves an existing `.keras` model; it does not train a model.
- `segmentation_utils.py` contains dependency-light inference contracts used by the UI and tests.
- `Data` is only a historical placeholder. It is not a dataset.
- No trustworthy benchmark metric is committed. Run training/evaluation with the dataset before reporting metrics.

## Data and model prerequisites

The notebook requires a local directory containing these non-empty directories:

```text
<CITYSCAPES_DATA_ROOT>/train/*.jpg
<CITYSCAPES_DATA_ROOT>/val/*.jpg
```

Each image pair is expected to be one PNG/JPEG file containing an RGB input image immediately followed by an RGB color mask of equal width. The total image width must therefore be even. The notebook validates that the directories exist, are non-empty, have readable three-channel pair files, reject unsupported mask colors, and have no filename overlap between train and validation. It never deletes files from the dataset.

Cityscapes is licensed data. Obtain it from the [official Cityscapes website](https://www.cityscapes-dataset.com/) or from a permitted image-pairs export, and follow its license terms. Set the path explicitly:

```bash
export CITYSCAPES_DATA_ROOT=/absolute/path/to/cityscapes_data
```

The Streamlit app requires a trained model artifact. The artifact must expose one image input shaped `[batch, height, width, 3]` and one segmentation output shaped `[batch, height, width, 8]` (height/width may be dynamic):

```bash
export CITYSCAPES_MODEL=/absolute/path/to/cityscapes_unet_model.keras
```

If this variable is unset, the app looks for `cityscapes_unet_model.keras` in the repository root and stops with an actionable error if it is absent. No model is included in this repository.

## Installation

Use Python 3.10 or 3.11 in a virtual environment. TensorFlow is a large dependency and may require platform-specific installation choices.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt  # optional static checks/tests
```

## Validation before training

These checks do not require TensorFlow or the dataset:

```bash
python validate_project.py
ruff check stream_segmentation.py validate_project.py tests
python -m unittest discover -s tests -v
```

## Training and evaluation

Open the notebook after setting `CITYSCAPES_DATA_ROOT` and run it from the first cell. The notebook sets deterministic seeds, uses the supplied train/validation split without creating a new overlapping split, and computes pixel accuracy, loss, per-class IoU, mean IoU over classes present in the evaluated set, and per-class/mean Dice from a confusion matrix. These metrics are only produced after a real run; this repository does not claim precomputed results.

The notebook writes model and demo artifacts under `ARTIFACT_DIR` (default: `artifacts/`):

```bash
export ARTIFACT_DIR=/absolute/path/to/output-directory
```

A missing dataset is a hard error with the expected path. A missing model is a hard error in the app. Do not create placeholder data or metrics to bypass either check.

## Streamlit inference

```bash
streamlit run stream_segmentation.py
```

Upload an image or a video after configuring `CITYSCAPES_MODEL`. The app checks model input/output shape and class count before rendering predictions, normalizes logits when necessary, validates output spatial dimensions, and refuses invalid class masks rather than reporting misleading coverage percentages. Camera mode processes one frame per Streamlit run rather than entering an uninterruptible Python loop.

## Limitations and audit notes

- The notebook uses the dataset's existing train/validation directories; it cannot establish that the upstream split is scene-disjoint beyond checking filename overlap.
- The eight-class color grouping is a project-specific mapping, not the full official Cityscapes 30/19-class taxonomy. Confirm the mask encoding before training.
- No end-to-end training, GPU run, or metric reproduction was possible during repository validation because the dataset and model artifact are not present in the repository and TensorFlow is not assumed to be installed.
