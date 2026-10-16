# Malaria cell image classification

This repository contains a small TensorFlow/Keras CNN for binary malaria-cell image classification. **The image dataset and trained artifacts are not included**, so this repository cannot produce or claim model metrics by itself.

## Setup

Use a virtual environment and install the pinned major-version ranges:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run repository validation without TensorFlow or image data:

```bash
python validate_repository.py
pytest -q
```

## Dataset contract

Obtain the dataset from an authorized source and keep it outside Git. Set `MALARIA_DATA_DIR` or pass `--data-dir`:

```text
<dataset-root>/
├── training_set/
│   ├── <class-a>/image files...
│   └── <class-b>/image files...
└── testing_set/
    ├── <class-a>/image files...
    └── <class-b>/image files...
```

Both splits must contain the same exactly two class-directory names, with at least one supported image in every class (`bmp`, `gif`, `jpeg`, `jpg`, `png`, `tif`, or `tiff`). Missing, undecodable, or malformed data stops with an actionable error; no data is generated or silently substituted. Exact byte-identical images across training and testing, and identical bytes assigned to different classes within a split, are rejected as leakage or label contradiction.

## Training and evaluation

```bash
python malaria_pipeline.py \
  --data-dir /path/to/dataset-root \
  --epochs 10 \
  --batch-size 32 \
  --seed 42 \
  --output results.json
```

The pipeline creates a fixed-seed validation subset from `training_set`. Augmentation and early stopping use only that validation subset; `testing_set` is held out until the final evaluation. The test generator is not shuffled, and the reported threshold metrics include accuracy, precision, recall, specificity, F1, and confusion counts. A metric is reported as `null` when its denominator is undefined rather than being invented.

`malaria-pred.ipynb` is a thin, cleaned notebook wrapper around the same pipeline. Set `MALARIA_DATA_DIR` before running it. It has no saved outputs and will stop clearly when data or TensorFlow is unavailable.

## Reproducibility and limitations

The default seed is 42 and the pipeline requests TensorFlow deterministic operations when supported. Exact reproducibility can still depend on TensorFlow, CUDA/CPU kernels, hardware, and library versions. The repository does not include a data provenance record, model weights, or measured performance. Exact-byte checks do not detect semantic near-duplicates or patient-level overlap. Do not interpret this educational classifier as a clinical diagnostic device; any clinical use would require an appropriately designed, independently validated study.
