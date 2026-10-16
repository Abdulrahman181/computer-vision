# Weed detection

This repository prepares a COCO-labelled weed dataset for YOLO object detection. It does **not** include the dataset or model weights. The notebook is a reproducible workflow template; it stops with an actionable error when required data or artifacts are absent.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export WEED_DATA_ROOT=/absolute/path/to/weed-detection
export WEED_ARTIFACT_ROOT=/absolute/path/to/weed-detection-artifacts
jupyter notebook weed-detection.ipynb
```

Expected input layout:

```text
$WEED_DATA_ROOT/
└── train/
    ├── _annotations.coco.json
    └── image files referenced by the COCO JSON
```

`WEED_DATA_ROOT` and `WEED_ARTIFACT_ROOT` are intentionally explicit. No Kaggle path, local user path, credentials, or download is assumed. Keep datasets, weights, and generated runs outside Git; `.gitignore` covers common artifact locations.

## Workflow guarantees

1. COCO files and referenced images are validated before conversion. Invalid boxes are skipped and counted; labels are never invented.
2. COCO file names are checked for path traversal before files are copied.
3. Splitting happens before any future offline augmentation, and Roboflow-style `.rf.` derivatives are grouped by their pre-hash stem. Ultralytics' online training augmentation can then run only inside the training split.
4. The seed is explicit (`WEED_SEED`, default `42`). Training is opt-in via `WEED_RUN_TRAINING=1`; the default notebook run only prepares data and reports what is missing.
5. Validation uses the validation split only. An independent test report is not claimed because this repository does not provide test labels.

## Validation

Run lightweight checks without data or GPU:

```bash
python -m pytest -q
ruff check .
python -m compileall -q weed_detection.py tests
```

Heavy YOLO training and inference are **not** run by the static test suite. They require installed Ultralytics dependencies, the real dataset, model weights (download permission or a local path), and enough compute.

## Audit record

See [`ENGINEERING_REVIEW.md`](ENGINEERING_REVIEW.md) for the original defects, changes made, validation performed, and remaining limitations. Existing notebook outputs and stale metrics were removed; no performance result in this repository should be treated as a fresh benchmark.
