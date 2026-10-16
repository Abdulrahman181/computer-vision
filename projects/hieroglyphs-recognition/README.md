# Egyptian Hieroglyphs Recognition

This repository contains a reproducible transfer-learning notebook for **Egyptian hieroglyph classification** from cropped object-detection annotations. The dataset is deliberately not committed because of size and licensing constraints.

## Data contract

Set `HIEROGLYPHS_DATA_DIR` to a dataset root with this layout:

```text
$HIEROGLYPHS_DATA_DIR/
├── train/_annotations.csv
├── valid/_annotations.csv
└── test/_annotations.csv
```

Each annotation CSV must contain `filename`, `class`, `xmin`, `ymin`, `xmax`, and `ymax`, and every referenced image must exist inside its split directory. The notebook validates the schema and boxes, rejects exact filename overlap between splits as a leakage safeguard, learns the class mapping from **train only**, and reports a clear error when data is absent. No dataset or metric result is claimed in this repository.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HIEROGLYPHS_DATA_DIR=/absolute/path/to/data
python validate_project.py
jupyter notebook egyptian-hieroglyphs-using-tl-cnn.ipynb
```

Run notebook cells from top to bottom. Generated models are written to `artifacts/` by default and are ignored by Git. The default model is `ResNet50` with ImageNet weights; use `HIEROGLYPHS_USE_IMAGENET_WEIGHTS=0` when weight download is unavailable (that run is not comparable to ImageNet transfer learning). Optional controls include `HIEROGLYPHS_SEED`, `HIEROGLYPHS_MODEL`, `HIEROGLYPHS_IMAGE_SIZE`, `HIEROGLYPHS_BATCH_SIZE`, `HIEROGLYPHS_EPOCHS`, and `HIEROGLYPHS_ARTIFACT_DIR`.

## Validation and evaluation notes

- `validate_project.py` performs dependency-free notebook JSON/AST checks and runs Ruff checks when Ruff is installed.
- Validation data is used during checkpoint selection; test data is loaded and evaluated only after training.
- Test metrics include accuracy, weighted precision, weighted recall, weighted F1, and loss. They are computed only when the real dataset and ML dependencies are available.
- Results depend on the dataset split, random seed, preprocessing, TensorFlow version, and hardware. This educational/research artifact is not a production system.
