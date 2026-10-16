# Engineering Review — hieroglyphs-recognition

**Repository:** `Abdulrahman181/hieroglyphs-recognition`  
**Review scope:** Python/Notebook/README/dependencies/project hygiene  
**Review date:** 2026-10-13  
**Execution environment:** shared Sandbox; no dataset was present in the repository

## Executive summary

The repository consisted of one training notebook plus minimal documentation and an unbounded dependency list. The notebook was syntactically parseable, but it was not safe to rerun portably or to trust its evaluation flow: it assumed a local dataset, used independent one-hot encoders for each split, hard-coded 95 output classes, had no explicit reproducibility controls, lacked robust file/box validation, and printed a test F1 value that was not computed in the original evaluation cell. The repository also did not contain a static validation command.

The notebook has been rewritten as a clean, output-free, data-dependent workflow. It now validates the dataset contract before loading images, learns labels from the training split only, rejects exact filename overlap between splits, protects against path traversal, uses portable `pathlib` paths, derives the output dimension from the training classes, sets deterministic seeds where TensorFlow supports them, uses sparse labels consistently, and evaluates the untouched test split only after training. A dependency-free `validate_project.py` script was added for CI/lightweight environments, and the README, requirements, and ignore rules were expanded.

## Findings and remediation

### Critical/high impact findings

1. **Original notebook had no usable data fallback and relied on implicit working-directory state.** Missing CSVs or images would fail with low-level errors, and paths were built with string interpolation.  
   **Remediation:** `HIEROGLYPHS_DATA_DIR` is documented and resolved with `pathlib`; missing split/CSV/image errors identify the exact expected path and configuration. Annotation paths are checked to remain inside their split directory.

2. **Class encoding was fit independently on train, validation, and test.** This can produce incompatible one-hot column ordering and invalid evaluation when a split has a different class set.  
   **Remediation:** class names are sorted and mapped from train only; validation/test classes absent from train are rejected; sparse integer labels and `sparse_categorical_crossentropy` remove encoder-shape ambiguity.

3. **The classifier output was hard-coded to 95 classes.** This could silently mismatch the actual dataset.  
   **Remediation:** output width is `len(class_names)` from the validated training annotations.

4. **The original metric cell referenced `f1_test` without assigning it.**  
   **Remediation:** test predictions are generated once and all test metrics, including weighted F1, are computed before reporting.

5. **Split/data leakage controls were absent.**  
   **Remediation:** exact filename overlap across train/valid/test is rejected before loading; validation is used for checkpoint selection and test data is evaluated only after fitting.

### Medium/quality findings

- Bounding-box coordinates and required columns were not validated; malformed or out-of-bounds crops could fail later or produce invalid inputs. The new workflow validates required columns, non-empty CSVs, missing values, integer coordinates, positive boxes, image bounds, and non-empty crops.
- Random preview selection was not seeded. Preview sampling is now deterministic and does not mutate training data.
- The original model head included both `Flatten` and `GlobalAveragePooling2D` on the same base output, leaving the unused `Flatten` result. The repaired head uses global average pooling once.
- The original workflow used TensorFlow metric names for precision/recall without robustly checking history keys. The repaired training metrics are deliberately limited to accuracy/loss, while sklearn computes explicit weighted test precision/recall/F1 with `zero_division=0`.
- Output artifacts used a bare filename. Checkpoints now go to configurable `HIEROGLYPHS_ARTIFACT_DIR` (default `artifacts/`), which is ignored by Git.
- The original README did not describe the exact CSV schema, test-evaluation policy, no-data behavior, or validation command. Documentation now does.
- Requirements were unbounded and omitted Jupyter despite the repository being notebook-first. Broad compatible ranges and `opencv-python-headless` were added. The environment did not have TensorFlow/OpenCV/scikit-learn installed, so dependency installation was not attempted and model training was not claimed.
- Old notebook outputs/execution counts were removed; the final notebook contains no saved outputs.

## Files changed

- `.gitignore` — ignores `artifacts/`, package metadata, and pytest cache in addition to existing datasets/models/caches.
- `README.md` — documents the dataset contract, commands, controls, leakage/evaluation policy, and limitations.
- `requirements.txt` — adds bounded versions for NumPy, pandas, Matplotlib, OpenCV headless, tqdm, TensorFlow, scikit-learn, and Jupyter.
- `egyptian-hieroglyphs-using-tl-cnn.ipynb` — rewritten into a portable, deterministic, validated workflow with dynamic class mapping and clean outputs.
- `validate_project.py` — new dependency-free static checker for required files, notebook JSON/AST, stale outputs, portability markers, and Ruff fatal/undefined-name checks when Ruff is available.
- `ENGINEERING_REVIEW.md` — this review.

## Validation performed

- `python3 validate_project.py` — **passed**.
- `ruff check --isolated --select E9,F63,F7,F821 egyptian-hieroglyphs-using-tl-cnn.ipynb validate_project.py` — **passed**.
- Python `ast.parse` over every notebook code cell — **passed** (8 code cells).
- Notebook JSON parse — **passed** (12 cells; 8 code cells).
- Stale notebook output/execution check — **passed** (0 outputs; all code execution counts are `None`).
- `git diff --check` — **passed**.
- Static validator rerun with `HIEROGLYPHS_DATA_DIR=/tmp/nonexistent-hieroglyphs` — **passed**, because static validation intentionally does not import ML packages or require data.

## Not run / limitations

A full notebook execution, model download, training run, and numerical metric report were **not performed**. The repository contains no dataset, and the Sandbox environment lacked TensorFlow, OpenCV, and scikit-learn. This report therefore makes no accuracy/F1 or runtime claim. After installing `requirements.txt` and providing a dataset satisfying the documented contract, run the notebook from top to bottom and record metrics from that real run.

## Local commit

A local-only commit was created after the review. No changes were pushed to GitHub.

## Second Senior Review (2026-10-16)

A second manual pass verified dynamic class mapping, path safety, sparse-label loss, notebook hygiene, and validator behavior. The missing regression-test layer was completed with `tests/test_validate_project.py`, covering validator execution, AST parsing, stale-output protection, and required data/model-contract markers. No dataset or TensorFlow runtime was present, so no training metrics were claimed.

Validation: `python3 validate_project.py`, `pytest -q` (2 passed), `compileall`, and `git diff --check` passed.
