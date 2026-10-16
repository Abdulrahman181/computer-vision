# Engineering review — Skin-cancer-prediction

**Date:** 2026-10-14
**Scope:** `Abdulrahman181/Skin-cancer-prediction` only
**Commit:** local commit created after validation (see Git history)

## Executive summary

The repository initially contained only `skincancer.ipynb`, with no dataset, dependency manifest, documentation, tests, or ignore rules. The notebook had a runnable-looking but non-reproducible training flow and used the directory named `test` as validation during training. The notebook has now been rewritten as a clean, statically validated workflow. No training, metrics, or model artifact are claimed because the repository does not contain image data and TensorFlow is not installed in this sandbox.

## Findings and remediation

| Finding | Severity | Remediation |
| --- | --- | --- |
| `train/` and `test/` were absent from the repository and paths were implicit relative paths | Critical for execution | Added an explicit `SKINCANCER_DATA_ROOT` configuration and clear `FileNotFoundError` before any generator is created; documented the required two-class directory layout. |
| The original flow passed `test_generator` as `validation_data` during all epochs | High: test leakage/model-selection bias | The revised flow creates a deterministic 20% validation split from `train/`; `test/` is not evaluated until after training. |
| No reproducibility controls | High | Added seed `42` through `tf.keras.utils.set_random_seed`, deterministic TensorFlow ops where supported, fixed generator seeds, and non-shuffled evaluation generators. Hardware/library nondeterminism is reported rather than hidden. |
| Evaluation reported only validation accuracy and did not use the held-out set | High | Added held-out loss/accuracy, confusion matrix, classification report, and ROC-AUC when both test classes are observed. No metrics are fabricated. |
| `Adam(lr=...)` used a deprecated argument | Medium | Replaced with `learning_rate=0.001`. |
| Optional `temp.jpg` and fixed `skin_cancer_classifier.h5` paths were undocumented and could fail or pollute the repository | Medium | Prediction is opt-in through `PREDICT_IMAGE` and fails clearly when explicitly requested but missing; output is configurable through `MODEL_PATH`, defaults to ignored `artifacts/skin_cancer_classifier.keras`, and is checked after saving. |
| Notebook contained historical execution outputs, including environment-specific paths and stale metrics | Medium | Cleared all code-cell outputs and execution counts. No old metrics are presented as current results. |
| No dependency, usage, or safety documentation | Medium | Added `requirements.txt`, `README.md`, `.gitignore`, and this review. README explicitly states that this is not a medical device or diagnosis. |
| No automated static validation | Medium | Added `scripts/validate_notebook.py` and fast pytest checks for JSON/Python syntax, clean outputs, required guards, and required project files. Ruff is used for linting. |

## Data leakage and evaluation assessment

The corrected notebook does not use the held-out `test/` directory for training or validation. The validation split is deterministic at the file-generator level. However, the repository provides no patient IDs, acquisition metadata, or duplicate manifests. Therefore this maintenance change **cannot establish patient-level independence** or rule out near-duplicate images crossing train/validation/test. A clinical-grade evaluation should split by patient (and ideally acquisition site/time), audit duplicates, and report confidence intervals and calibration.

The two folder names are treated as dataset labels, not medical truth. The sigmoid score is interpreted against a fixed 0.5 threshold for the demonstration only; threshold selection should be based on a prespecified validation policy and clinical cost analysis.

## Dependency and execution assessment

`requirements.txt` declares NumPy, Matplotlib, scikit-learn, TensorFlow, Jupyter, pytest, and Ruff with bounded major versions. TensorFlow is a heavy optional runtime dependency and is not available in this sandbox; no notebook execution or training was attempted. The absence of `train/` and `test/` is an intentional hard stop with a clear message, not a reason to invent sample data or metrics.

## Validation performed

- `python scripts/validate_notebook.py` — passed.
- `pytest -q` — passed after the report was added (tests do not require data or TensorFlow).
- `ruff check .` — passed after automatic import/style fixes.
- `python -m json.tool skincancer.ipynb` — passed.
- Every code cell was parsed with Python AST by the validator.
- Notebook outputs and execution counts are empty/`null`.

## Changed files

- `skincancer.ipynb` — cleaned and corrected workflow.
- `README.md` — setup, data contract, safety and usage documentation.
- `requirements.txt` — runtime and static/test dependencies.
- `.gitignore` — excludes local data, images, model artifacts and notebook caches.
- `scripts/validate_notebook.py` — dependency-free static validator.
- `tests/test_static_project.py` — fast repository tests.
- `ENGINEERING_REVIEW.md` — this review.

## Limitations and next steps

1. Provide a legally usable, patient-disjoint dataset outside Git and run the notebook in a compatible TensorFlow environment.
2. Add a dataset manifest with patient/group identifiers and a duplicate/near-duplicate audit before reporting performance.
3. Use an untouched external test set and prespecify thresholds; add sensitivity, specificity, PPV/NPV, calibration, confidence intervals, and subgroup analysis.
4. Treat model outputs as research artifacts only until clinical, regulatory, and safety review is complete.

## Second Senior Review

**Date:** 2026-10-16
**Scope:** all tracked files, the complete `skincancer.ipynb`, Python/config/test/README/requirements files, repository status, and the diff from the previous local audit commit.

### Residual findings and fixes

| Finding | Severity | Fix applied |
| --- | --- | --- |
| The preflight check accepted any regular file, while `flow_from_directory` only consumes supported image extensions. A class containing only text/unknown files could pass preflight and then fail or produce an empty generator. | High | Added Keras-compatible image-extension filtering and per-class effective image counts for both splits. |
| A 20% directory split can produce no validation image for a class when the class is too small, making validation unreliable or causing runtime failures. | High | Require at least `ceil(1 / validation_split)` (five at the fixed 20% split) supported training images per class, with a clear error. |
| Generator class order was left to implicit directory discovery. | Medium | Pass the validated `train_classes`/`test_classes` explicitly to every generator and retain mapping consistency checks. |
| `MODEL_PATH` accepted values that Keras could not save as a model artifact, and saving was not followed by a load check. | Medium | Require `.keras` or `.h5`, then reload the saved artifact with `compile=False` and verify input/output shapes. |
| Prediction code assumed binary indices without an explicit contract. | Medium | Restore/retain the optional prediction cell and reject any unexpected class-index mapping before reporting a label. |
| Static validation did not reject an empty/malformed `cells` collection and did not enforce the new data/artifact contracts. | Medium | Extended `scripts/validate_notebook.py` to validate the cell collection and require supported-image, minimum-data, and artifact-load logic. |
| The static test used a path mutation plus a late import, producing an E402 lint failure. | Low | Load the validator with `importlib.util` without altering import paths; Ruff now passes. |

### Review conclusions

- The held-out `test/` directory remains excluded from fitting and validation; it is evaluated only after training.
- The notebook has no embedded outputs or execution counts, and all code cells parse as Python AST.
- No dataset, model weights, secrets, or fabricated metrics were added. TensorFlow is not installed in this sandbox, so notebook execution, training, artifact serialization, and numerical metrics were intentionally not claimed.
- The review did not change the repository's dependency declarations; they remain documented in `requirements.txt` and require installation in a compatible Python/TensorFlow environment.

### Validation performed in this second review

- `python3 scripts/validate_notebook.py` — passed.
- `pytest -q` — **3 passed**.
- `ruff check .` — passed.
- `python3 -m json.tool skincancer.ipynb` — passed.
- `python3 -m compileall -q scripts tests` — passed.
- `git diff --check` — passed.
- Confirmed all code-cell outputs are empty and execution counts are `null`.
- Confirmed no untracked dataset, image, model, or generated artifact is staged for the commit.

### Remaining risks

1. No patient IDs, acquisition metadata, duplicate manifest, or near-duplicate audit exists; file-level splitting cannot prove patient-level independence.
2. The repository contains no data, so class balance, image decodability, actual TensorFlow version compatibility, GPU determinism, training convergence, and end-to-end artifact loading remain unverified here.
3. Folder names are dataset labels rather than clinical truth. The 0.5 threshold is a demonstration default, not a clinically validated operating point.
4. The small CNN and reported metrics, when eventually produced, require an untouched external test set, confidence intervals, calibration, subgroup analysis, and clinical/regulatory review before any medical use.
