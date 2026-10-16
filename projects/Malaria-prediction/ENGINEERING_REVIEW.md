# Engineering review: `Abdulrahman181/Malaria-prediction`

**Review scope:** all repository files present at the audit start, including `malaria-pred.ipynb`, its Python code cells, dependency declarations (none existed initially), and repository documentation (none existed initially).

## Baseline findings

The repository initially contained only one notebook. It had no README, dependency manifest, tests, validator, or ignore rules. The notebook had these engineering issues:

- It contained hard-coded Kaggle paths (`/kaggle/input/...`) and therefore could not run outside the original environment.
- It used removed or obsolete APIs and signatures: standalone `keras` imports, `Convolution2D` positional arguments, `Dense(output_dim=...)`, `fit_generator`, `evaluate_generator`, `nb_epoch`, and `nb_val_samples`.
- The notebook JSON was not schema-clean: markdown cells carried `execution_count`, and the file contained large stale outputs, including partial training logs and plots.
- The test directory was passed as `validation_data` and used for early stopping. This is test-set leakage and makes the reported test performance unsuitable as an untouched final estimate.
- The run had no explicit seed or deterministic-operation request, fixed arbitrary `steps_per_epoch`/validation sample counts, no reliable class-index handling, and accuracy was the only configured metric.
- A single-image prediction assumed class semantics instead of using a verified class-index mapping.
- The dataset, weights, and artifacts were not present, so no baseline metric can be verified or honestly reported.

## Remediation

- Added `malaria_pipeline.py`, a reusable CLI/module using current `tensorflow.keras` APIs with lazy TensorFlow imports. It validates the dataset contract, fails closed on absent/malformed data, uses a seeded validation subset from `training_set`, keeps `testing_set` untouched until final evaluation, and derives generator sizes from the data.
- Added exact byte-level cross-split duplicate detection. A duplicate raises an error rather than silently training on leaked data. Near-duplicates, patient-level overlap, and provenance remain dataset-level limitations and require domain metadata.
- Added reproducibility controls for Python, NumPy, and TensorFlow seeds, plus TensorFlow deterministic operations where supported. Reproducibility is still hardware/kernel dependent.
- Added threshold metrics (accuracy, precision, recall, specificity, F1, and confusion counts) computed from final test predictions. Undefined ratios are represented as `null`; no values are fabricated.
- Replaced `malaria-pred.ipynb` with a valid, output-free wrapper around the audited module. It has no hard-coded local/Kaggle paths and stops clearly when the configured data or TensorFlow is absent.
- Added `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `validate_repository.py`, and `tests/test_repository.py`.

## Validation performed

The following checks passed in the shared Sandbox after the changes:

```text
python validate_repository.py
Repository static validation passed: notebook JSON, outputs, syntax, and legacy paths checked.

python -m compileall -q malaria_pipeline.py validate_repository.py tests
pytest -q
5 passed in 0.33s

ruff check .
All checks passed!
```

The CLI was also invoked with a nonexistent data directory. It exited with code 2 and the actionable message that `MALARIA_DATA_DIR` or `--data-dir` must point to a directory containing `training_set/` and `testing_set/`.

## Not run / limitations

A complete training and evaluation run was **not** claimed. The repository has no image data or saved artifacts, and TensorFlow is not installed in the audit environment. Consequently, there are no measured accuracy, loss, or clinical-performance claims in this review. Installing the declared dependencies and supplying an authorized dataset are prerequisites for a real run. The dataset source/provenance, patient identifiers, acquisition grouping, and semantic near-duplicate policy are not included and must be reviewed before interpreting any future metrics.


## Second Senior Review

### Scope and findings

The second review re-read every tracked file, the complete notebook, Python/config/test/README/requirements files, and Git history/diff. The working tree was clean at review start. No credentials, hard-coded secrets, tracked dataset, model weights, generated result artifact, or additional hidden application/UI/API file was present.

The remaining actionable gaps were:

- Dataset validation checked extensions and aggregate split counts, but did not require every class to contain an image or verify that image bytes were decodable.
- Exact duplicate detection covered train/test leakage but did not catch identical bytes assigned to contradictory labels within one split.
- Public validation accepted malformed notebook root/cell/source structures and did not reject all obsolete training argument names.
- Metric input validation converted labels to integers, which could silently coerce non-integral numeric labels.

### Fixes in this review

- Added explicit image-size and non-negative-seed validation, per-class non-empty checks, Pillow `verify()` checks for every supported image, and within-split cross-label duplicate detection.
- Tightened `classification_metrics` to require finite binary labels and probabilities in `[0, 1]` without coercion of fractional labels.
- Hardened notebook validation for root/cell/source schema and stale code-only fields; added checks for `nb_epoch` and `nb_val_samples`.
- Added targeted tests for empty classes, contradictory same-split labels, invalid metric inputs, undecodable images, and the static notebook validator. Added Pillow as an explicit dependency for the image contract.
- Updated the README to document these checks and the remaining limits of exact-byte leakage detection.

### Second-review validation

```text
python validate_repository.py
python -m compileall -q malaria_pipeline.py validate_repository.py tests
pytest -q
ruff check .
```

The complete training/evaluation path was not run: this checkout has no authorized image dataset or saved artifacts, and TensorFlow is not installed in the audit environment. No metrics were fabricated or claimed. The local commit created for this review is the source of truth for the changes above; nothing was pushed.

### Remaining risks

Patient-level leakage, semantic near-duplicates, provenance/licensing, label quality, class imbalance, calibration, confidence intervals, external validation, and clinical suitability still require dataset/domain review. TensorFlow/Keras behavior and determinism remain version/hardware dependent. The notebook is an output-free wrapper and is not an interactive production UI or saved-model serving API.
