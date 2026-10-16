# Engineering review — weed-detection

## Scope

Reviewed every tracked file in the repository. The original repository contained one notebook, `weed-detection.ipynb`, with 15 executed code cells and large embedded outputs. No dataset, requirements file, README, tests, or configuration file was present.

## Findings and remediation

| Area | Finding | Remediation |
|---|---|---|
| Portability | Hard-coded `/kaggle/input` and `/kaggle/working` paths made the notebook non-portable and caused immediate failure outside Kaggle. | Replaced paths with `WEED_DATA_ROOT` and `WEED_ARTIFACT_ROOT` environment variables, with portable defaults. |
| Missing-data behavior | The notebook assumed a particular Kaggle dataset, annotation file, test filename, and prior training run. | Added explicit directory/file checks and actionable `FileNotFoundError` messages. Data and weights are not fabricated. |
| Dependency management | `!pip install ultralytics` was embedded in the notebook; no dependency manifest existed. | Added `requirements.txt`; training imports are guarded and training is opt-in. |
| Leakage | Offline augmentation was performed on every image before a random split, so an original and its transformed copy could land in different splits. | The replacement converts and splits first, groups `.rf.` derivatives, and leaves augmentation to the training process. |
| Reproducibility | The split was only partly deterministic, image ordering was filesystem-dependent, and no seed was passed to training. | Sorted inputs, explicit seed (`WEED_SEED`, default 42), deterministic grouped split, and seed/deterministic training arguments. |
| COCO conversion | Annotation lookup was O(images × annotations), broad exceptions hid malformed data, boxes were not validated, and file names were not checked for traversal. | Indexed images, validates dimensions/categories/boxes, counts skipped annotations, and rejects path traversal. |
| Evaluation validity | The notebook printed stale embedded metrics and validated only the generated validation set; no independent labelled test set was supplied. | Removed outputs and stale results. The notebook reports validation metrics only when a real model/artifact exists and explicitly does not claim an independent test score. |
| Safety/cleanliness | The original first cell recursively deleted the entire working directory and the notebook embedded model output/images. | No destructive global cleanup; generated artifacts are scoped to the configured artifact directory and notebook outputs are empty. |
| Maintainability | No README, ignore rules, tests, or reusable code existed. | Added README, `.gitignore`, `weed_detection.py`, and static/functional tests. |

## Validation performed

- Parsed the notebook as JSON and compiled every code cell with Python AST/`compile`.
- Confirmed notebook execution counts are cleared and all outputs are removed.
- Ran `pytest -q` and `ruff check .` after the changes.
- Ran `python -m compileall -q weed_detection.py tests`.

The static/functional checks do not execute YOLO training, GPU inference, network weight downloads, or a real dataset run.

## Remaining limitations

- The actual dataset is not part of this repository, so label quality, corrupt image rates, class balance, near-duplicate similarity, and end-to-end training cannot be verified here.
- The repository has no labelled test split. Validation metrics are not a substitute for an independent test benchmark.
- `ultralytics` training remains compute- and dependency-heavy and is intentionally disabled by default.
- Grouping by filename stem reduces common duplicate leakage but cannot detect visually similar images with unrelated names. A future data-quality pipeline should add perceptual-hash or source/sequence grouping if provenance is available.
- No API service was present; therefore API authentication, request limits, and schema checks are not applicable.

No performance result was generated or endorsed by this review.

## Second Senior Review (2026-10-16)

A second manual pass rechecked the COCO-to-YOLO conversion contract, split-before-augmentation logic, artifact guards, notebook state, and tests. No additional correctness defect was found in the repaired workflow. Dataset and trained weights remain required for a real run.

Validation: `pytest -q` (2 passed), `compileall`, notebook JSON validation, and `git diff --check` passed.
