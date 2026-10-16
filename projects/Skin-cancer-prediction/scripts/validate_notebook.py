"""Static validation for the repository notebook; does not import TensorFlow."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "skincancer.ipynb"
BANNED_PATHS = ("/home/santhu/", "C:\\Users\\", "\\\\Users\\")


def load_notebook() -> dict:
    try:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Notebook is missing: {NOTEBOOK}") from exc
    if notebook.get("nbformat") != 4:
        raise ValueError("Expected a nbformat 4 notebook.")
    if not isinstance(notebook.get("cells"), list) or not notebook["cells"]:
        raise ValueError("Notebook must contain a non-empty cells list.")
    return notebook


def validate_notebook(notebook: dict) -> list[str]:
    errors: list[str] = []
    code_sources: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        code_sources.append(source)
        try:
            ast.parse(source, filename=f"cell-{index}")
        except SyntaxError as exc:
            errors.append(f"cell {index} has invalid Python: {exc}")
        if cell.get("outputs"):
            errors.append(f"cell {index} contains stale execution outputs")
        if cell.get("execution_count") is not None:
            errors.append(f"cell {index} has a stale execution count")
        for banned in BANNED_PATHS:
            if banned in source:
                errors.append(f"cell {index} contains local-only path {banned!r}")

    source = "\n".join(code_sources)
    required_fragments = {
        "train/test validation": "validation_split",
        "held-out evaluation": "model.evaluate(test_generator",
        "deterministic seed": "set_random_seed",
        "missing data guard": "Dataset directory is missing",
        "artifact guard": "Model artifact was not created",
        "supported image validation": "SUPPORTED_IMAGE_EXTENSIONS",
        "minimum data validation": "minimum_train_images",
        "artifact load validation": "load_model(model_path",
    }
    for name, fragment in required_fragments.items():
        if fragment not in source:
            errors.append(f"missing required {name} guard/logic: {fragment}")
    return errors


def main() -> int:
    try:
        errors = validate_notebook(load_notebook())
    except ValueError as exc:
        errors = [str(exc)]
    if errors:
        print("Notebook validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Notebook validation passed: {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
