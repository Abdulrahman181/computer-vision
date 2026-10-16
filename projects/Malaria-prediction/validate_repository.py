"""Static checks that do not require TensorFlow or the image dataset."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "malaria-pred.ipynb"


def validate_notebook(path: Path = NOTEBOOK) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read notebook: {exc}"]
    if not isinstance(notebook, dict):
        return ["notebook root must be an object"]
    if notebook.get("nbformat") != 4:
        errors.append("notebook must use nbformat 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return errors + ["notebook cells must be a list"]
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cell {index} must be an object")
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            errors.append(f"cell {index} has an invalid cell_type")
        source = cell.get("source", [])
        valid_source = isinstance(source, str) or (
            isinstance(source, list) and all(isinstance(line, str) for line in source)
        )
        if not valid_source:
            errors.append(f"cell {index} has invalid source")
            source_text = ""
        else:
            source_text = source if isinstance(source, str) else "".join(source)
        if cell_type != "code" and ("execution_count" in cell or "outputs" in cell):
            errors.append(f"cell {index} has code-only execution fields")
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None:
                errors.append(f"cell {index} has a stale execution count")
            if cell.get("outputs") != []:
                errors.append(f"cell {index} has stale outputs")
            try:
                ast.parse(source_text)
            except SyntaxError as exc:
                errors.append(f"cell {index} has invalid Python: {exc}")
            forbidden_tokens = (
                "/kaggle/",
                "fit_generator",
                "evaluate_generator",
                "nb_epoch",
                "nb_val_samples",
            )
            for forbidden in forbidden_tokens:
                if forbidden in source_text:
                    errors.append(f"cell {index} contains forbidden legacy text: {forbidden}")
    return errors


def main() -> int:
    errors = validate_notebook()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Repository static validation passed: notebook JSON, outputs, syntax, and "
        "legacy paths checked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
