"""Static checks for the hieroglyphs-recognition repository.

This script intentionally does not import TensorFlow or require the dataset. It
is suitable for CI and for quick validation in a lightweight environment.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "egyptian-hieroglyphs-using-tl-cnn.ipynb"
REQUIRED_FILES = (NOTEBOOK, ROOT / "README.md", ROOT / "requirements.txt")
REQUIRED_MARKERS = (
    "HIEROGLYPHS_DATA_DIR",
    "Path(",
    "class_to_id",
    "unknown_classes",
    "sparse_categorical_crossentropy",
)


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")

    if NOTEBOOK.is_file():
        try:
            notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid notebook JSON: {exc}")
        else:
            sources: list[str] = []
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = "".join(cell.get("source", []))
                sources.append(source)
                try:
                    ast.parse(source, filename=f"{NOTEBOOK.name}:cell-{index}")
                except SyntaxError as exc:
                    errors.append(f"syntax error in notebook cell {index}: {exc}")
                if cell.get("outputs"):
                    errors.append(f"notebook cell {index} contains stale outputs")

            combined = "\n\n".join(sources)
            for marker in REQUIRED_MARKERS:
                if marker not in combined:
                    errors.append(f"notebook is missing required safeguard/configuration: {marker}")
            if re.search(r"(?:/home/|[A-Za-z]:\\\\)", combined):
                errors.append("notebook contains a hard-coded absolute filesystem path")
            if re.search(r"Dense\(95", combined):
                errors.append("notebook contains a hard-coded 95-class output")

    ruff = shutil.which("ruff")
    if ruff:
        result = subprocess.run(
            [ruff, "check", "--isolated", "--select", "E9,F63,F7,F821", str(NOTEBOOK), str(Path(__file__))],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            errors.append("ruff reported errors:\n" + (result.stdout + result.stderr).strip())
    else:
        print("ruff not installed; AST checks still ran")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Static validation passed: required files, notebook syntax, outputs, portability, and safeguards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
