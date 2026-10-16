from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "egyptian-hieroglyphs-using-tl-cnn.ipynb"
VALIDATOR = ROOT / "validate_project.py"


def test_validator_passes_and_notebook_is_clean() -> None:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR)], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None


def test_notebook_has_dynamic_class_and_data_contracts() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for marker in (
        "HIEROGLYPHS_DATA_DIR",
        "class_to_id",
        "unknown_classes",
        "sparse_categorical_crossentropy",
        "Path(",
    ):
        assert marker in source
