"""Fast repository checks that do not require data or TensorFlow."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_notebook", ROOT / "scripts" / "validate_notebook.py"
)
assert _VALIDATOR_SPEC and _VALIDATOR_SPEC.loader
_VALIDATOR = importlib.util.module_from_spec(_VALIDATOR_SPEC)
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR)
load_notebook = _VALIDATOR.load_notebook
validate_notebook = _VALIDATOR.validate_notebook


def test_notebook_is_valid_and_clean() -> None:
    assert validate_notebook(load_notebook()) == []


def test_notebook_has_no_embedded_execution_outputs_or_counts() -> None:
    notebook = json.loads((ROOT / "skincancer.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert all(not cell.get("outputs") for cell in code_cells)
    assert all(cell.get("execution_count") is None for cell in code_cells)


def test_expected_project_files_exist() -> None:
    for filename in ("README.md", "requirements.txt", ".gitignore", "ENGINEERING_REVIEW.md"):
        assert (ROOT / filename).is_file(), filename
