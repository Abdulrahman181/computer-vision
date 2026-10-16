"""Static repository checks that do not require TensorFlow or the dataset."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REQUIRED_FILES = (
    ".gitignore",
    "README.md",
    "requirements.txt",
    "requirements-dev.txt",
    "segmentation_utils.py",
    "stream_segmentation.py",
    "autonomous-driving-scene-segmentation.ipynb",
    "tests/test_static.py",
)
REQUIRED_RUNTIME_PACKAGES = {"tensorflow", "streamlit", "opencv-python-headless", "numpy", "pillow", "pandas"}


def _python_files(repo: Path) -> list[Path]:
    return sorted(
        path for path in repo.rglob("*.py")
        if ".venv" not in path.parts and "venv" not in path.parts and "__pycache__" not in path.parts
    )


def _validate_requirements(repo: Path) -> list[str]:
    errors: list[str] = []
    requirements = repo / "requirements.txt"
    if not requirements.is_file():
        return errors
    names = {
        line.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].strip().lower()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(REQUIRED_RUNTIME_PACKAGES - names)
    if missing:
        errors.append(f"requirements.txt is missing runtime packages: {', '.join(missing)}")
    return errors


def _validate_notebook(notebook_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"notebook is not valid JSON: {exc}"]
    if notebook.get("nbformat", 0) < 4 or not isinstance(notebook.get("cells"), list):
        errors.append("notebook must use nbformat 4 or newer and contain a cells list")
        return errors
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source, filename=f"notebook cell {index}")
        except SyntaxError as exc:
            errors.append(f"notebook cell {index} syntax error: {exc}")
        if cell.get("execution_count") is not None:
            errors.append(f"notebook cell {index} has stale execution_count")
        if cell.get("outputs"):
            errors.append(f"notebook cell {index} has stale outputs")
        for forbidden in ("/kaggle/", "/home/", "C:\\\\"):
            if forbidden in source:
                errors.append(f"notebook cell {index} contains local-only path {forbidden!r}")
        if "os.remove" in source or "shutil.rmtree" in source:
            errors.append(f"notebook cell {index} deletes dataset/workspace files")
    all_code = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code")
    for required, description in (
        ("tf.image.decode_image", "PNG/JPEG-compatible decoder"),
        ("tf.debugging.assert_equal", "even image-pair width validation"),
        ("tf.debugging.assert_less", "mask palette/tolerance validation"),
    ):
        if required not in all_code:
            errors.append(f"notebook is missing {description}: {required}")
    return errors


def validate_repo(repo: Path) -> list[str]:
    """Return actionable static validation errors for the complete repository."""
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        if not (repo / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for path in _python_files(repo):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(f"{path.relative_to(repo)} Python parse error: {exc}")

    errors.extend(_validate_requirements(repo))
    notebook_path = repo / "autonomous-driving-scene-segmentation.ipynb"
    if notebook_path.is_file():
        errors.extend(_validate_notebook(notebook_path))

    stream_path = repo / "stream_segmentation.py"
    if stream_path.is_file():
        stream_source = stream_path.read_text(encoding="utf-8")
        if "if __name__ == \"__main__\":" not in stream_source:
            errors.append("stream_segmentation.py must expose a guarded main entry point")
        if "prediction_to_probabilities" not in stream_source:
            errors.append("stream_segmentation.py must validate or normalize model predictions")
    return errors


def main() -> int:
    repo = Path(__file__).resolve().parent
    errors = validate_repo(repo)
    if errors:
        print("Project validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Project validation passed (static checks only; no dataset/model execution performed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
