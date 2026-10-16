"""Static tests intentionally avoid importing TensorFlow or Streamlit."""
import ast
import unittest
from pathlib import Path

from validate_project import validate_repo

REPO = Path(__file__).resolve().parents[1]


class StaticProjectTests(unittest.TestCase):
    def test_project_validator_passes(self) -> None:
        self.assertEqual(validate_repo(REPO), [])

    def test_python_sources_parse(self) -> None:
        for path in (REPO / "stream_segmentation.py", REPO / "validate_project.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


if __name__ == "__main__":
    unittest.main()
