from __future__ import annotations

import json
from pathlib import Path

from weed_detection import coco_to_yolo, split_dataset

REPO_ROOT = Path(__file__).parents[1]


def test_notebook_is_clean_and_all_code_compiles() -> None:
    notebook = json.loads((REPO_ROOT / "weed-detection.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    for cell in notebook["cells"]:
        assert cell.get("outputs", []) == []
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            source = "".join(cell.get("source", []))
            compile(source, "weed-detection.ipynb", "exec")
            assert "/kaggle/" not in source
            assert "!pip install" not in source


def test_coco_conversion_and_grouped_split(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "plant_a.rf.one.jpg").write_bytes(b"not decoded by converter")
    (train / "plant_b.rf.two.jpg").write_bytes(b"not decoded by converter")
    (train / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "categories": [{"id": 7, "name": "weed"}],
                "images": [
                    {"id": 1, "file_name": "plant_a.rf.one.jpg", "width": 100, "height": 100},
                    {"id": 2, "file_name": "plant_b.rf.two.jpg", "width": 100, "height": 100},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 20, 30, 40]},
                    {"id": 2, "image_id": 2, "category_id": 7, "bbox": [0, 0, 100, 100]},
                ],
            }
        ),
        encoding="utf-8",
    )
    converted = tmp_path / "converted"
    report = coco_to_yolo(train / "_annotations.coco.json", train, converted)
    assert report["images"] == 2
    assert (converted / "labels" / "plant_a.rf.one.txt").read_text(encoding="utf-8").startswith("0 ")
    split = split_dataset(converted / "images", tmp_path / "dataset", validation_fraction=0.5, seed=42)
    assert split == {"train": 1, "val": 1}
    assert set((tmp_path / "dataset" / "images" / "train").iterdir()) | set((tmp_path / "dataset" / "images" / "val").iterdir())
