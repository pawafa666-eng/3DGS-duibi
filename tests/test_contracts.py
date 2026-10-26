from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.config import load_experiment_definition
from common.dataset import validate_dataset_layout


def test_load_experiment_definition_requires_public_fields(tmp_path: Path) -> None:
    config_dir = tmp_path / "01_demo"
    config_dir.mkdir()
    (config_dir / "experiment.json").write_text(
        json.dumps(
            {
                "id": "01",
                "name": "Demo",
                "category": "test",
                "source": {"url": "https://example.com/repo.git"},
                "mask_strategy": "metrics_only",
                "train": {"command": ["python", "train.py"]},
                "test": {"command": ["python", "render.py"]},
                "artifacts": ["metrics.json"],
            }
        ),
        encoding="utf-8",
    )

    definition = load_experiment_definition(config_dir)

    assert definition.id == "01"
    assert definition.name == "Demo"
    assert definition.train_command == ["python", "train.py"]


def test_validate_dataset_layout_reports_missing_split_files(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "train" / "images").mkdir(parents=True)
    (dataset / "train" / "masks").mkdir()
    (dataset / "train" / "sparse" / "0").mkdir(parents=True)
    (dataset / "test" / "images").mkdir(parents=True)
    (dataset / "test" / "masks").mkdir()
    (dataset / "test" / "sparse" / "0").mkdir(parents=True)

    report = validate_dataset_layout(dataset)

    assert report.ok is False
    assert any("cameras.bin" in error for error in report.errors)
