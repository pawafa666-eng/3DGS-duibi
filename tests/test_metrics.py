from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from common.metrics import evaluate


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), (value, value, value)).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (2, 2), 255).save(path)


def _evaluate_fixture(tmp_path: Path, render_root: Path) -> dict:
    data_root = tmp_path / "data"
    masks_root = tmp_path / "masks"
    output_path = tmp_path / "metrics.json"
    return evaluate(render_root, data_root, masks_root, output_path)


def test_numeric_renders_map_to_stably_sorted_images_and_record_target_stems(tmp_path: Path) -> None:
    data_images = tmp_path / "data" / "images"
    _write_rgb(data_images / "imageb_002.jpg", 80)
    _write_rgb(data_images / "imageb_001.jpg", 40)
    _write_mask(tmp_path / "masks" / "imageb_001.mask.png")
    _write_mask(tmp_path / "masks" / "imageb_002.mask.png")

    render_root = tmp_path / "algorithm" / "deep" / "test" / "renders"
    _write_rgb(render_root / "00001.png", 81)
    _write_rgb(render_root / "00000.png", 41)

    result = _evaluate_fixture(tmp_path, tmp_path / "algorithm")

    assert result["status"] == "completed"
    assert [item["image"] for item in result["per_image"]] == ["imageb_001", "imageb_002"]


def test_exact_stems_take_precedence_and_numeric_fallback_uses_remaining_images(tmp_path: Path) -> None:
    data_images = tmp_path / "data" / "images"
    _write_rgb(data_images / "imageb_002.jpg", 80)
    _write_rgb(data_images / "imageb_001.jpg", 40)
    _write_mask(tmp_path / "masks" / "imageb_001.mask.png")
    _write_mask(tmp_path / "masks" / "imageb_002.mask.png")

    render_root = tmp_path / "algorithm" / "renders"
    _write_rgb(render_root / "imageb_001.png", 41)
    _write_rgb(render_root / "00000.png", 81)

    result = _evaluate_fixture(tmp_path, tmp_path / "algorithm")

    assert result["status"] == "completed"
    assert [item["image"] for item in result["per_image"]] == ["imageb_002", "imageb_001"]


def test_nested_train_renders_are_not_treated_as_test_renders(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "data" / "images" / "imageb_001.jpg", 40)
    _write_mask(tmp_path / "masks" / "imageb_001.mask.png")

    render_root = tmp_path / "algorithm" / "test" / "nested" / "renders"
    _write_rgb(render_root / "00000.png", 41)
    _write_rgb(tmp_path / "algorithm" / "train" / "renders" / "00001.png", 41)

    result = _evaluate_fixture(tmp_path, tmp_path / "algorithm")
    written = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert result["num_images"] == 1
    assert written["per_image"][0]["image"] == "imageb_001"
