from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import pytest

from common.contracts import DatasetLayout, ExperimentDefinition, StageDefinition
from common.dataset_compat import (
    DatasetCompatibilityError,
    prepare_standard_colmap_dataset,
)
from common.runner import build_context, run_experiment


CUSTOM_POINT_FORMAT = "<QdddBBBddddQ"
STANDARD_POINT_HEADER_FORMAT = "<QdddBBBd"


def _write_custom_points(path: Path, rows: list[tuple]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(rows)))
        for row in rows:
            handle.write(struct.pack(CUSTOM_POINT_FORMAT, *row))


def _make_dataset(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    dataset_root = tmp_path / "数据" / "地大隧道"
    source_bytes: dict[str, bytes] = {}
    rows = [
        (7, 1.25, -2.5, 3.75, 12, 34, 56, 0.0, 0.0, 1.0, 0.125, 0),
        (8, -4.0, 5.5, 6.25, 200, 150, 100, 1.0, 0.0, 0.0, 0.25, 0),
    ]
    for split in ("train", "test"):
        split_root = dataset_root / split
        (split_root / "images").mkdir(parents=True)
        (split_root / "masks").mkdir()
        sparse_root = split_root / "sparse" / "0"
        sparse_root.mkdir(parents=True)
        (split_root / "images" / "frame.png").write_bytes(b"image-" + split.encode())
        (split_root / "masks" / "frame.mask.png").write_bytes(b"mask-" + split.encode())
        for name, content in {
            "cameras.bin": b"camera-" + split.encode(),
            "images.bin": b"image-index-" + split.encode(),
        }.items():
            path = sparse_root / name
            path.write_bytes(content)
            source_bytes[str(path)] = content
        points_path = sparse_root / "points3D.bin"
        _write_custom_points(points_path, rows)
        source_bytes[str(points_path)] = points_path.read_bytes()
    return dataset_root, source_bytes


def _read_standard_points(path: Path) -> list[tuple]:
    points: list[tuple] = []
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            point = struct.unpack(STANDARD_POINT_HEADER_FORMAT, handle.read(43))
            track_length = struct.unpack("<Q", handle.read(8))[0]
            handle.read(8 * track_length)
            points.append((*point, track_length))
        assert handle.read() == b""
    return points


def test_standard_copy_converts_both_splits_and_preserves_all_inputs(tmp_path: Path) -> None:
    dataset_root, source_bytes = _make_dataset(tmp_path)
    original_hashes = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in source_bytes
    }

    result = prepare_standard_colmap_dataset(dataset_root, tmp_path / "缓存")

    assert result.cache_hit is False
    assert result.root != dataset_root
    for split in ("train", "test"):
        split_root = result.root / split
        assert (split_root / "images").is_dir()
        assert (split_root / "masks").is_dir()
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            assert (split_root / "sparse" / "0" / name).is_file()
        assert (split_root / "sparse" / "0" / "cameras.bin").read_bytes() == (
            b"camera-" + split.encode()
        )
        assert (split_root / "sparse" / "0" / "images.bin").read_bytes() == (
            b"image-index-" + split.encode()
        )
        points = _read_standard_points(split_root / "sparse" / "0" / "points3D.bin")
        assert points == [
            (7, 1.25, -2.5, 3.75, 12, 34, 56, 0.125, 0),
            (8, -4.0, 5.5, 6.25, 200, 150, 100, 0.25, 0),
        ]

    for path, digest in original_hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_standard_copy_reuses_valid_cache_without_rewriting_converted_points(tmp_path: Path) -> None:
    dataset_root, _ = _make_dataset(tmp_path)

    first = prepare_standard_colmap_dataset(dataset_root, tmp_path / "缓存")
    converted_path = first.root / "train" / "sparse" / "0" / "points3D.bin"
    first_mtime = converted_path.stat().st_mtime_ns

    second = prepare_standard_colmap_dataset(dataset_root, tmp_path / "缓存")

    assert second.cache_hit is True
    assert second.root == first.root
    assert converted_path.stat().st_mtime_ns == first_mtime


def test_fallback_copy_is_reused_as_a_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset_root, _ = _make_dataset(tmp_path)

    def fallback_copy(source: Path, target: Path, split: str, name: str) -> str:
        shutil.copytree(source, target)
        return "copy"

    monkeypatch.setattr("common.dataset_compat._link_directory", fallback_copy)

    first = prepare_standard_colmap_dataset(dataset_root, tmp_path / "缓存")
    second = prepare_standard_colmap_dataset(dataset_root, tmp_path / "缓存")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.root == first.root


def test_invalid_custom_points_error_identifies_split_and_source(tmp_path: Path) -> None:
    dataset_root, _ = _make_dataset(tmp_path)
    points_path = dataset_root / "train" / "sparse" / "0" / "points3D.bin"
    points_path.write_bytes(struct.pack("<Q", 1) + b"truncated")

    with pytest.raises(DatasetCompatibilityError) as raised:
        prepare_standard_colmap_dataset(dataset_root, tmp_path / "缓存")

    message = str(raised.value)
    assert "train" in message
    assert str(points_path) in message
    assert "standard COLMAP" in message


def test_runner_routes_standard_methods_to_cache_and_raw_methods_to_source(tmp_path: Path) -> None:
    dataset_root, _ = _make_dataset(tmp_path)
    layout = DatasetLayout(
        dataset_root,
        dataset_root / "train",
        dataset_root / "test",
        dataset_root / "train" / "masks",
        dataset_root / "test" / "masks",
    )

    standard_dir = tmp_path / "02_demo"
    standard_dir.mkdir()
    (standard_dir / "experiment.json").write_text(
        json.dumps({"dataset_compat": "standard_colmap"}), encoding="utf-8"
    )
    standard_definition = ExperimentDefinition(
        id="02",
        directory="02_demo",
        name="Standard",
        category="test",
        source_url="local",
        source_repo_dir="repo",
        mask_strategy="native",
        train=StageDefinition(commands=[["python", "train.py", "{train_data}"]]),
        test=StageDefinition(commands=[["python", "render.py", "{test_data}"]]),
        artifacts=["train", "test"],
    )
    standard_context = build_context(tmp_path, standard_definition, tmp_path / "output" / "01", layout, 1)
    standard_result = run_experiment(standard_definition, standard_context, "train", dry_run=True)
    standard_command = (tmp_path / "output" / "01" / "02_demo" / "command.txt").read_text(encoding="utf-8")

    raw_dir = tmp_path / "01_demo"
    raw_dir.mkdir()
    (raw_dir / "experiment.json").write_text(
        json.dumps({"dataset_compat": "raw"}), encoding="utf-8"
    )
    raw_definition = ExperimentDefinition(
        id="01",
        directory="01_demo",
        name="Raw",
        category="test",
        source_url="local",
        source_repo_dir="runtime",
        mask_strategy="metrics_only",
        train=StageDefinition(commands=[["python", "train.py", "{train_data}"]]),
        test=StageDefinition(commands=[["python", "render.py", "{test_data}"]]),
        artifacts=["train", "test"],
    )
    raw_context = build_context(tmp_path, raw_definition, tmp_path / "output" / "02", layout, 1)
    raw_result = run_experiment(raw_definition, raw_context, "train", dry_run=True)
    raw_command = (tmp_path / "output" / "02" / "01_demo" / "command.txt").read_text(encoding="utf-8")

    assert standard_result["status"] == "planned"
    assert str(dataset_root / "train") not in standard_command
    assert ".dataset_compat" in standard_command
    assert raw_result["status"] == "planned"
    assert str(dataset_root / "train") in raw_command
    assert ".dataset_compat" not in raw_command


def test_dry_run_plans_standard_cache_without_creating_point_copy(tmp_path: Path) -> None:
    dataset_root, _ = _make_dataset(tmp_path)
    layout = DatasetLayout(
        dataset_root,
        dataset_root / "train",
        dataset_root / "test",
        dataset_root / "train" / "masks",
        dataset_root / "test" / "masks",
    )
    experiment_dir = tmp_path / "02_demo"
    experiment_dir.mkdir()
    (experiment_dir / "experiment.json").write_text(
        json.dumps({"dataset_compat": "standard_colmap"}), encoding="utf-8"
    )
    definition = ExperimentDefinition(
        id="02",
        directory="02_demo",
        name="Standard",
        category="test",
        source_url="local",
        source_repo_dir="repo",
        mask_strategy="native",
        train=StageDefinition(commands=[["python", "train.py", "{train_data}"]]),
        test=StageDefinition(commands=[["python", "render.py", "{test_data}"]]),
        artifacts=["train", "test"],
    )
    run_dir = tmp_path / "output" / "01"
    context = build_context(tmp_path, definition, run_dir, layout, 1)

    result = run_experiment(definition, context, "train", dry_run=True)
    status = json.loads((run_dir / "02_demo" / "status.json").read_text(encoding="utf-8"))
    planned_cache = Path(status["dataset_compat_cache_dir"])

    assert result["status"] == "planned"
    assert status["dataset_compat_cache_hit"] is False
    assert isinstance(status["dataset_compat_cache_dir"], str)
    assert planned_cache == run_dir.parent / ".dataset_compat" / planned_cache.name
    assert not planned_cache.exists()
    assert not (planned_cache / "train" / "sparse" / "0" / "points3D.bin").exists()
