from __future__ import annotations

import os
from pathlib import Path

from common.contracts import ExperimentDefinition, StageDefinition
from common.summary import SUMMARY_FIELDS, build_summary_rows
from common.telemetry import average_iteration_time, collect_model_artifacts, sample_gpu_memory_mb


def test_collect_model_artifacts_reads_ply_vertex_count_and_size(tmp_path: Path) -> None:
    model_path = tmp_path / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    model_path.parent.mkdir(parents=True)
    model_path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 37\nproperty float x\nend_header\n",
        encoding="ascii",
    )

    result = collect_model_artifacts(tmp_path)

    assert result["model_file"] == str(model_path)
    assert result["gaussian_count"] == 37
    assert result["model_size_bytes"] == model_path.stat().st_size
    assert result["model_size_mb"] > 0


def test_reduced_3dgs_counts_degree_groups_without_counting_codebook(tmp_path):
    path = tmp_path / "point_cloud.ply"
    path.write_text("ply\nformat binary_little_endian 1.0\n"
                    "element vertex_0 2\nproperty float x\n"
                    "element vertex_1 0\nproperty float x\n"
                    "element vertex_2 3\nproperty float x\n"
                    "element vertex_3 0\nproperty float x\n"
                    "element codebook_centers 256\nproperty float opacity\nend_header\n")
    assert collect_model_artifacts(tmp_path)["gaussian_count"] == 5


def test_average_iteration_time_returns_none_without_a_valid_iteration_count() -> None:
    assert average_iteration_time(12.0, 3) == 4.0
    assert average_iteration_time(12.0, 0) is None
    assert average_iteration_time(None, 3) is None


def test_gpu_sampling_reports_unavailable_tool_without_fabricating_zero() -> None:
    sample = sample_gpu_memory_mb(os.getpid(), executable="missing-nvidia-smi-for-test")

    assert sample.value_mb is None
    assert sample.reason == "nvidia-smi not found"


def test_summary_exposes_efficiency_and_resource_fields(tmp_path: Path) -> None:
    definition = ExperimentDefinition(
        id="01",
        directory="01_demo",
        name="Demo",
        category="test",
        source_url="local",
        source_repo_dir="repo",
        mask_strategy="metrics_only",
        train=StageDefinition(commands=[["python", "train.py"]]),
        test=StageDefinition(commands=[["python", "render.py"]]),
        artifacts=["train", "test"],
    )
    output = tmp_path / "01_demo"
    output.mkdir()
    (output / "status.json").write_text('{"status":"completed"}', encoding="utf-8")
    (output / "metrics.json").write_text(
        '{"training_time_sec": 12.0, "avg_iteration_time_sec": 0.004, '
        '"render_fps": 8.0, "peak_gpu_memory_mb": 4096.0, '
        '"gaussian_count": 123, "model_size_mb": 7.5}',
        encoding="utf-8",
    )

    row = build_summary_rows(tmp_path, [definition])[0]

    required = {
        "avg_iteration_time_sec",
        "peak_gpu_memory_mb",
        "training_time_sec",
        "render_fps",
        "gaussian_count",
        "model_size_mb",
    }
    assert required.issubset(SUMMARY_FIELDS)
    assert all(row[field] != "" for field in required)
