from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


SUMMARY_FIELDS = [
    "id",
    "name",
    "directory",
    "status",
    "mask_strategy",
    "source_url",
    "exit_code",
    "failed_stage",
    "PSNR",
    "SSIM",
    "LPIPS",
    "training_time_sec",
    "avg_iteration_time_sec",
    "render_time_sec",
    "rendered_frames",
    "render_fps",
    "peak_gpu_memory_mb",
    "peak_gpu_memory_reason",
    "gaussian_count",
    "model_file",
    "model_size_bytes",
    "model_size_mb",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _metrics(output_dir: Path) -> dict[str, Any]:
    candidates = [
        output_dir / "metrics.json",
        output_dir / "final_metrics.json",
        output_dir / "test_metrics.json",
        output_dir / "test" / "test_metrics.json",
        output_dir / "test" / "metrics.json",
    ]
    for path in candidates:
        data = _read_json(path)
        if data:
            return data
    return {}


def build_summary_rows(run_dir: Path, definitions: Iterable[Any]) -> list[dict[str, Any]]:
    rows = []
    for definition in definitions:
        output_dir = Path(run_dir) / definition.directory
        status = _read_json(output_dir / "status.json")
        metrics = _metrics(output_dir)
        row = {
            "id": definition.id,
            "name": definition.name,
            "directory": definition.directory,
            "status": status.get("status", "pending"),
            "mask_strategy": definition.mask_strategy,
            "source_url": definition.source_url,
            "exit_code": status.get("exit_code", ""),
            "failed_stage": status.get("failed_stage", ""),
            "PSNR": metrics.get("PSNR", metrics.get("psnr", "")),
            "SSIM": metrics.get("SSIM", metrics.get("ssim", "")),
            "LPIPS": metrics.get("LPIPS", metrics.get("lpips", "")),
            "training_time_sec": metrics.get("training_time_sec", ""),
            "avg_iteration_time_sec": metrics.get("avg_iteration_time_sec", ""),
            "render_time_sec": metrics.get("render_time_sec", ""),
            "rendered_frames": metrics.get("rendered_frames", ""),
            "render_fps": metrics.get("render_fps", ""),
            "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb", ""),
            "peak_gpu_memory_reason": metrics.get("peak_gpu_memory_reason", ""),
            "gaussian_count": metrics.get("gaussian_count", ""),
            "model_file": metrics.get("model_file", ""),
            "model_size_bytes": metrics.get("model_size_bytes", ""),
            "model_size_mb": metrics.get("model_size_mb", ""),
        }
        rows.append(row)
    return rows


def write_summary(run_dir: Path, definitions: Iterable[Any]) -> list[dict[str, Any]]:
    rows = build_summary_rows(run_dir, definitions)
    with (Path(run_dir) / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (Path(run_dir) / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows
