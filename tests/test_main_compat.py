from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import discover_experiments
from common.contracts import DatasetReport
from main import write_run_config


ROOT = Path(__file__).resolve().parents[1]


def test_run_config_records_dataset_compatibility_mode(tmp_path: Path) -> None:
    args = argparse.Namespace(mode="both", iterations=1)
    report = DatasetReport(ok=True, errors=[], warnings=[], counts={})

    write_run_config(
        tmp_path,
        args,
        tmp_path / "data",
        report,
        discover_experiments(ROOT),
    )

    payload = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    modes = {item["id"]: item["dataset_compat"] for item in payload["experiments"]}
    assert modes["01"] == "raw"
    assert all(modes[f"{index:02d}"] == "standard_colmap" for index in range(2, 10))
    assert modes["10"] == "raw"
