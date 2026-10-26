from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.contracts import DatasetReport
from main import write_run_config


def test_resume_config_does_not_overwrite_original_run_config(tmp_path: Path) -> None:
    original_path = tmp_path / "run_config.json"
    original_path.write_text('{"original": true}', encoding="utf-8")
    args = argparse.Namespace(mode="both", iterations=30000)
    report = DatasetReport(ok=True, errors=[], warnings=[], counts={"train_images": 1})

    write_run_config(
        tmp_path,
        args,
        tmp_path / "data",
        report,
        [],
        preserve_existing=True,
    )

    assert json.loads(original_path.read_text(encoding="utf-8")) == {"original": True}
    assert (tmp_path / "resume_config.json").is_file()
