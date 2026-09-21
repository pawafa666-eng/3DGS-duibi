from __future__ import annotations

import sys
from pathlib import Path

from smoke_train import build_main_command, detect_training_started, run_smoke_command


def test_detect_training_started_requires_training_progress():
    assert not detect_training_started("Loading Training Cameras")
    assert detect_training_started("Training progress: 0%|          | 0/30000")
    assert detect_training_started("[ITER 1] Evaluating train")


def test_build_main_command_runs_one_selected_algorithm():
    command = build_main_command(Path("output/smoke"), "03")
    assert command[:4] == [sys.executable, "-u", "main.py", "--mode"]
    assert "--only" in command
    assert command[command.index("--only") + 1] == "03"
    assert command[command.index("--output") + 1] == str(Path("output/smoke"))


def test_run_smoke_command_stops_after_training_starts(tmp_path):
    fixture = tmp_path / "fake_train.py"
    fixture.write_text(
        "import time\n"
        "print('import ok', flush=True)\n"
        "print('Training progress: 0%| 0/30000', flush=True)\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    result = run_smoke_command(
        [sys.executable, "-u", str(fixture)],
        cwd=tmp_path,
        timeout=5,
        terminate_wait=2,
    )
    assert result["started"] is True
    assert result["terminated"] is True
    assert result["duration_sec"] < 5
