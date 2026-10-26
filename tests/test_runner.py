from __future__ import annotations

import json
import sys
from pathlib import Path

from common.contracts import DatasetLayout, ExperimentDefinition, StageDefinition
from common.runner import (
    build_context,
    expand_command,
    next_run_dir,
    resolve_resume_dir,
    run_experiment,
    should_skip_experiment,
    _run_command,
)


def test_next_run_dir_uses_max_numeric_directory(tmp_path: Path) -> None:
    (tmp_path / "01").mkdir()
    (tmp_path / "03").mkdir()

    assert next_run_dir(tmp_path).name == "04"


def test_resolve_resume_dir_accepts_numeric_name(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    (output_root / "03").mkdir(parents=True)

    assert resolve_resume_dir(output_root, "03") == (output_root / "03").resolve()


def test_resume_only_skips_completed_experiment_with_all_artifacts(tmp_path: Path) -> None:
    experiment_output = tmp_path / "03" / "01_3DGS"
    experiment_output.mkdir(parents=True)
    (experiment_output / "status.json").write_text(
        json.dumps({"status": "completed", "artifacts": ["metrics.json"]}),
        encoding="utf-8",
    )
    (experiment_output / "metrics.json").write_text("{}", encoding="utf-8")

    assert should_skip_experiment(experiment_output) is True
    assert should_skip_experiment(tmp_path / "03" / "02_2DGS") is False


def test_incomplete_artifact_prevents_resume_skip(tmp_path: Path) -> None:
    experiment_output = tmp_path / "03" / "01_3DGS"
    experiment_output.mkdir(parents=True)
    (experiment_output / "status.json").write_text(
        json.dumps({"status": "completed", "artifacts": ["metrics.json"]}),
        encoding="utf-8",
    )

    assert should_skip_experiment(experiment_output) is False


def test_empty_artifact_directories_prevent_resume_skip(tmp_path: Path) -> None:
    experiment_output = tmp_path / "03" / "01_3DGS"
    (experiment_output / "train").mkdir(parents=True)
    (experiment_output / "test").mkdir()
    (experiment_output / "status.json").write_text(
        json.dumps({"status": "completed", "artifacts": ["train", "test"]}),
        encoding="utf-8",
    )

    assert should_skip_experiment(experiment_output) is False


def test_expand_command_replaces_absolute_paths() -> None:
    command = expand_command(
        ["python", "train.py", "-s", "{train_data}", "-m", "{output}", "--iterations", "{iterations}"],
        {"train_data": "D:/data/train", "output": "D:/output/03/01_3DGS", "iterations": 30000},
    )

    assert command == [
        "python",
        "train.py",
        "-s",
        "D:/data/train",
        "-m",
        "D:/output/03/01_3DGS",
        "--iterations",
        "30000",
    ]


def test_completed_experiment_stays_resumable_after_second_inspection(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "01_demo"
    experiment_dir.mkdir()
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
    (tmp_path / "repo").mkdir()
    (tmp_path / "data" / "train").mkdir(parents=True)
    (tmp_path / "data" / "test").mkdir(parents=True)
    layout = DatasetLayout(tmp_path / "data", tmp_path / "data" / "train", tmp_path / "data" / "test", tmp_path, tmp_path)
    run_dir = tmp_path / "run"
    context = build_context(tmp_path, definition, run_dir, layout, 1)
    output = run_dir / "01_demo"
    output.mkdir(parents=True)
    (output / "train").mkdir()
    (output / "test").mkdir()
    model = output / "train" / "point_cloud" / "iteration_1" / "point_cloud.ply"
    model.parent.mkdir(parents=True)
    model.write_text("ply\nelement vertex 1\nend_header\n", encoding="utf-8")
    (output / "test" / "test_metrics.json").write_text(
        '{"status":"completed", "num_images":1}', encoding="utf-8"
    )
    (output / "status.json").write_text('{"status":"completed", "artifacts":["train", "test"]}', encoding="utf-8")

    first = run_experiment(definition, context, "both", dry_run=True)

    assert first["status"] == "completed"
    assert should_skip_experiment(output) is True


def test_failed_command_preserves_stderr_tail(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "01_demo"
    experiment_dir.mkdir()
    definition = ExperimentDefinition(
        id="01",
        directory="01_demo",
        name="Demo",
        category="test",
        source_url="local",
        source_repo_dir="repo",
        mask_strategy="metrics_only",
        train=StageDefinition(
            commands=[
                [
                    sys.executable,
                    "-c",
                    "import sys; print('intentional failure', file=sys.stderr); sys.exit(3)",
                ]
            ]
        ),
        test=StageDefinition(commands=[[sys.executable, "-c", "print('unused')"]]),
        artifacts=["train", "test"],
    )
    repo = experiment_dir / "repo"
    repo.mkdir()
    layout = DatasetLayout(tmp_path / "data", tmp_path / "data" / "train", tmp_path / "data" / "test", tmp_path, tmp_path)
    context = build_context(tmp_path, definition, tmp_path / "run", layout, 1)

    result = run_experiment(definition, context, "train")

    assert result["status"] == "failed"
    assert "intentional failure" in result["stderr_tail"]


def test_successful_run_writes_training_render_and_model_telemetry(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "01_demo"
    repo = experiment_dir / "repo"
    repo.mkdir(parents=True)
    train_script = repo / "train.py"
    train_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "out = Path(sys.argv[1])\n"
        "model = out / 'point_cloud' / 'iteration_3' / 'point_cloud.ply'\n"
        "model.parent.mkdir(parents=True, exist_ok=True)\n"
        "model.write_text('ply\\nformat ascii 1.0\\nelement vertex 5\\nend_header\\n')\n",
        encoding="utf-8",
    )
    render_script = repo / "render.py"
    render_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "out = Path(sys.argv[1]) / 'renders'\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / '000.png').write_bytes(b'PNG')\n",
        encoding="utf-8",
    )
    experiment_dir.mkdir(exist_ok=True)
    definition = ExperimentDefinition(
        id="01",
        directory="01_demo",
        name="Demo",
        category="test",
        source_url="local",
        source_repo_dir="repo",
        mask_strategy="metrics_only",
        train=StageDefinition(
            commands=[[sys.executable, str(train_script), "{train_output}", "--iterations", "{iterations}"]]
        ),
        test=StageDefinition(commands=[[sys.executable, str(render_script), "{test_output}"]]),
        artifacts=["train", "test"],
    )
    layout = DatasetLayout(tmp_path / "data", tmp_path / "data" / "train", tmp_path / "data" / "test", tmp_path, tmp_path)
    context = build_context(tmp_path, definition, tmp_path / "run", layout, 3)

    result = run_experiment(definition, context, "both")
    metrics = json.loads((tmp_path / "run" / "01_demo" / "metrics.json").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert metrics["training_time_sec"] is not None
    assert metrics["avg_iteration_time_sec"] is not None
    assert metrics["rendered_frames"] == 1
    assert metrics["render_fps"] > 0
    assert metrics["gaussian_count"] == 5
    assert metrics["model_size_bytes"] > 0


def test_zero_exit_without_artifacts_is_not_completed(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "01_demo"
    repo = experiment_dir / "repo"
    repo.mkdir(parents=True)
    train_script = repo / "train.py"
    train_script.write_text("# exits successfully without writing a model\n", encoding="utf-8")
    definition = ExperimentDefinition(
        id="01",
        directory="01_demo",
        name="Demo",
        category="test",
        source_url="local",
        source_repo_dir="repo",
        mask_strategy="metrics_only",
        train=StageDefinition(
            commands=[[sys.executable, str(train_script), "{train_output}"]]
        ),
        test=StageDefinition(commands=[[sys.executable, "-c", "print('unused')"]]),
        artifacts=["train", "test"],
    )
    layout = DatasetLayout(tmp_path / "data", tmp_path / "data" / "train", tmp_path / "data" / "test", tmp_path, tmp_path)
    context = build_context(tmp_path, definition, tmp_path / "run", layout, 1)

    result = run_experiment(definition, context, "train")

    assert result["status"] == "failed"
    assert result["failed_stage"] == "train"
    assert "artifact" in result["error"]


def test_missing_render_metrics_prevent_successful_test_stage(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "01_demo"
    repo = experiment_dir / "repo"
    repo.mkdir(parents=True)
    render_script = repo / "render.py"
    render_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "out = Path(sys.argv[1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'test_metrics.json').write_text('{\"status\": \"missing_renders\", \"num_images\": 0}')\n",
        encoding="utf-8",
    )
    definition = ExperimentDefinition(
        id="01",
        directory="01_demo",
        name="Demo",
        category="test",
        source_url="local",
        source_repo_dir="repo",
        mask_strategy="metrics_only",
        train=StageDefinition(commands=[[sys.executable, "-c", "print('unused')"]]),
        test=StageDefinition(
            commands=[[sys.executable, str(render_script), "{test_output}"]]
        ),
        artifacts=["train", "test"],
    )
    layout = DatasetLayout(tmp_path / "data", tmp_path / "data" / "train", tmp_path / "data" / "test", tmp_path, tmp_path)
    context = build_context(tmp_path, definition, tmp_path / "run", layout, 1)

    result = run_experiment(definition, context, "test")

    assert result["status"] == "failed"
    assert result["failed_stage"] == "test"
    assert "test" in result["error"]


def test_run_command_streams_child_output_to_console_and_log(tmp_path: Path, capsys) -> None:
    stdout_path = tmp_path / "train_stdout.log"
    stderr_path = tmp_path / "train_stderr.log"
    result = _run_command(
        [
            sys.executable,
            "-c",
            "import sys; print('iteration 7/10', flush=True); print('gpu warning', file=sys.stderr, flush=True)",
        ],
        tmp_path,
        stdout_path,
        stderr_path,
    )

    captured = capsys.readouterr()
    assert result.exit_code == 0
    assert "iteration 7/10" in captured.out
    assert "gpu warning" in captured.err
    assert "iteration 7/10" in stdout_path.read_text(encoding="utf-8")
    assert "gpu warning" in stderr_path.read_text(encoding="utf-8")


def test_run_command_reports_heartbeat_when_child_is_silent(tmp_path: Path, capsys) -> None:
    result = _run_command(
        [sys.executable, "-c", "import time; time.sleep(0.08)"],
        tmp_path,
        tmp_path / "stdout.log",
        tmp_path / "stderr.log",
        label="01 demo / train",
        heartbeat_sec=0.01,
    )

    captured = capsys.readouterr()
    assert result.exit_code == 0
    assert "[01 demo / train] still running" in captured.out
