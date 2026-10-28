import json
from pathlib import Path

import pytest

from common.config import discover_experiments
from common.dataset import dataset_layout
from common.runner import build_context, _expanded_stage, run_experiment, should_skip_experiment
from main import parse_args

ROOT = Path(__file__).resolve().parents[1]


def test_default_is_training_without_test_stage():
    assert parse_args([]).mode == "train"


@pytest.mark.parametrize("method_id", ["01", "02", "03", "05", "06", "07", "08", "09", "10"])
def test_train_only_commands_disable_validation_and_keep_saving(tmp_path, method_id):
    definition = next(d for d in discover_experiments(ROOT) if d.id == method_id)
    context = build_context(ROOT, definition, tmp_path / "run", dataset_layout(tmp_path / "data"), 30000)
    commands = _expanded_stage(definition.train, context, mode="train", experiment_id=method_id, stage_name="train")
    command = commands[0]
    original = _expanded_stage(definition.train, context, mode="both", experiment_id=method_id, stage_name="train")[0]
    assert command[command.index("--test_iterations") + 1] == "-1"
    assert command[command.index("--iterations") + 1] == "30000"
    if "--save_iterations" in original:
        assert command[command.index("--save_iterations") + 1] == "30000"
    else:
        assert "--save_iterations" not in command
    if method_id in ("06", "08"):
        assert "--train_only" in command
    if "--test_iterations" in original:
        assert original[original.index("--test_iterations") + 1] == "30000"


def failed_test_result(tmp_path):
    definition = next(d for d in discover_experiments(ROOT) if d.id == "02")
    context = build_context(ROOT, definition, tmp_path / "run", dataset_layout(tmp_path / "data"), 30000)
    output = context["output"]
    model = output / "train/point_cloud/iteration_30000/point_cloud.ply"
    model.parent.mkdir(parents=True)
    model.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n")
    previous = {"status": "failed", "mode": "both", "failed_stage": "test",
                "error": "Missing expected artifact(s): test", "artifacts": ["train", "test"]}
    (output / "status.json").write_text(json.dumps(previous))
    return definition, context, output, previous


def test_training_resume_skips_model_with_later_test_failure(tmp_path):
    _, _, output, _ = failed_test_result(tmp_path)
    assert should_skip_experiment(output, mode="train")
    assert not should_skip_experiment(output, mode="both")


def test_training_resume_retains_test_failure_evidence_without_launch(tmp_path, monkeypatch):
    definition, context, output, previous = failed_test_result(tmp_path)
    def unexpected(*args, **kwargs):
        raise AssertionError("Completed training must not be relaunched")
    monkeypatch.setattr("common.runner._run_command", unexpected)
    result = run_experiment(definition, context, "train")
    assert result["status"] == "completed" and result["artifacts"] == ["train"]
    assert result["train_reused"]
    assert json.loads((output / result["previous_status_file"]).read_text()) == previous
    assert not should_skip_experiment(output, mode="test")


def test_training_failure_with_partial_saved_model_must_not_be_skipped(tmp_path):
    _, _, output, previous = failed_test_result(tmp_path)
    previous["failed_stage"] = "train"
    (output / "status.json").write_text(json.dumps(previous))
    assert not should_skip_experiment(output, mode="train")


def test_octree_train_only_does_not_reenable_periodic_validation():
    import ast
    from types import SimpleNamespace
    source = ROOT / "08_Octree_GS/runtime/train.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    entry = next(node for node in tree.body if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))
    blocks = []
    for node in entry.body:
        if isinstance(node, ast.If) and any(
            isinstance(child, ast.Assign) and any(isinstance(target, ast.Attribute)
                and target.attr == "test_iterations" for target in child.targets)
                for child in ast.walk(node)):
            blocks.append(node)
    args = SimpleNamespace(train_only=True, test_iterations=[-1], iterations=30000)
    exec(compile(ast.Module(body=blocks, type_ignores=[]), str(source), "exec"), {"args": args})
    assert args.test_iterations == []
