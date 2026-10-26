"""Safety tests use tiny import fixtures, never a GPU or a training substitute.

Fixtures test the blocked adapter's process/logging boundary only. Successful
fixture imports are deliberately NOT evidence of SuGaR training or saving.
"""
from __future__ import annotations

import hashlib
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from common import sugar_smoke


ROOT = Path(__file__).resolve().parents[1]
REAL_REPO = ROOT / "04_SuGaR" / "runtime"


def fixture_repo(tmp_path, *, failing=False, slow=False):
    repo = tmp_path / "repo with spaces 中文"
    repo.mkdir()
    for module, symbol in sugar_smoke.IMPORT_TARGETS:
        path = repo.joinpath(*module.split(".")).with_suffix(".py")
        path.parent.mkdir(exist_ok=True)
        prefix = ""
        if failing:
            prefix = "raise ImportError('SUGAR_DIAGNOSTIC_SENTINEL')\n"
        elif slow:
            prefix = "import time\nprint('before timeout', flush=True)\ntime.sleep(30)\n"
        path.write_text(
            prefix + f"def {symbol}(*args, **kwargs):\n"
            "    raise AssertionError('TRAINING MUST NEVER BE CALLED')\n",
            encoding="utf-8",
        )
    # Import as a module must never execute a guarded training entry point.
    (repo / "train.py").write_text(
        "if __name__ == '__main__':\n"
        "    raise AssertionError('ENTRY POINT EXECUTED')\n", encoding="utf-8"
    )
    return repo


def run_fixture(tmp_path, *, failing=False):
    repo = fixture_repo(tmp_path, failing=failing)
    data = tmp_path / "synthetic data"
    data.mkdir()
    output = tmp_path / "diagnostics"
    report = sugar_smoke.run_smoke(repo, data, output, iterations=10, import_timeout=5)
    return repo, output, report


def test_successful_imports_never_mean_training_success(tmp_path):
    _, output, report = run_fixture(tmp_path)
    assert all(p["status"] == "imported" for p in report["imports"])
    assert report["status"] == "blocked"
    assert report["success"] is False
    assert report["training_started"] is False
    assert report["sugar_checkpoint"] is None
    assert report["artifacts"] == []
    for stage in ("vanilla_initialization", "sugar_coarse", "mesh", "sugar_refine"):
        expected = "not_verified" if stage in ("mesh", "sugar_refine") else "blocked"
        assert report["stages"][stage]["status"] == expected
        assert report["stages"][stage]["executed_iterations"] == 0
    assert report["stages"]["sugar_coarse"]["requested_iterations"] == 10
    assert report["stages"]["mesh"]["requested_iterations"] is None
    assert json.loads((output / "report.json").read_text("utf-8")) == report
    assert not list(output.rglob("*.pt"))
    assert not list(output.rglob("*.ply"))


def test_real_import_failure_and_traceback_are_saved(tmp_path):
    _, output, report = run_fixture(tmp_path, failing=True)
    probe = report["imports"][0]
    assert probe["module"] == "sugar_scene.sugar_model"
    assert probe["status"] == "failed"
    assert probe["returncode"] != 0
    assert "Traceback" in Path(probe["stderr_path"]).read_text("utf-8")
    assert "SUGAR_DIAGNOSTIC_SENTINEL" in (output / "stderr.log").read_text("utf-8")
    assert "environment_import_failed" in {b["code"] for b in report["blockers"]}


@pytest.mark.parametrize("iterations", [0, -1, 11, 7000, True])
def test_budget_rejected_before_import_or_output(tmp_path, iterations):
    with pytest.raises(ValueError, match="exactly 10"):
        sugar_smoke.run_smoke(tmp_path, tmp_path, tmp_path / "output", iterations=iterations)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("destination", ["repo", "repo/child", "data", "data/child", "."])
def test_output_cannot_overlap_sources_or_data(tmp_path, destination):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        sugar_smoke.run_smoke(repo, data, tmp_path / destination)


def test_existing_output_and_vanilla_checkpoint_cannot_be_accepted(tmp_path):
    repo = fixture_repo(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "old output"
    vanilla = output / "point_cloud" / "iteration_10" / "point_cloud.ply"
    vanilla.parent.mkdir(parents=True)
    vanilla.write_bytes(b"ply\nend_header\n")
    with pytest.raises(FileExistsError):
        sugar_smoke.run_smoke(repo, data, output)
    assert vanilla.read_bytes() == b"ply\nend_header\n"
    assert not (output / "report.json").exists()


def test_sources_and_data_remain_unchanged(tmp_path):
    repo = fixture_repo(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "transforms_train.json").write_text('{"frames": []}', encoding="utf-8")
    def snapshot():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in (repo, data) for p in folder.rglob("*") if p.is_file()}
    before = snapshot()
    sugar_smoke.run_smoke(repo, data, tmp_path / "output")
    assert snapshot() == before  # Includes absence of repository __pycache__ files.


def test_import_probe_hides_gpus_and_uses_current_python(tmp_path):
    repo = fixture_repo(tmp_path)
    module, symbol = sugar_smoke.IMPORT_TARGETS[0]
    repo.joinpath(*module.split(".")).with_suffix(".py").write_text(
        "import os, sys\n"
        "print('python=' + sys.executable)\n"
        "print('cuda=' + repr(os.environ['CUDA_VISIBLE_DEVICES']))\n"
        f"class {symbol}: pass\n", encoding="utf-8"
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    probe = sugar_smoke.probe_import(repo, logs, module, symbol, timeout=5)
    stdout = Path(probe["stdout_path"]).read_text("utf-8")
    assert probe["status"] == "imported"
    assert f"python={sys.executable}" in stdout
    assert "cuda=''" in stdout
    assert probe["command"][0] == sys.executable
    assert "-c" in probe["command"]
    assert not any("train_full_pipeline.py" in arg for arg in probe["command"])


def test_probe_checks_requested_symbol_not_just_package_name(tmp_path):
    repo = fixture_repo(tmp_path)
    module, symbol = sugar_smoke.IMPORT_TARGETS[0]
    repo.joinpath(*module.split(".")).with_suffix(".py").write_text("", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    result = sugar_smoke.probe_import(repo, logs, module, symbol, timeout=5)
    assert result["status"] == "failed"
    assert symbol in Path(result["stderr_path"]).read_text("utf-8")


def test_probe_timeout_preserves_partial_stdout_and_stderr(tmp_path):
    repo = fixture_repo(tmp_path, slow=True)
    logs = tmp_path / "logs"
    logs.mkdir()
    module, symbol = sugar_smoke.IMPORT_TARGETS[0]
    probe = sugar_smoke.probe_import(repo, logs, module, symbol, timeout=0.2)
    assert probe["status"] == "timeout"
    assert probe["returncode"] is None
    assert "before timeout" in Path(probe["stdout_path"]).read_text("utf-8")
    assert "timeout" in Path(probe["stderr_path"]).read_text("utf-8").lower()


def test_changed_or_missing_source_is_blocked_without_running_it(tmp_path):
    repo = fixture_repo(tmp_path)
    (repo / "train_full_pipeline.py").write_text(
        "raise AssertionError('SOURCE EXECUTED')\n", encoding="utf-8"
    )
    audit = sugar_smoke.audit_source(repo)
    assert audit["supported"] is False
    assert "source_contract_unverified" in {b["code"] for b in audit["blockers"]}


@pytest.mark.skipif(not REAL_REPO.is_dir(), reason="external SuGaR checkout absent")
def test_actual_checkout_exposes_all_three_unsafe_coarse_schedules():
    audit = sugar_smoke.audit_source(REAL_REPO)
    assert audit["supported"] is False
    for name in ("coarse_sdf", "coarse_density", "coarse_density_and_dn_consistency"):
        file = audit["files"][f"sugar_trainers/{name}.py"]
        assignments = file["assignments"]
        assert any(a["name"] == "num_iterations" and a["value"] == 15000 for a in assignments)
        assert any(a["name"] == "iteration" and a["value"] == 6999 for a in assignments)
        assert any(a["name"] == "start_sdf_regularization_from" and a["value"] == 9000
                   for a in assignments)
        assert len(file["sha256"]) == 64
        assert file["save_model_lines"]
    assert audit["coarse_smoke_supported"] is True
    assert "mesh_refine_require_sugar_artifacts" in {b["code"] for b in audit["blockers"]}


@pytest.mark.skipif(not REAL_REPO.is_dir(), reason="external SuGaR checkout absent")
def test_full_pipeline_shell_commands_and_missing_coarse_cli_cap_are_recorded():
    audit = sugar_smoke.audit_source(REAL_REPO)
    launches = audit["files"]["train_full_pipeline.py"]["shell_launches"]
    assert len(launches) == 2
    assert any("--iterations 7_000" in launch["source"] for launch in launches)
    flags = {flag for entry in audit["files"]["train.py"]["cli_arguments"]
             for flag in entry["flags"]}
    assert "--refinement_iterations" in flags
    assert "--iterations" not in flags
    assert "full_pipeline_unbounded_launcher" in {b["code"] for b in audit["blockers"]}


@pytest.mark.parametrize("timeout", [0, -1, 61, float("inf"), float("nan")])
def test_invalid_timeout_rejected_before_output(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        sugar_smoke.run_smoke(tmp_path / "repo", tmp_path / "data", tmp_path / "out",
                              import_timeout=timeout)
    assert not (tmp_path / "out").exists()


def test_syntax_error_in_official_source_remains_auditable_and_blocked(tmp_path):
    repo = fixture_repo(tmp_path)
    broken = repo / "sugar_trainers/coarse_sdf.py"
    broken.write_text("def broken(\n", encoding="utf-8")
    audit = sugar_smoke.audit_source(repo)
    record = audit["files"]["sugar_trainers/coarse_sdf.py"]
    assert "SyntaxError" in record["error"]
    assert record["sha256"] == hashlib.sha256(broken.read_bytes()).hexdigest()
    assert audit["supported"] is False


def test_cli_blocks_nonzero_and_writes_diagnostics_without_training(tmp_path):
    repo = fixture_repo(tmp_path, failing=True)
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "output"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "common/sugar_smoke.py"),
         "--repo", str(repo), "--data", str(data), "--output", str(output),
         "--iterations", "10", "--import-timeout", "5"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=30,
    )
    assert completed.returncode == 2
    assert b"blocked" in completed.stdout
    report = json.loads((output / "report.json").read_text("utf-8"))
    assert not report["success"]
    assert report["sugar_checkpoint"] is None


def load_window_helper():
    """Exercise trainer's budget policy without importing CUDA dependencies."""
    path = REAL_REPO / "sugar_trainers/coarse_density_and_dn_consistency.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "_smoke_iteration_window"]
    assert len(helpers) == 1, "Official trainer needs an explicit optional smoke budget"
    namespace = {}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_smoke_iteration_window"]


def test_trainer_window_decouples_ten_updates_from_7000_and_preserves_default():
    window = load_window_helper()
    assert window(SimpleNamespace()) is None
    assert window(SimpleNamespace(smoke_iterations=None)) is None
    assert window(SimpleNamespace(smoke_iterations=10, iteration_to_load=10)) == (11, 20)
    assert window(SimpleNamespace(smoke_iterations=10, iteration_to_load=7000)) == (7001, 7010)


@pytest.mark.parametrize("budget", [0, -1, 11, 7000, True, 10.0])
def test_trainer_rejects_unsafe_smoke_budget_before_model_initialization(budget):
    window = load_window_helper()
    with pytest.raises(ValueError):
        window(SimpleNamespace(smoke_iterations=budget, iteration_to_load=10))


def test_checkpoint_validator_rejects_vanilla_and_wrong_update_counts():
    torch = pytest.importorskip("torch")
    checkpoint = {
        "state_dict": {key: torch.ones(4, 3) for key in sugar_smoke.SUGAR_STATE_KEYS},
        "iteration": 20, "optimizer_state_dict": {
            "state": {0: {"step": torch.tensor(10)}}, "param_groups": [{"params": [0]}]},
        "smoke_metadata": {"scope": "coarse_training", "actual_updates": 10,
                           "first_iteration": 11, "last_iteration": 20,
                           "initialization_iteration": 10, "losses": [0.1] * 10,
                           "full_pipeline_verified": False},
    }
    assert sugar_smoke.validate_sugar_checkpoint(checkpoint, torch, 10)["actual_updates"] == 10
    with pytest.raises(ValueError):
        sugar_smoke.validate_sugar_checkpoint(({"xyz": torch.ones(4, 3)}, 10), torch, 10)
    checkpoint["smoke_metadata"]["actual_updates"] = 9
    with pytest.raises(ValueError, match="updates"):
        sugar_smoke.validate_sugar_checkpoint(checkpoint, torch, 10)


def test_stage_worker_command_has_explicit_budget_and_no_shell(tmp_path):
    command = sugar_smoke.build_worker_command("coarse", tmp_path / "repo", tmp_path / "data",
                                               tmp_path / "output", 10)
    assert command[0] == sys.executable
    assert command[command.index("--iterations") + 1] == "10"
    assert command[command.index("--_worker") + 1] == "coarse"
    assert not any("train_full_pipeline.py" in token for token in command)


def mock_training_boundary(monkeypatch, tmp_path, fail_stage=None):
    """Orchestration unit double, never evidence of real algorithm training."""
    repo = fixture_repo(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(sugar_smoke, "audit_source", lambda _: {
        "coarse_smoke_supported": True, "blockers": [], "files": {}})
    def probe(repo, logs, module, symbol, **kwargs):
        stdout, stderr = logs / (module + ".out"), logs / (module + ".err")
        stdout.write_text("", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        # Optional mesh import failures must not invalidate coarse environment success.
        status = "failed" if module.startswith("sugar_extractors.") else "imported"
        return {"module": module, "status": status, "stdout_path": str(stdout), "stderr_path": str(stderr)}
    monkeypatch.setattr(sugar_smoke, "probe_import", probe)
    monkeypatch.setattr(sugar_smoke, "_prepare_data", lambda data, output: data)
    calls = []
    def execute(stage, repo, data, output, iterations, timeout):
        calls.append(stage)
        stderr = output / f"{stage}.stderr.log"
        stderr.write_text("unit boundary fixture", encoding="utf-8")
        if stage == fail_stage:
            return {"status": "failed", "returncode": 1, "stderr_path": str(stderr)}
        checkpoint = output / f"{stage}.unit_fixture"
        checkpoint.write_text("NOT A MODEL; orchestration test only", encoding="utf-8")
        return {"status": "passed", "returncode": 0, "actual_updates": 10 if stage == "coarse" else 9,
                "executed_iterations": 10, "checkpoint_loaded": True,
                "checkpoint": str(checkpoint), "artifacts": [str(checkpoint)], "stderr_path": str(stderr)}
    monkeypatch.setattr(sugar_smoke, "execute_worker", execute)
    return repo, data, calls


def test_passed_report_is_only_coarse_scope_and_mesh_is_not_verified(monkeypatch, tmp_path):
    repo, data, calls = mock_training_boundary(monkeypatch, tmp_path)
    output = tmp_path / "output"
    output.mkdir()  # Controller may reserve an empty method/train directory.
    report = sugar_smoke.run_smoke(repo, data, output)
    assert calls == ["vanilla", "coarse"]
    assert report["status"] == "passed" and report["exit_code"] == 0
    assert report["scope"] == "coarse_training"
    assert report["pipeline_status"] == "partial_coarse"
    assert report["actual_updates"] == 10
    assert report["full_pipeline_verified"] is False
    assert report["stages"]["mesh"]["status"] == "not_verified"
    assert report["stages"]["sugar_refine"]["status"] == "not_verified"
    assert json.loads((output / "sugar_smoke_report.json").read_text("utf-8")) == report


@pytest.mark.parametrize("failed", ["vanilla", "coarse"])
def test_vanilla_success_alone_cannot_pass_sugar(monkeypatch, tmp_path, failed):
    repo, data, calls = mock_training_boundary(monkeypatch, tmp_path, fail_stage=failed)
    report = sugar_smoke.run_smoke(repo, data, tmp_path / "out")
    assert calls == (["vanilla"] if failed == "vanilla" else ["vanilla", "coarse"])
    assert report["status"] == "failed" and report["exit_code"] == 1
    assert report["actual_updates"] == 0 and report["sugar_checkpoint"] is None
    assert report["full_pipeline_verified"] is False


def test_preflight_only_never_launches_training(monkeypatch, tmp_path):
    repo, data, calls = mock_training_boundary(monkeypatch, tmp_path)
    report = sugar_smoke.run_smoke(repo, data, tmp_path / "out", preflight_only=True)
    assert not calls
    assert report["status"] == "blocked" and not report["training_started"]


def test_prepare_data_keeps_original_input_readonly(tmp_path):
    data = tmp_path / "data"
    sparse = data / "sparse/0"
    sparse.mkdir(parents=True)
    (data / "images").mkdir()
    (data / "images/view.png").write_bytes(b"unit input fixture, not a training image")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).write_bytes(b"unit COLMAP layout fixture")
    output = tmp_path / "output"
    output.mkdir()
    copied = sugar_smoke._prepare_data(data, output)
    (copied / "sparse/0/points3D.ply").write_bytes(b"loader-generated output")
    assert not (sparse / "points3D.ply").exists()
    assert (copied / "images/view.png").read_bytes() == (data / "images/view.png").read_bytes()


def test_official_save_method_serializes_real_cpu_tensor_checkpoint(tmp_path):
    """Checks the real official save method/torch reload; no algorithm training."""
    torch = pytest.importorskip("torch")
    path = REAL_REPO / "sugar_scene/sugar_model.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    model_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SuGaR")
    save = next(n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == "save_model")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[save], type_ignores=[]), str(path), "exec"), namespace)
    state = {key: torch.ones(4, 3) for key in sugar_smoke.SUGAR_STATE_KEYS}
    holder = SimpleNamespace(state_dict=lambda: state)
    output = tmp_path / "20.pt"
    namespace["save_model"](holder, str(output), iteration=20,
        optimizer_state_dict={"state": {0: {"step": torch.tensor(10)}}, "param_groups": [{"params": [0]}]},
        smoke_metadata={"scope": "coarse_training", "actual_updates": 10, "first_iteration": 11,
                        "last_iteration": 20, "initialization_iteration": 10,
                        "losses": [0.1] * 10, "full_pipeline_verified": False})
    reloaded = torch.load(output, map_location="cpu", weights_only=True)
    assert sugar_smoke.validate_sugar_checkpoint(reloaded, torch, 10)["checkpoint_loaded"]
    reloaded["state_dict"]["_points"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        sugar_smoke.validate_sugar_checkpoint(reloaded, torch, 10)
