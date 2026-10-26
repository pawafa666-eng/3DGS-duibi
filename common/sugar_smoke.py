"""Official vanilla + SuGaR coarse smoke adapter (standard-library entry).

PUBLIC CONTRACT: --repo --data --output --iterations 10, from the intended
SuGaR Python, including WSL Ubuntu. Vanilla runs ten official iterations and
saves iteration_10; upstream skips the last optimizer step, hence nine updates.
The opt-in patched dn_consistency trainer then performs exactly TEN optimizer
updates, 11..20, starting from the initialization counter 10. The default
smoke_iterations=None leaves the formal trainer path unchanged.

SUCCESS IS SCOPED: sugar_smoke_report.json status=passed, scope=coarse_training,
actual_updates=10, full_pipeline_verified=false validates only coarse training
and its genuine checkpoint. The caller must classify it as partial_coarse.
Mesh/refine remain not_verified. Entropy/SDF/depth-normal losses gated beyond
7000/9000 are excluded; no loss schedules are compressed. Exit 0 means scoped
coarse success, exit 2 blocked, exit 1 execution failure. report.json is an alias.

The normalized patched trainer SHA256 is pinned below. Unknown trainers fail
closed rather than run 7000/15000 steps. Workers use this same interpreter and
official optimization/save functions, with a subprocess timeout. Input data is
copied under output before loaders generate PLY files. --preflight-only disables
training. No dependency installation or replacement algorithm is used.

SOURCE CONTRACT, audited checkout 7c10c4ae4a267dece512f5c7f40ed212a0a2ab44:
* train_full_pipeline.py launches vanilla for 7000 steps and train.py with
  iteration_to_load=7000 via os.system; neither is a safe short-test launcher.
* All three sugar_trainers/coarse_*.py functions hard-code num_iterations=15000,
  restart their counter at 7000-1, and activate their surface loss after 9000.
  Their args have no coarse iteration budget. Setting a new args.iterations
  attribute is ignored. An absolute cap of 10 executes no coarse updates;
  ten additional updates reach 7009, before the defining surface losses.
* refine.py DOES read args.refinement_iterations, but needs a real coarse
  SuGaR checkpoint and extracted mesh. A vanilla PLY cannot replace either.
* Real SuGaR saves use SuGaR.save_model -> torch.save on a checkpoint dict
  containing state_dict, iteration, optimizer_state_dict, etc., rather than
  vanilla point_cloud/iteration_10/*.ply.

The unmodified full-pipeline contract remains unsupported; the optional trainer
patch provides ONLY the explicitly scoped coarse smoke contract above.

Usage, from the intended SuGaR Python environment:
  python common/sugar_smoke.py --repo 04_SuGaR/repo --data <synthetic-scene> \
      --output <new-diagnostics-directory> --iterations 10

Only a new or empty output directory outside repo/data is allowed. Budgets
apply per executed stage; Mesh/refine are not run. Source hashes, counters,
interpreter, imports and checkpoint validation are recorded.
imports/*.stderr.log and aggregated stderr.log preserve actual tracebacks.
Imports use sys.executable in isolated subprocesses with GPUs hidden, bytecode
writes disabled, and a bounded timeout. Import success is dependency evidence
only; training workers preserve the caller's CUDA visibility.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace


TRAINER_RELATIVE = "sugar_trainers/coarse_density_and_dn_consistency.py"
TRAINER_SMOKE_SHA256 = "ecb6f3248bccc346097de1deb76288f6baf59d6d068d8a99c48066c4fe5225f2"
SUGAR_STATE_KEYS = ("_points", "_scales", "_quaternions", "_sh_coordinates_dc", "all_densities")


IMPORT_TARGETS = (
    ("sugar_scene.sugar_model", "SuGaR"),
    ("sugar_trainers.coarse_sdf", "coarse_training_with_sdf_regularization"),
    ("sugar_trainers.coarse_density", "coarse_training_with_density_regularization"),
    ("sugar_trainers.coarse_density_and_dn_consistency",
     "coarse_training_with_density_regularization_and_dn_consistency"),
    ("sugar_extractors.coarse_mesh", "extract_mesh_from_coarse_sugar"),
    ("sugar_trainers.refine", "refined_training"),
    ("sugar_extractors.refined_mesh", "extract_mesh_and_texture_from_refined_sugar"),
)
SOURCE_FILES = (
    "train_full_pipeline.py", "train.py", "gaussian_splatting/train.py",
    "sugar_scene/gs_model.py", "sugar_scene/sugar_model.py",
) + tuple(module.replace(".", "/") + ".py" for module, _ in IMPORT_TARGETS[1:])
TRACKED_ASSIGNMENTS = {
    "num_iterations", "iteration", "initialize_from_trained_3dgs",
    "start_sdf_regularization_from", "start_sdf_estimation_from",
    "start_dn_consistency_from", "regularize_from", "save_milestones",
    "surface_mesh_to_bind_path", "sugar_checkpoint_path",
}


def _constant(node):
    """Evaluate only literal evidence and integer +/-; never execute repo code."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = _constant(node.left), _constant(node.right)
        if type(left) is int and type(right) is int:
            return left + right if isinstance(node.op, ast.Add) else left - right
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def audit_source(repo):
    """Read-only AST evidence, not a capability promise for arbitrary versions."""
    repo = Path(repo).resolve()
    files = {}
    for relative in SOURCE_FILES:
        record = {"assignments": [], "args_read": [], "save_model_lines": [],
                  "shell_launches": [], "cli_arguments": []}
        files[relative] = record
        try:
            path = repo / relative
            # Do not follow a source symlink outside the selected checkout.
            path.resolve().relative_to(repo)
            raw = path.read_bytes()
            record["sha256"] = hashlib.sha256(raw).hexdigest()
            source = raw.decode("utf-8-sig")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in TRACKED_ASSIGNMENTS:
                            record["assignments"].append({
                                "name": target.id, "line": node.lineno,
                                "expression": ast.get_source_segment(source, node.value),
                                "value": _constant(node.value),
                            })
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id == "args"):
                    record["args_read"].append(node.attr)
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "save_model"):
                    record["save_model_lines"].append(node.lineno)
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "os" and node.func.attr == "system"):
                    record["shell_launches"].append({
                        "line": node.lineno, "source": ast.get_source_segment(source, node),
                    })
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "add_argument"):
                    record["cli_arguments"].append({
                        "line": node.lineno,
                        "flags": [arg.value for arg in node.args
                                  if isinstance(arg, ast.Constant) and isinstance(arg.value, str)],
                    })
            record["args_read"] = sorted(set(record["args_read"]))
        except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"

    blockers = []
    coarse_files = [name for name in files if name.startswith("sugar_trainers/coarse_")]
    schedule_matches = True
    evidence = []
    for name in coarse_files:
        record = files[name]
        # An added dynamic override must invalidate this specific source diagnosis.
        limits = [a for a in record["assignments"] if a["name"] == "num_iterations"]
        schedule_matches = schedule_matches and bool(limits) and all(
            a["value"] == 15000 for a in limits)
        for variable, expected in (("num_iterations", 15000), ("iteration", 6999),
                                   ("start_sdf_regularization_from", 9000)):
            matching = [a for a in record["assignments"]
                        if a["name"] == variable and a["value"] == expected]
            schedule_matches = schedule_matches and bool(matching)
            evidence.extend(f"{name}:{a['line']}" for a in matching)
    if schedule_matches:
        blockers.extend([
            {"code": "coarse_has_no_budget_argument", "evidence": evidence,
             "message": "All three official coarse trainers hard-code 15000 as the "
                        "terminal iteration and start at 6999. Passing args.iterations=10 "
                        "cannot bound the unpatched functions. Only the explicitly "
                        "opt-in patched dn_consistency smoke path is called."},
            {"code": "surface_loss_outside_budget", "evidence": evidence,
             "message": "A terminal iteration of 10 runs zero coarse updates from 6999. "
                        "Ten additional updates end at 7009, but SDF/density surface "
                        "regularization starts only after 9000 (dn_consistency also "
                        "starts after 9000). No schedule compression or 7000-step "
                        "fallback is performed."},
        ])
    else:
        blockers.append({
            "code": "source_contract_unverified", "evidence": coarse_files,
            "message": "Coarse source is absent, changed or unrecognized. No bounded "
                       "training integration is verified for this checkout; fail closed.",
        })
    errors = [name for name, record in files.items() if "error" in record]
    if errors:
        blockers.append({"code": "source_audit_incomplete", "evidence": errors,
                         "message": "Required official source files could not be audited."})
    launches = files["train_full_pipeline.py"]["shell_launches"]
    if any("--iterations 7_000" in launch["source"] for launch in launches):
        blockers.append({
            "code": "full_pipeline_unbounded_launcher",
            "evidence": [f"train_full_pipeline.py:{launch['line']}" for launch in launches],
            "message": "The official full-pipeline launcher uses os.system, hard-codes "
                       "7000 vanilla iterations and loads iteration 7000. Refinement "
                       "iterations alone do not cap the pipeline; this launcher is never run.",
        })
    blockers.append({
        "code": "mesh_refine_require_sugar_artifacts",
        "evidence": ["train.py", "sugar_extractors/coarse_mesh.py", "sugar_trainers/refine.py"],
        "message": "Mesh requires a real coarse SuGaR state_dict checkpoint; refine "
                   "requires its extracted mesh. Neither stage is executed. Vanilla "
                   "PLY files or successful imports cannot satisfy these dependencies.",
    })
    trainer_path = repo / TRAINER_RELATIVE
    normalized_hash = None
    try:
        normalized_hash = hashlib.sha256(trainer_path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()
    except (OSError, UnicodeError):
        pass
    coarse_supported = normalized_hash == TRAINER_SMOKE_SHA256
    if coarse_supported:
        blockers = [b for b in blockers if b["code"] != "source_contract_unverified"]
    return {"supported": False, "coarse_smoke_supported": coarse_supported,
            "trainer_normalized_sha256": normalized_hash, "files": files, "blockers": blockers}


_IMPORT_CODE = r"""
import importlib, os, pathlib, sys
repo = pathlib.Path(sys.argv[1]).resolve()
sys.path[:0] = [str(repo), str(repo / 'gaussian_splatting')]
sys.path.extend(p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep)
                if p and pathlib.Path(p).is_absolute())
module = importlib.import_module(sys.argv[2])
path = pathlib.Path(module.__file__).resolve()
path.relative_to(repo)  # Installed modules from another checkout cannot satisfy the probe.
symbol = getattr(module, sys.argv[3])
if not callable(symbol):
    raise TypeError('Requested SuGaR symbol is not callable: ' + sys.argv[3])
print('Imported ' + sys.argv[2] + ':' + sys.argv[3] + ' from ' + str(path), flush=True)
"""


def probe_import(repo, logs, module, symbol, *, timeout=30):
    """Import official symbols without invoking them; preserve raw child output."""
    if (module, symbol) not in IMPORT_TARGETS:
        raise ValueError("Only audited SuGaR import targets are permitted")
    if not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("Import timeout must be greater than 0 and at most 60 seconds")
    repo, logs = Path(repo).resolve(), Path(logs).resolve()
    stdout_path = logs / f"{module}.stdout.log"
    stderr_path = logs / f"{module}.stderr.log"
    # -I ignores PYTHONIOENCODING; -X utf8 also protects Windows Chinese paths.
    command = [sys.executable, "-I", "-B", "-X", "utf8", "-u", "-c", _IMPORT_CODE,
               str(repo), module, symbol]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    result = {"module": module, "symbol": symbol, "command": command,
              "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
              "returncode": None, "status": "failed"}
    start = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            child = subprocess.run(command, cwd=repo, env=env, stdout=stdout,
                                   stderr=stderr, timeout=timeout, check=False)
            result["returncode"] = child.returncode
            result["status"] = "imported" if child.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            stderr.write(f"\nImport timeout after {timeout} seconds; child killed.\n".encode())
        except OSError as exc:
            stderr.write(f"\n{type(exc).__name__}: {exc}\n".encode("utf-8"))
    result["duration_sec"] = round(time.monotonic() - start, 3)
    return result


def _overlap(left, right):
    return left == right or left in right.parents or right in left.parents


def validate_sugar_checkpoint(checkpoint, torch, iterations):
    """Validate a freshly produced SuGaR checkpoint, not a vanilla PLY/tuple."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Not a SuGaR checkpoint dictionary")
    state = checkpoint.get("state_dict", {})
    for key in SUGAR_STATE_KEYS:
        value = state.get(key)
        if not torch.is_tensor(value) or not value.numel() or not torch.isfinite(value).all():
            raise ValueError(f"Missing, empty or non-finite SuGaR tensor: {key}")
    metadata = checkpoint.get("smoke_metadata", {})
    if metadata.get("actual_updates") != iterations:
        raise ValueError("SuGaR checkpoint actual_updates differs from requested updates")
    if (metadata.get("scope") != "coarse_training"
            or metadata.get("full_pipeline_verified") is not False
            or metadata.get("initialization_iteration") != iterations
            or metadata.get("first_iteration") != iterations + 1
            or metadata.get("last_iteration") != 2 * iterations
            or checkpoint.get("iteration") != 2 * iterations):
        raise ValueError("SuGaR checkpoint scope or iteration window is inconsistent")
    losses = metadata.get("losses", [])
    if len(losses) != iterations or not all(math.isfinite(v) for v in losses):
        raise ValueError("SuGaR checkpoint must contain ten finite step losses")
    optimizer = checkpoint.get("optimizer_state_dict", {})
    steps = [float(value["step"]) for value in optimizer.get("state", {}).values() if "step" in value]
    if (not optimizer.get("param_groups") or not steps or max(steps) != iterations
            or any(not math.isfinite(v) or not 0 <= v <= iterations for v in steps)):
        raise ValueError("SuGaR optimizer state does not corroborate ten real updates")
    return {**metadata, "optimizer_max_step": max(steps), "n_gaussians": int(state["_points"].shape[0]),
            "validated_state_keys": list(SUGAR_STATE_KEYS), "checkpoint_loaded": True}


def build_worker_command(stage, repo, data, output, iterations):
    if stage not in ("vanilla", "coarse") or type(iterations) is not int or iterations != 10:
        raise ValueError("Worker requires vanilla/coarse and exactly 10 iterations")
    return [sys.executable, "-I", "-B", "-X", "utf8", "-u", str(Path(__file__).resolve()),
            "--_worker", stage, "--repo", str(repo), "--data", str(data),
            "--output", str(output), "--iterations", str(iterations)]


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_report(output, report):
    for name in ("sugar_smoke_report.json", "report.json"):
        _atomic_json(output / name, report)


def _run_worker(stage, repo, data, output, iterations):
    """The only functions here importing ML libraries run in an explicit worker."""
    sys.path[:0] = [str(repo / "gaussian_splatting"), str(repo)]
    # -I ignores automatic PYTHONPATH processing; explicitly include the
    # controller-selected extension overlay for this algorithm only.
    sys.path.extend(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                    if p and Path(p).is_absolute())
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the selected SuGaR Python environment")
    if stage == "vanilla":
        vanilla = output / "vanilla"
        # Import the official training module without its GUI/server main block.
        official = runpy.run_path(str(repo / "gaussian_splatting/train.py"), run_name="sugar_smoke_vanilla")
        parser = argparse.ArgumentParser()
        model = official["ModelParams"](parser)
        opt = official["OptimizationParams"](parser)
        pipe = official["PipelineParams"](parser)
        args = parser.parse_args(["--source_path", str(data), "--model_path", str(vanilla),
                                  "--iterations", str(iterations)])
        official["safe_state"](False)
        official["training"](model.extract(args), opt.extract(args), pipe.extract(args),
                             [], [iterations], [iterations], None, -1)
        ply = vanilla / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
        checkpoint = vanilla / f"chkpnt{iterations}.pth"
        if not ply.is_file() or not checkpoint.is_file():
            raise RuntimeError("Official vanilla iteration-10 saves are missing")
        # This is trusted output created in this fresh run, never an input pickle.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(saved, tuple) or len(saved) != 2 or saved[1] != iterations:
            raise ValueError("Invalid vanilla initialization checkpoint")
        from plyfile import PlyData
        points = PlyData.read(str(ply))["vertex"]
        if not len(points) or any(not torch.isfinite(torch.as_tensor(points[key].copy())).all()
                                  for key in ("x", "y", "z", "opacity")):
            raise ValueError("Vanilla PLY has empty or non-finite geometry")
        cameras = json.loads((vanilla / "cameras.json").read_text(encoding="utf-8"))
        if not cameras:
            raise ValueError("Vanilla initialization saved no cameras")
        result = {"status": "passed", "training_iterations": iterations,
                  "executed_iterations": iterations, "actual_updates": iterations - 1,
                  "checkpoint": str(checkpoint), "point_cloud": str(ply), "n_gaussians": len(points),
                  "artifacts": [str(ply), str(checkpoint)], "checkpoint_loaded": True,
                  "counter_note": "Official vanilla skips optimizer.step on the last iteration."}
    else:
        if not audit_source(repo)["coarse_smoke_supported"]:
            raise RuntimeError("Patched SuGaR trainer hash differs; refusing unbounded training")
        from sugar_trainers.coarse_density_and_dn_consistency import (
            coarse_training_with_density_regularization_and_dn_consistency as train_coarse)
        args = SimpleNamespace(
            scene_path=str(data), checkpoint_path=str(output / "vanilla") + os.sep,
            output_dir=str(output / "coarse"), iteration_to_load=iterations,
            smoke_iterations=iterations, estimation_factor=0.2, normal_factor=0.2,
            eval=False, gpu=0, white_background=False,
        )
        path = Path(train_coarse(args)).resolve()
        path.relative_to((output / "coarse").resolve())
        if path.suffix != ".pt" or not path.is_file() or not path.stat().st_size:
            raise ValueError("Official SuGaR trainer did not save a genuine coarse checkpoint")
        # This checkpoint was just produced by this worker in its fresh output
        # folder. Official SuGaR metadata includes NumPy scalar values.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        validation = validate_sugar_checkpoint(checkpoint, torch, iterations)
        result = {"status": "passed", "executed_iterations": iterations, **validation,
                  "checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "artifacts": [str(path)], "surface_regularization_executed": False,
                  "excluded": ["entropy regularization (>7000)", "SDF/density surface losses (>9000)",
                               "depth-normal consistency (>9000)", "mesh", "refine"]}
    _atomic_json(output / f"{stage}_worker_result.json", result)
    return 0


def execute_worker(stage, repo, data, output, iterations, timeout):
    command = build_worker_command(stage, repo, data, output, iterations)
    stdout_path, stderr_path = output / f"{stage}.stdout.log", output / f"{stage}.stderr.log"
    result = {"command": command, "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
              "status": "failed", "returncode": None}
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            child = subprocess.run(command, cwd=repo, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                   stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            result["returncode"] = child.returncode
            if child.returncode == 0:
                result.update(json.loads((output / f"{stage}_worker_result.json").read_text(encoding="utf-8")))
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            stderr.write(f"\nTraining worker timeout after {timeout}s; killed.\n".encode())
        except (OSError, ValueError) as exc:
            stderr.write(f"\n{type(exc).__name__}: {exc}\n".encode("utf-8"))
    return result


def _prepare_data(data, output):
    sparse = data / "sparse/0"
    if (not (data / "images").is_dir() or not any((data / "images").iterdir())
            or not any((sparse / f"cameras.{ext}").is_file() and (sparse / f"images.{ext}").is_file()
                       for ext in ("bin", "txt"))
            or not any((sparse / f"points3D.{ext}").is_file() for ext in ("bin", "txt"))):
        raise ValueError("SuGaR coarse smoke needs standard COLMAP images and sparse/0 cameras/images/points3D")
    target = output / "input"
    shutil.copytree(data / "images", target / "images")
    shutil.copytree(data / "sparse", target / "sparse")
    return target


def run_smoke(repo, data, output, *, iterations=10, import_timeout=30, stage_timeout=180,
              preflight_only=False):
    """Run official bounded stages; passing applies strictly to coarse_training."""
    if type(iterations) is not int or iterations != 10:
        raise ValueError("The smoke budget must be exactly 10 iterations per training stage")
    if not math.isfinite(import_timeout) or not 0 < import_timeout <= 60:
        raise ValueError("Import timeout must be greater than 0 and at most 60 seconds")
    if not math.isfinite(stage_timeout) or not 0 < stage_timeout <= 1800:
        raise ValueError("Stage timeout must be greater than 0 and at most 1800 seconds")
    repo, data, output = (Path(path).resolve() for path in (repo, data, output))
    if _overlap(output, repo) or _overlap(output, data):
        raise ValueError("Output must not overlap the repository or input data")
    if not repo.is_dir() or not data.is_dir():
        raise ValueError("--repo and --data must be existing directories")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be new or empty; refusing stale artifacts")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "imports"
    logs.mkdir()
    audit = audit_source(repo)
    stages = {}
    for stage in ("vanilla_initialization", "sugar_coarse", "mesh", "sugar_refine"):
        stages[stage] = {
            "status": "not_verified" if stage in ("mesh", "sugar_refine") else "blocked",
            "requested_iterations": None if stage in ("mesh", "sugar_refine") else 10,
            "executed_iterations": 0, "artifacts": [],
            "reason": "Outside coarse_training smoke scope." if stage in ("mesh", "sugar_refine")
                      else "Preflight pending; no training started.",
        }
    imports = [probe_import(repo, logs, module, symbol, timeout=import_timeout)
               for module, symbol in IMPORT_TARGETS]
    blockers = []
    if not audit["coarse_smoke_supported"]:
        blockers.append({"code": "source_contract_unverified", "evidence": [TRAINER_RELATIVE],
                         "message": "Apply patches/04_SuGaR_smoke.patch to the audited upstream "
                                    "trainer; no bounded integration is approved for this source hash."})
    required = {IMPORT_TARGETS[0][0], IMPORT_TARGETS[3][0]}
    failed = [probe["module"] for probe in imports
              if probe["module"] in required and probe["status"] != "imported"]
    if failed:
        blockers.append({"code": "environment_import_failed", "evidence": failed,
                         "message": "Real SuGaR imports failed or timed out in the current "
                                    "interpreter; inspect stderr.log and imports/*.stderr.log."})
    report = {
        "schema_version": 2, "algorithm": "SuGaR", "status": "blocked", "exit_code": 2,
        "scope": "coarse_training", "actual_updates": 0, "full_pipeline_verified": False,
        "success": False, "training_started": False, "sugar_checkpoint": None,
        "artifacts": [], "iterations": iterations, "budget_scope": "per_training_stage",
        "python_executable": sys.executable, "python_version": sys.version,
        "repo": str(repo), "data": str(data), "output": str(output),
        "data_validation": "not_performed_training_blocked",
        "stages": stages, "source_audit": audit, "imports": imports, "blockers": blockers,
        "limitations": ["Mesh/refine not executed or verified.",
                        "Entropy, SDF/density surface losses and depth-normal consistency excluded "
                        "at coarse counter 11..20; no geometric reconstruction claim.",
                        "Vanilla runs 10 iterations but upstream performs 9 optimizer updates."],
    }
    if preflight_only:
        blockers.append({"code": "preflight_only", "evidence": [],
                         "message": "Training explicitly disabled by --preflight-only."})
    _write_report(output, report)
    if not blockers:
        try:
            effective_data = _prepare_data(data, output)
            report["effective_data"] = str(effective_data)
            report["data_validation"] = "standard_colmap_layout_copied_input"
        except (OSError, ValueError) as exc:
            blockers.append({"code": "invalid_data", "evidence": [str(data)], "message": str(exc)})
    if not blockers:
        report["training_started"] = True
        _write_report(output, report)
        for worker, key in (("vanilla", "vanilla_initialization"), ("coarse", "sugar_coarse")):
            stages[key] = {**stages[key], "status": "running", "reason": "Official training worker executing."}
            _write_report(output, report)
            execution = execute_worker(worker, repo, effective_data, output, iterations, stage_timeout)
            stages[key].update(execution)
            if execution["status"] != "passed":
                report["status"], report["exit_code"] = "failed", 1
                blockers.append({"code": f"{worker}_training_failed", "evidence": [execution["stderr_path"]],
                                 "message": "Official training/save/validation worker failed; see raw stderr."})
                break
            stages[key].pop("reason", None)
        else:
            coarse = stages["sugar_coarse"]
            # No PLY discovery/globbing: require this run's validated official save.
            if (coarse.get("actual_updates") == iterations and coarse.get("checkpoint_loaded") is True
                    and Path(coarse["checkpoint"]).is_file()):
                report.update(status="passed", exit_code=0, success=True, actual_updates=iterations,
                              sugar_checkpoint=coarse["checkpoint"], artifacts=coarse["artifacts"],
                              pipeline_status="partial_coarse")
            else:
                report.update(status="failed", exit_code=1)
                blockers.append({"code": "coarse_save_not_verified", "evidence": [],
                                 "message": "Coarse execution lacks verified updates and SuGaR checkpoint."})
    with (output / "stderr.log").open("wb") as stderr:
        for blocker in blockers:
            stderr.write(f"[blocked:{blocker['code']}] {blocker['message']}\n".encode("utf-8"))
        for probe in imports:
            stderr.write(f"\n--- {probe['module']} ({probe['status']}) ---\n".encode("utf-8"))
            stderr.write(Path(probe["stderr_path"]).read_bytes())
        for stage in stages.values():
            if "stderr_path" in stage:
                stderr.write(f"\n--- worker {stage['status']} ---\n".encode())
                stderr.write(Path(stage["stderr_path"]).read_bytes())
    _write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, choices=(10,), default=10)
    parser.add_argument("--import-timeout", type=float, default=30,
                        help="Per-module import deadline in seconds, 0 < timeout <= 60")
    parser.add_argument("--stage-timeout", type=float, default=180,
                        help="Per-training-worker timeout in seconds, at most 1800")
    parser.add_argument("--preflight-only", action="store_true", help="Import/source diagnostics without training")
    parser.add_argument("--_worker", choices=("vanilla", "coarse"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return _run_worker(args._worker, args.repo.resolve(), args.data.resolve(), args.output.resolve(),
                           args.iterations)
    try:
        report = run_smoke(args.repo, args.data, args.output, iterations=args.iterations,
                           import_timeout=args.import_timeout, stage_timeout=args.stage_timeout,
                           preflight_only=args.preflight_only)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "scope": report["scope"],
                      "actual_updates": report["actual_updates"], "full_pipeline_verified": False,
                      "report": str(Path(report["output"]) / "sugar_smoke_report.json")}, ensure_ascii=True))
    print(f"{report['status'].upper()}: coarse_training scope only; full_pipeline_verified=false; "
          "see sugar_smoke_report.json and stderr.log.",
          file=sys.stderr)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
