from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .contracts import DatasetLayout, ExperimentDefinition, StageDefinition
from .dataset_compat import (
    DatasetCompatibilityError,
    dataset_compatibility_mode,
    prepare_context_dataset,
)
from .telemetry import (
    CommandObservation,
    average_iteration_time,
    collect_model_artifacts,
    count_changed_render_images,
    observe_process,
    snapshot_render_images,
)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def next_run_dir(output_root: Path) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    numbers = [int(child.name) for child in output_root.iterdir() if child.is_dir() and child.name.isdigit()]
    return output_root / f"{max(numbers, default=0) + 1:02d}"


def resolve_resume_dir(output_root: Path, value: str | Path) -> Path:
    raw = Path(value)
    candidate = output_root / f"{int(raw.name):02d}" if raw.name.isdigit() and not raw.is_absolute() and len(raw.parts) == 1 else raw
    candidate = candidate if candidate.is_absolute() else Path(output_root).parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Resume run directory does not exist: {candidate}")
    return candidate


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def should_skip_experiment(experiment_output: Path) -> bool:
    status = _read_json(Path(experiment_output) / "status.json")
    if not status or status.get("status") != "completed":
        return False
    artifacts = status.get("artifacts", [])
    if not isinstance(artifacts, list):
        return False
    return all(_artifact_ready(Path(experiment_output), str(artifact)) for artifact in artifacts)


def _artifact_ready(output_dir: Path, artifact: str) -> bool:
    """Require real stage output instead of accepting controller-created directories."""
    path = Path(output_dir) / artifact
    if path.is_file():
        return path.stat().st_size > 0
    if not path.is_dir() or not any(item.is_file() for item in path.rglob("*")):
        return False
    if artifact == "train":
        return collect_model_artifacts(path).get("model_file") is not None
    if artifact == "test":
        metrics_path = path / "test_metrics.json"
        metrics = _read_json(metrics_path)
        if metrics is None:
            return True
        if metrics.get("status") != "completed":
            return False
        return int(metrics.get("num_images", 0)) > 0
    return True


def expand_command(command: Iterable[str], values: dict[str, Any]) -> list[str]:
    return [str(part).format_map({key: str(value) for key, value in values.items()}) for part in command]


def command_line(command: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_commands(path: Path, commands: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(command_line(command) for command in commands) + "\n", encoding="utf-8")


def _tail_text(path: Path, max_chars: int = 4_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:].strip()


def _status(output_dir: Path, **updates: Any) -> dict[str, Any]:
    current = _read_json(output_dir / "status.json") or {}
    current.update(updates)
    current["updated_at"] = now_iso()
    atomic_write_json(output_dir / "status.json", current)
    return current


def _stage_context(context: dict[str, Any], stage: StageDefinition, stage_name: str) -> dict[str, Any]:
    values = dict(context)
    values["stage"] = stage_name
    values["cwd"] = context["repo"] if stage.cwd == "repo" else context["experiment_dir"] / stage.cwd
    return values


def _script_missing(command: list[str]) -> str | None:
    for token in command:
        if token.lower().endswith((".py", ".ps1", ".bat", ".exe")) and not Path(token).exists():
            return token
    return None


def _write_console(stream: Any, data: bytes) -> None:
    """Write child output to a console or pytest-style text capture safely."""
    if not data:
        return
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        try:
            buffer.write(data)
            buffer.flush()
            return
        except (AttributeError, OSError, TypeError):
            pass
    stream.write(data.decode("utf-8", errors="replace"))
    stream.flush()


def _run_command(
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str] | None = None,
    label: str | None = None,
    heartbeat_sec: float = 30.0,
) -> CommandObservation:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    display_label = label or "command"
    _write_console(sys.stdout, f"\n[{display_label}] started; live output follows\n".encode("utf-8"))
    with stdout_path.open("ab") as stdout_handle, stderr_path.open("ab") as stderr_handle:
        stdout_handle.write(f"\n$ {command_line(command)}\n".encode("utf-8", errors="replace"))
        stdout_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=merged_env,
        )
        console_lock = threading.Lock()
        output_lock = threading.Lock()
        stop_heartbeat = threading.Event()
        last_output = time.monotonic()

        def pump(pipe: Any, log_handle: Any, console: Any) -> None:
            nonlocal last_output
            try:
                while True:
                    chunk = pipe.read(4096)
                    if not chunk:
                        return
                    log_handle.write(chunk)
                    log_handle.flush()
                    with output_lock:
                        last_output = time.monotonic()
                    with console_lock:
                        _write_console(console, chunk)
            finally:
                pipe.close()

        pumps = [
            threading.Thread(
                target=pump,
                args=(process.stdout, stdout_handle, sys.stdout),
                name=f"{display_label}-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=pump,
                args=(process.stderr, stderr_handle, sys.stderr),
                name=f"{display_label}-stderr",
                daemon=True,
            ),
        ]
        for pump_thread in pumps:
            pump_thread.start()

        def heartbeat() -> None:
            if heartbeat_sec <= 0:
                return
            last_notice = 0.0
            while not stop_heartbeat.wait(min(1.0, heartbeat_sec)):
                with output_lock:
                    silent_for = time.monotonic() - last_output
                now = time.monotonic()
                if silent_for >= heartbeat_sec and now - last_notice >= heartbeat_sec:
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(silent_for))
                    with console_lock:
                        _write_console(
                            sys.stdout,
                            f"\n[{display_label}] still running; no output for {elapsed}\n".encode("utf-8"),
                        )
                    last_notice = now

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"{display_label}-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            observation = observe_process(process)
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=2.0)
            for pump_thread in pumps:
                pump_thread.join(timeout=2.0)
        return observation


def _is_render_command(command: list[str]) -> bool:
    names = {Path(token).name.lower() for token in command}
    return "render.py" in names and not any("common/metrics.py" in token.replace("\\", "/") for token in command)


def _telemetry_payload(telemetry: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    payload = dict(telemetry)
    model = collect_model_artifacts(output_dir / "train")
    model_file = model.get("model_file")
    if model_file:
        try:
            model["model_file"] = str(Path(model_file).resolve().relative_to(output_dir.resolve()))
        except ValueError:
            model["model_file"] = str(model_file)
    payload.update(model)
    return payload


def _write_metrics(output_dir: Path, telemetry: dict[str, Any]) -> dict[str, Any]:
    existing = _read_json(output_dir / "test" / "test_metrics.json") or {}
    merged = dict(existing)
    merged.update(_telemetry_payload(telemetry, output_dir))
    atomic_write_json(output_dir / "metrics.json", merged)
    return merged


def build_context(
    root: Path,
    experiment: ExperimentDefinition,
    run_dir: Path,
    layout: DatasetLayout,
    iterations: int,
) -> dict[str, Any]:
    experiment_dir = root / experiment.directory
    output_dir = run_dir / experiment.directory
    repo = experiment_dir / experiment.source_repo_dir
    return {
        "root": root,
        "experiment": experiment.directory,
        "data_root": layout.root,
        "train_data": layout.train,
        "test_data": layout.test,
        "train_masks": layout.train_masks,
        "test_masks": layout.test_masks,
        "output": output_dir,
        "train_output": output_dir / "train",
        "test_output": output_dir / "test",
        "experiment_dir": experiment_dir,
        "repo": repo,
        "iterations": iterations,
        "dataset_compat": dataset_compatibility_mode(root, experiment.directory),
        "dataset_compat_cache_root": run_dir.parent / ".dataset_compat",
    }


def _expanded_stage(stage: StageDefinition, context: dict[str, Any]) -> list[list[str]]:
    return [expand_command(command, context) for command in stage.commands]


def run_experiment(
    experiment: ExperimentDefinition,
    context: dict[str, Any],
    mode: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_dir = Path(context["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if (
        not context.get("_dataset_context_prepared")
        or context.get("_dataset_context_dry_run") != dry_run
    ):
        try:
            context = prepare_context_dataset(context, dry_run=dry_run)
        except DatasetCompatibilityError as error:
            return _status(
                output_dir,
                status="blocked",
                error=str(error),
                finished_at=now_iso(),
                exit_code=2,
                dataset_compat=context.get("dataset_compat", "raw"),
                dataset_root=str(context.get("data_root", "")),
            )
    required_artifacts = ["train"] if mode == "train" else ["test"] if mode == "test" else ["train", "test"]
    all_commands: list[list[str]] = []
    stages: list[tuple[str, StageDefinition]] = []
    if mode in ("train", "both"):
        stages.append(("train", experiment.train))
    if mode in ("test", "both"):
        stages.append(("test", experiment.test))
    for _, stage in stages:
        all_commands.extend(_expanded_stage(stage, context))
    write_commands(output_dir / "command.txt", all_commands)
    if should_skip_experiment(output_dir):
        current = _read_json(output_dir / "status.json") or {"status": "completed"}
        current["reason"] = "completed artifacts already exist"
        return current
    base = {
        "experiment_id": experiment.id,
        "experiment": experiment.directory,
        "name": experiment.name,
        "mode": mode,
        "mask_strategy": experiment.mask_strategy,
        "dataset_compat": context.get("dataset_compat", "raw"),
        "dataset_root": str(context["data_root"]),
        "dataset_compat_cache_dir": str(context["dataset_compat_cache_dir"])
        if context.get("dataset_compat_cache_dir")
        else None,
        "dataset_compat_cache_hit": context.get("dataset_compat_cache_hit"),
        "source_url": experiment.source_url,
        "artifacts": required_artifacts,
        "started_at": now_iso(),
        "exit_code": None,
    }
    atomic_write_json(output_dir / "status.json", base)
    if dry_run:
        return _status(output_dir, status="planned", finished_at=now_iso())
    telemetry: dict[str, Any] = {
        "training_time_sec": None,
        "avg_iteration_time_sec": None,
        "render_time_sec": None,
        "rendered_frames": None,
        "render_fps": None,
        "peak_gpu_memory_mb": None,
        "peak_gpu_memory_reason": None,
    }
    iteration_target = int(context["iterations"]) if any(
        "--iterations" in command for command in experiment.train.commands
    ) else None
    repo = Path(context["repo"])
    if not repo.is_dir():
        return _status(output_dir, status="blocked", error=f"Missing source repository: {repo}", finished_at=now_iso(), exit_code=127)
    for stage_name, stage in stages:
        stage_started = time.perf_counter()
        render_snapshot = snapshot_render_images(output_dir) if stage_name == "test" else {}
        stage_commands = _expanded_stage(stage, context)
        cwd_value = _stage_context(context, stage, stage_name)["cwd"]
        cwd = Path(cwd_value)
        if not cwd.is_dir():
            return _status(output_dir, status="blocked", error=f"Missing stage cwd: {cwd}", failed_stage=stage_name, finished_at=now_iso(), exit_code=127)
        for command in stage_commands:
            missing = _script_missing(command)
            if missing:
                return _status(output_dir, status="blocked", error=f"Missing command file: {missing}", failed_stage=stage_name, finished_at=now_iso(), exit_code=127)
            _status(output_dir, status="running", current_stage=stage_name)
        atomic_write_json(output_dir / "stage_status.json", {"stage": stage_name, "status": "running", "updated_at": now_iso()})
        stdout_path = output_dir / f"{stage_name}_stdout.log"
        stderr_path = output_dir / f"{stage_name}_stderr.log"
        for command in stage_commands:
            observation = _run_command(
                command,
                cwd,
                stdout_path,
                stderr_path,
                label=f"{experiment.id} {experiment.name} / {stage_name}",
            )
            if observation.peak_gpu_memory_mb is not None:
                current_peak = telemetry["peak_gpu_memory_mb"]
                telemetry["peak_gpu_memory_mb"] = (
                    observation.peak_gpu_memory_mb
                    if current_peak is None
                    else max(current_peak, observation.peak_gpu_memory_mb)
                )
                telemetry["peak_gpu_memory_reason"] = None
            elif telemetry["peak_gpu_memory_reason"] is None:
                telemetry["peak_gpu_memory_reason"] = observation.peak_gpu_memory_reason
            if stage_name == "test" and _is_render_command(command):
                rendered_frames = count_changed_render_images(output_dir, render_snapshot)
                telemetry["render_time_sec"] = observation.duration_sec
                telemetry["rendered_frames"] = rendered_frames
                if rendered_frames > 0 and observation.duration_sec > 0:
                    telemetry["render_fps"] = rendered_frames / observation.duration_sec
            if observation.exit_code != 0:
                if stage_name == "train":
                    telemetry["training_time_sec"] = time.perf_counter() - stage_started
                    telemetry["avg_iteration_time_sec"] = average_iteration_time(
                        telemetry["training_time_sec"], iteration_target
                    )
                _write_metrics(output_dir, telemetry)
                return _status(
                    output_dir,
                    status="failed",
                    failed_stage=stage_name,
                    finished_at=now_iso(),
                    exit_code=observation.exit_code,
                    error=f"Command failed with exit code {observation.exit_code}",
                    stderr_tail=_tail_text(stderr_path),
                    **_telemetry_payload(telemetry, output_dir),
                )
        if stage_name == "train":
            telemetry["training_time_sec"] = time.perf_counter() - stage_started
            telemetry["avg_iteration_time_sec"] = average_iteration_time(
                telemetry["training_time_sec"], iteration_target
            )
        stage_artifacts = stage.artifacts or ([stage_name] if stage_name in required_artifacts else [])
        missing_artifacts = [
            artifact for artifact in stage_artifacts if not _artifact_ready(output_dir, artifact)
        ]
        if missing_artifacts:
            _write_metrics(output_dir, telemetry)
            atomic_write_json(
                output_dir / "stage_status.json",
                {
                    "stage": stage_name,
                    "status": "failed",
                    "updated_at": now_iso(),
                    "missing_artifacts": missing_artifacts,
                },
            )
            return _status(
                output_dir,
                status="failed",
                failed_stage=stage_name,
                finished_at=now_iso(),
                exit_code=1,
                error=f"Missing expected artifact(s): {', '.join(missing_artifacts)}",
                **_telemetry_payload(telemetry, output_dir),
            )
        atomic_write_json(output_dir / "stage_status.json", {"stage": stage_name, "status": "completed", "updated_at": now_iso()})
    metrics = _write_metrics(output_dir, telemetry)
    completion = dict(metrics)
    completion.update(status="completed", finished_at=now_iso(), exit_code=0)
    return _status(output_dir, **completion)
