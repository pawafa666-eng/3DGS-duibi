from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GpuMemorySample:
    value_mb: float | None
    reason: str | None = None


@dataclass(frozen=True)
class CommandObservation:
    exit_code: int
    duration_sec: float
    peak_gpu_memory_mb: float | None
    peak_gpu_memory_reason: str | None


def average_iteration_time(total_seconds: float | None, iterations: int | None) -> float | None:
    if total_seconds is None or iterations is None or iterations <= 0:
        return None
    return float(total_seconds) / int(iterations)


def _process_ids(root_pid: int) -> set[int]:
    ids = {int(root_pid)}
    try:
        import psutil  # type: ignore
    except ImportError:
        return ids
    try:
        ids.update(child.pid for child in psutil.Process(root_pid).children(recursive=True))
    except (psutil.Error, OSError, ValueError):
        return ids
    return ids


def sample_gpu_memory_mb(pid: int, executable: str = "nvidia-smi") -> GpuMemorySample:
    try:
        result = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except FileNotFoundError:
        return GpuMemorySample(None, "nvidia-smi not found")
    except subprocess.TimeoutExpired:
        return GpuMemorySample(None, "nvidia-smi timed out")
    except OSError as error:
        return GpuMemorySample(None, f"nvidia-smi unavailable: {error}")
    if result.returncode != 0:
        return GpuMemorySample(None, f"nvidia-smi failed with exit code {result.returncode}")

    records: list[tuple[int, float]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) != 2:
            continue
        try:
            record_pid = int(fields[0])
            memory = float(fields[1])
        except ValueError:
            continue
        records.append((record_pid, memory))
    if not records:
        return GpuMemorySample(None, "no compute process was visible")

    tracked = _process_ids(pid)
    selected = [memory for record_pid, memory in records if record_pid in tracked]
    if not selected:
        return GpuMemorySample(None, "experiment process was not visible in nvidia-smi")
    return GpuMemorySample(sum(selected), None)


class GpuMemoryMonitor:
    def __init__(self, pid: int, interval_sec: float = 0.5, executable: str = "nvidia-smi") -> None:
        self.pid = int(pid)
        self.interval_sec = max(float(interval_sec), 0.1)
        self.executable = executable
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb: float | None = None
        self.reason: str | None = None

    def _sample_once(self) -> None:
        sample = sample_gpu_memory_mb(self.pid, self.executable)
        if sample.value_mb is not None:
            self.peak_mb = sample.value_mb if self.peak_mb is None else max(self.peak_mb, sample.value_mb)
        elif self.reason is None:
            self.reason = sample.reason

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_sec)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._sample_once()


MODEL_SUFFIX_RANK = {".ply": 3, ".splat": 3, ".pt": 2, ".pth": 2, ".ckpt": 2}


def extract_ply_vertex_count(path: Path) -> int | None:
    try:
        header = Path(path).read_bytes()[:1_048_576].decode("ascii", errors="replace")
    except OSError:
        return None
    match = re.search(r"(?m)^element\s+vertex\s+(\d+)\s*$", header)
    return int(match.group(1)) if match else None


def _model_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in Path(root).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MODEL_SUFFIX_RANK:
            continue
        if path.name.lower().startswith("input"):
            continue
        candidates.append(path)
    return candidates


def collect_model_artifacts(root: Path) -> dict[str, Any]:
    candidates = _model_candidates(Path(root))
    if not candidates:
        return {
            "model_file": None,
            "model_size_bytes": None,
            "model_size_mb": None,
            "gaussian_count": None,
            "model_artifact_reason": "no model file found",
        }
    ply_candidates = [path for path in candidates if path.suffix.lower() == ".ply"]
    if ply_candidates:
        candidates = ply_candidates
    model_path = max(
        candidates,
        key=lambda path: (
            path.stat().st_mtime_ns,
            MODEL_SUFFIX_RANK[path.suffix.lower()],
            path.stat().st_size,
        ),
    )
    size_bytes = model_path.stat().st_size
    return {
        "model_file": str(model_path),
        "model_size_bytes": size_bytes,
        "model_size_mb": size_bytes / (1024 * 1024),
        "gaussian_count": extract_ply_vertex_count(model_path) if model_path.suffix.lower() == ".ply" else None,
        "model_artifact_reason": None,
    }


def count_render_images(root: Path) -> int:
    return sum(1 for path in Path(root).rglob("*.png") if path.is_file())


def snapshot_render_images(root: Path) -> dict[str, int]:
    snapshot: dict[str, int] = {}
    for path in Path(root).rglob("*.png"):
        if path.is_file():
            try:
                snapshot[str(path)] = path.stat().st_mtime_ns
            except OSError:
                continue
    return snapshot


def count_changed_render_images(root: Path, before: dict[str, int]) -> int:
    changed = 0
    for path in Path(root).rglob("*.png"):
        if not path.is_file():
            continue
        try:
            old_mtime = before.get(str(path))
            if old_mtime is None or path.stat().st_mtime_ns > old_mtime:
                changed += 1
        except OSError:
            continue
    return changed


def observe_process(process: Any, interval_sec: float = 0.5) -> CommandObservation:
    started = time.perf_counter()
    monitor = GpuMemoryMonitor(process.pid, interval_sec=interval_sec)
    monitor.start()
    exit_code = int(process.wait())
    monitor.stop()
    return CommandObservation(
        exit_code=exit_code,
        duration_sec=time.perf_counter() - started,
        peak_gpu_memory_mb=monitor.peak_mb,
        peak_gpu_memory_reason=monitor.reason,
    )
