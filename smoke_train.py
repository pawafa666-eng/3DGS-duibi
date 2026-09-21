"""Start each algorithm until real training progress appears, then stop it."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
DEFAULT_METHODS = [f"{number:02d}" for number in range(1, 11)]
TRAINING_PATTERNS = (
    re.compile(r"training progress.*(?:\d+%|\d+\s*/\s*\d+)", re.IGNORECASE),
    re.compile(r"\[\s*iter(?:ation)?\s+\d+\s*\]", re.IGNORECASE),
    re.compile(r"\b(?:iteration|iter)\s*[:#=]\s*\d+", re.IGNORECASE),
)


def detect_training_started(line: str) -> bool:
    return any(pattern.search(line) for pattern in TRAINING_PATTERNS)


def build_main_command(output: Path, method_id: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "main.py",
        "--mode",
        "train",
        "--only",
        method_id,
        "--continue-on-error",
        "--output",
        str(output),
    ]


def _terminate_process_tree(process: subprocess.Popen[str], wait_seconds: float) -> bool:
    if process.poll() is not None:
        return False
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except ProcessLookupError:
            return False
        try:
            process.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    return True


def run_smoke_command(
    command: list[str],
    cwd: Path,
    timeout: float,
    terminate_wait: float,
    log_path: Path | None = None,
) -> dict:
    started_at = time.monotonic()
    output_queue: queue.Queue[str | None] = queue.Queue()
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
        start_new_session=(os.name != "nt"),
    )

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line)
        output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")

    started = False
    terminated = False
    lines = []
    try:
        while True:
            elapsed = time.monotonic() - started_at
            if elapsed >= timeout:
                break
            try:
                line = output_queue.get(timeout=min(0.5, timeout - elapsed))
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if line is None:
                break
            lines.append(line)
            if log_handle is not None:
                log_handle.write(line)
                log_handle.flush()
            print(line, end="", flush=True)
            if detect_training_started(line):
                started = True
                break
    finally:
        if started:
            terminated = _terminate_process_tree(process, terminate_wait)
        elif process.poll() is None:
            _terminate_process_tree(process, terminate_wait)
        if log_handle is not None:
            log_handle.close()
        reader.join(timeout=terminate_wait)

    if process.poll() is None:
        process.kill()
        process.wait()
    duration = time.monotonic() - started_at
    return {
        "started": started,
        "terminated": terminated,
        "exit_code": process.returncode,
        "duration_sec": duration,
        "output_tail": "".join(lines)[-4000:],
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test algorithms: enter training, stop immediately, continue."
    )
    parser.add_argument("--only", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output", default="output/smoke")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--terminate-wait", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report_root = (ROOT / args.output).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    results = []
    for method_id in args.only:
        method_id = str(method_id).zfill(2)
        print(f"\n[{method_id}] Starting smoke training", flush=True)
        result = run_smoke_command(
            build_main_command(Path(args.output), method_id),
            ROOT,
            args.timeout,
            args.terminate_wait,
            report_root / f"{method_id}.log",
        )
        status = "started" if result["started"] else "failed"
        result.update({"id": method_id, "status": status})
        results.append(result)
        print(f"[{method_id}] {status}; moving to next algorithm", flush=True)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "training-smoke",
        "results": results,
    }
    report_path = report_root / "smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [item["id"] for item in results if item["status"] == "failed"]
    print(f"\nSmoke report: {report_path}")
    if failed:
        print("No training progress detected: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
