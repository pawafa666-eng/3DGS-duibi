"""Build each algorithm's actual CUDA extension into an isolated Linux folder.

Run INSIDE WSL using the project's Python environment. PyTorch must already be
installed. This does not install/replace GPU drivers or change system Python.
Uses separate import paths because rasterizers have incompatible APIs despite
sharing a package name. Compiler build outputs live on WSL's Linux filesystem.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from common.config import discover_experiments
from common.runtime_env import validate_native_modules

ROOT = Path(__file__).resolve().parent


def compiler_environment(cuda: Path, jobs: int):
    env = dict(os.environ)
    prefix = Path(sys.prefix)
    env.update(CUDA_HOME=str(cuda), MAX_JOBS=str(jobs),
               TORCH_CUDA_ARCH_LIST="12.0", PYTHONUNBUFFERED="1")
    # Prefer this conda environment's CUDA-compatible GCC over Ubuntu GCC 15.
    cc = prefix / "bin/x86_64-conda-linux-gnu-cc"
    cxx = prefix / "bin/x86_64-conda-linux-gnu-c++"
    if cc.exists() and cxx.exists():
        env.update(CC=str(cc), CXX=str(cxx), CUDAHOSTCXX=str(cxx))
    env["PATH"] = str(prefix / "bin") + os.pathsep + str(cuda / "bin") + os.pathsep + env.get("PATH", "")
    return env


def source_digest(source):
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*")):
        if not path.is_file() or any(p in {".git", "build", "dist", "__pycache__"} or p.endswith(".egg-info") for p in path.parts):
            continue
        if path.suffix not in {".py", ".cpp", ".cu", ".h", ".hpp", ".cuh", ".inl"}:
            continue
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:20]


def native_module(source):
    return {"diff-gaussian-rasterization": "diff_gaussian_rasterization._C",
            "diff-surfel-rasterization": "diff_surfel_rasterization._C",
            "simple-knn": "simple_knn._C", "Propagation": "gaussianpro"}[source.name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=[f"{i:02d}" for i in range(1, 11)])
    parser.add_argument("--extension-root", type=Path, default=Path.home() / "3dgs-comparison-extensions")
    parser.add_argument("--build-root", type=Path, default=Path.home() / "3dgs-comparison-builds")
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda-12.8"))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if sys.platform != "linux":
        parser.error("Use this script INSIDE WSL Ubuntu")
    import torch
    capability = torch.cuda.get_device_capability()
    env = compiler_environment(args.cuda, args.jobs)
    env["TORCH_CUDA_ARCH_LIST"] = f"{capability[0]}.{capability[1]}"
    args.extension_root.mkdir(parents=True, exist_ok=True)
    args.build_root.mkdir(parents=True, exist_ok=True)
    log_root = ROOT / "output/wsl_build_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root.mkdir(parents=True, exist_ok=False)
    results = []
    ids = {str(i).zfill(2) for i in args.only}
    for definition in discover_experiments(ROOT):
        if definition.id not in ids:
            continue
        repo = ROOT / definition.directory / definition.source_repo_dir
        submodules = repo / ("gaussian_splatting/submodules" if definition.id == "04" else "submodules")
        target = args.extension_root / definition.id
        target.mkdir(parents=True, exist_ok=True)
        sources = sorted(path for path in submodules.iterdir() if (path / "setup.py").is_file())
        if not sources:
            raise FileNotFoundError(f"No extension sources: {submodules}")
        for source in sources:
            digest = source_digest(source)
            legacy_stamp = digest + "-" + torch.__version__.replace("/", "_") + "-" + env["TORCH_CUDA_ARCH_LIST"]
            stamp = legacy_stamp + "-" + sys.implementation.cache_tag
            marker = target / ("." + source.name + "_built.json")
            reusable = False
            if marker.is_file():
                try:
                    if json.loads(marker.read_text()).get("stamp") in (stamp, legacy_stamp):
                        validate_native_modules(sys.executable, target, definition.id, env=env,
                                                modules=[native_module(source)])
                        reusable = True
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    print(f"[{definition.id}] cached extension needs rebuild: {error}", flush=True)
            if reusable:
                marker.write_text(json.dumps({"stamp": stamp, "source": str(source),
                                             "python": sys.executable, "abi": sys.implementation.cache_tag}))
                print(f"[{definition.id}] reuse {source.name}", flush=True)
                results.append({"id": definition.id, "package": source.name, "status": "reused"})
                continue
            build_source = args.build_root / definition.id / (source.name + "-" + digest)
            if not build_source.exists():
                shutil.copytree(source, build_source, ignore=shutil.ignore_patterns(".git", "build", "dist", "*.egg-info", "__pycache__", "*.pyd", "*.so"))
            command = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--upgrade", "--target", str(target), str(build_source)]
            log_path = log_root / f"{definition.id}_{source.name}.log"
            print(f"[{definition.id}] building {source.name}; {log_path}", flush=True)
            with log_path.open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            exit_code = result.returncode
            if exit_code == 0:
                try:
                    validate_native_modules(sys.executable, target, definition.id, env=env,
                                            modules=[native_module(source)])
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    with log_path.open("a") as log:
                        log.write(f"\nInstalled extension verification failed: {error}\n")
                    exit_code = 1
            record = {"id": definition.id, "package": source.name, "status": "built" if exit_code == 0 else "failed", "exit_code": exit_code, "log": str(log_path)}
            results.append(record)
            if exit_code == 0:
                marker.write_text(json.dumps({"stamp": stamp, "source": str(source), "python": sys.executable,
                                             "abi": sys.implementation.cache_tag}))
            else:
                print(log_path.read_text(errors="replace")[-10000:], flush=True)
            (log_root / "report.json").write_text(json.dumps(results, indent=2))
    (log_root / "report.json").write_text(json.dumps(results, indent=2))
    print(f"Build report: {log_root / 'report.json'}", flush=True)
    return int(any(item["status"] == "failed" for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
