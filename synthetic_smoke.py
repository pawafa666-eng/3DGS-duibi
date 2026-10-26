"""Train each real algorithm for a bounded number of iterations on synthetic data.

Run from the desired CUDA environment: python synthetic_smoke.py --only 01
No real data is read. Each invocation owns a fresh directory under output/.
Dependencies for this controller: numpy, Pillow, plyfile. Training dependencies
and CUDA extensions must be installed in each selected algorithm environment.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image

from common.config import discover_experiments
from common.runner import atomic_write_json, build_context, expand_command
from common.dataset import dataset_layout
from common.runtime_env import validate_native_modules

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 256  # Surfels starts at 1/4 scale; LPIPS needs at least 64 px.
SCOPE = ("A short synthetic-input training/save check only; does not establish "
         "30000-iteration stability, real-scene memory capacity, densification, "
         "late GaussianPro propagation, mesh quality, or research accuracy.")


def _pose(angle: float):
    center = np.array([4 * np.cos(angle), 4 * np.sin(angle), 1.4])
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    # Matrix to COLMAP quaternion (w,x,y,z), using a symmetric eigenproblem.
    m = rotation
    K = np.array([
        [m[0, 0]-m[1, 1]-m[2, 2], m[0, 1]+m[1, 0], m[0, 2]+m[2, 0], m[2, 1]-m[1, 2]],
        [m[0, 1]+m[1, 0], m[1, 1]-m[0, 0]-m[2, 2], m[1, 2]+m[2, 1], m[0, 2]-m[2, 0]],
        [m[0, 2]+m[2, 0], m[1, 2]+m[2, 1], m[2, 2]-m[0, 0]-m[1, 1], m[1, 0]-m[0, 1]],
        [m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1], np.trace(m)],
    ]) / 3
    _, vectors = np.linalg.eigh(K)
    q = vectors[[3, 0, 1, 2], -1]
    q *= 1 if q[0] >= 0 else -1
    return center, rotation, q, -rotation @ center


def _colors(points, normals):
    palette = np.array([[0.85, .26, .20], [.20, .58, .88], [.30, .78, .43]])
    dominant = np.abs(normals).argmax(axis=-1)
    checker = (np.floor((points + 1) * 6).sum(axis=-1) % 2).astype(bool)
    rgb = palette[dominant] * np.where(checker, .62, 1.)[..., None]
    return np.uint8(np.clip(rgb * 255, 0, 255))


def _render(angle, size=IMAGE_SIZE):
    center, rotation, q, t = _pose(angle)
    yy, xx = np.mgrid[:size, :size]
    focal = size * .95
    camera_rays = np.stack([(xx+.5-size/2)/focal, (yy+.5-size/2)/focal, np.ones_like(xx)], -1)
    rays = camera_rays @ rotation
    safe = np.where(np.abs(rays) < 1e-9, 1e-9, rays)
    a, b = (-1-center)/safe, (1-center)/safe
    near, far = np.minimum(a, b).max(axis=-1), np.maximum(a, b).min(axis=-1)
    hit = (near > 0) & (far >= near)
    positions = center + near[..., None] * rays
    normals = np.zeros_like(positions)
    axes = np.abs(positions).argmax(axis=-1)
    np.put_along_axis(normals, axes[..., None], np.take_along_axis(np.sign(positions), axes[..., None], -1), -1)
    image = _colors(positions, normals)
    image[~hit] = [14, 18, 24]
    normals[~hit] = 0
    # Surfels' optional mono normal input is in [0,1], channels first.
    mono_normal = ((normals @ rotation.T + 1) / 2).transpose(2, 0, 1).astype('f4')
    depth = np.where(hit, near, 0).astype('f4')
    return image, np.uint8(hit)*255, mono_normal, depth, q, t, focal


def generate_dataset(target: Path) -> dict[str, Path]:
    """Generate an independent, consistent cube scene in both point formats."""
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {target}")
    target.mkdir(parents=True)
    rng = np.random.default_rng(42)
    points = rng.uniform(-1, 1, size=(6000, 3))
    normals = np.zeros_like(points)
    axes = np.arange(6000) % 3
    signs = np.where(np.arange(6000) % 2, -1., 1.)
    points[np.arange(6000), axes] = signs
    normals[np.arange(6000), axes] = signs
    colors = _colors(points, normals)
    variants = {key: target / key for key in ("standard", "custom")}
    for variant, base in variants.items():
        for split, count, phase in (("train", 12, 0.), ("test", 4, .17)):
            root = base / split
            for directory in ("images", "masks", "normal", "depth", "sparse/0"):
                (root / directory).mkdir(parents=True)
            sparse = root / "sparse/0"
            with (sparse / "cameras.bin").open("wb") as handle:
                s = IMAGE_SIZE
                handle.write(struct.pack("<QiiQQdddd", 1, 1, 1, s, s, s*.95, s*.95, s/2, s/2))
            with (sparse / "images.bin").open("wb") as handle:
                handle.write(struct.pack("<Q", count))
                for i in range(count):
                    rgb, mask, mono, depth, q, t, _ = _render(phase + 2*np.pi*i/count)
                    name = f"view_{i:03d}.png"
                    Image.fromarray(rgb).save(root / "images" / name)
                    Image.fromarray(mask).save(root / "masks" / f"view_{i:03d}.mask.png")
                    np.save(root / "normal" / f"view_{i:03d}_normal.npy", mono)
                    np.save(root / "depth" / f"view_{i:03d}_depth.npy", depth)
                    handle.write(struct.pack("<i7di", i+1, *q, *t, 1))
                    handle.write(name.encode("ascii") + b"\0")
                    handle.write(struct.pack("<Q", 0))  # No 2D observations needed for GS loaders.
            with (sparse / "points3D.bin").open("wb") as handle:
                handle.write(struct.pack("<Q", len(points)))
                for i, (xyz, rgb, normal) in enumerate(zip(points, colors, normals)):
                    if variant == "custom":
                        handle.write(struct.pack("<QdddBBBddddQ", i+1, *xyz, *rgb, *normal, 0., 0))
                    else:
                        handle.write(struct.pack("<QdddBBBdQ", i+1, *xyz, *rgb, 0., 0))
            (sparse / "is_ground.txt").write_text("\n".join(str(int(z < -.99)) for z in points[:, 2])+"\n", encoding="ascii")
    atomic_write_json(target / "synthetic_manifest.json", {
        "synthetic": True, "geometry": "textured cube", "seed": 42,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE], "train_images": 12, "test_images": 4,
        "points": 6000, "mask_semantics": "255 valid foreground, 0 ignored background",
        "point_formats": {"standard": "COLMAP <QdddBBBdQ>", "custom": "normals <QdddBBBddddQ>"},
        "scope": SCOPE,
    })
    return variants


def _set_option(command, flag, values):
    if flag in command:
        index = command.index(flag)
        end = index+1
        while end < len(command) and not command[end].startswith("--"):
            # These flags occur after short -s/-m options in experiment configs.
            if command[end].startswith("-"):
                break
            end += 1
        command[index:end] = [flag, *map(str, values)]
    else:
        command.extend([flag, *map(str, values)])


def build_commands(definition, datasets, output, python, iterations):
    data = datasets["custom" if definition.id in ("01", "10") else "standard"]
    context = build_context(ROOT, definition, output.parent, dataset_layout(data), iterations)
    context.update(output=output, train_output=output / "train", test_output=output / "test")
    repo = Path(context["repo"])
    if definition.id == "04":
        return [[python, "-u", str(ROOT / "common/sugar_smoke.py"), "--repo", str(repo),
                 "--data", str(data / "train"), "--output", str(output / "train"),
                 "--iterations", str(iterations)]], repo
    commands = []
    for template in definition.train.commands:
        command = expand_command(template, context)
        command[0:1] = [python, "-u"]
        for flag in ("--iterations", "--save_iterations", "--checkpoint_iterations"):
            _set_option(command, flag, [iterations])
        _set_option(command, "--test_iterations", [-1])
        if definition.id in ("06", "08"):
            command.extend(["--train_only", "--skip_code_backup"])
        # Disable scheduled validation, but preserve training losses and optimizers.
        # Final model saving and checkpoints are mandatory and remain active.
        commands.append(command)
    return commands, repo


def child_environment(python, log_root):
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", WANDB_MODE="offline")
    env.setdefault("MPLBACKEND", "Agg")
    # Some isolated shells omit USERNAME; torchvision imports torch._dynamo,
    # which otherwise fails before training while resolving its cache folder.
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(log_root / ".torch_inductor"))
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    if sys.platform == "linux":
        cuda = Path(env.get("CUDA_HOME", "/usr/local/cuda-12.8"))
        if (cuda / "bin/nvcc").is_file():
            env.setdefault("CUDA_HOME", str(cuda))
            env["PATH"] = str(cuda / "bin") + os.pathsep + env["PATH"]
        prefix = Path(python).resolve().parent
        cc, cxx = prefix / "x86_64-conda-linux-gnu-cc", prefix / "x86_64-conda-linux-gnu-c++"
        if cc.is_file() and cxx.is_file():
            env.setdefault("CC", str(cc))
            env.setdefault("CXX", str(cxx))
            env.setdefault("CUDAHOSTCXX", str(cxx))
        env.setdefault("MAX_JOBS", "1")
    return env


def _stop_tree(process):
    if os.name == "nt":
        if process.poll() is not None:
            return
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        # start_new_session made this an owned group. Its leader may already
        # have exited while a worker still holds stdout or the GPU open.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_command(command, cwd, log_path, timeout, extra_env=None):
    """Real subprocess execution; a progress banner never counts as success."""
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=100)
    messages = queue.Queue()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        env = child_environment(command[0], log_path.parent)
        env.update(extra_env or {})
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=env, bufsize=0,
                                   start_new_session=(os.name != "nt"))
        def pump():
            try:
                while chunk := process.stdout.read(1024):
                    messages.put(chunk)
            finally:
                process.stdout.close()
                messages.put(None)
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        last_notice = started
        try:
            while True:
                if time.monotonic()-started >= timeout and not timed_out:
                    timed_out = True
                    _stop_tree(process)
                try:
                    chunk = messages.get(timeout=.2)
                except queue.Empty:
                    if time.monotonic()-last_notice > 30:
                        print(f"  still running ({time.monotonic()-started:.0f}s); log: {log_path.name}", flush=True)
                        last_notice = time.monotonic()
                    continue
                if chunk is None:
                    break
                line = chunk.decode("utf-8", errors="replace")
                log.write(line)
                log.flush()
                tail.append(line)
                print(line, end="", flush=True)
            exit_code = process.wait(timeout=5)
        finally:
            _stop_tree(process)
            reader.join(timeout=5)
    return {"exit_code": exit_code, "timed_out": timed_out, "seconds": time.monotonic()-started,
            "tail": "".join(tail)[-6000:], "log": str(log_path), "command": command}


def verify_model(output, method_id, iterations, python=None):
    from plyfile import PlyData
    directory = Path(output) / f"point_cloud/iteration_{iterations}"
    target = directory / "point_cloud.ply"
    if not target.is_file():
        raise ValueError(f"Missing iteration-{iterations} model: {target}")
    try:
        ply = PlyData.read(target)
        # Reduced-3DGS stores separate SH degree groups, including empty ones.
        groups = ([ply[f"vertex_{degree}"].data for degree in range(4)]
                  if method_id == "09" else [ply["vertex"].data])
        count = sum(len(vertices) for vertices in groups)
        if count == 0:
            raise ValueError("Empty point cloud")
        for vertices in groups:
            fields = set(vertices.dtype.names)
            if not {"x", "y", "z", "opacity"} <= fields:
                raise ValueError("Missing Gaussian fields")
            if not any(name.startswith("f_") for name in fields):
                raise ValueError("Missing trained Gaussian appearance/anchor features")
            for name in fields:
                if np.issubdtype(vertices[name].dtype, np.number) and not np.isfinite(vertices[name]).all():
                    raise ValueError(f"Model parameter {name} must be finite")
    except Exception as error:
        raise ValueError(f"Invalid saved model {target}: {error}") from error
    result = {"model": str(target), "vertices": count, "bytes": target.stat().st_size}
    if method_id in ("06", "08"):
        for name in ("opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"):
            path = directory / name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing or empty MLP checkpoint: {path}")
        python = python or sys.executable
        command = [python, str(ROOT / "common/check_anchor_save.py"), "--model-dir", str(output),
                   "--method-id", method_id, "--iterations", str(iterations)]
        check = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env=child_environment(python, Path(output)), timeout=45)
        if check.returncode:
            raise ValueError(f"Saved MLP/checkpoint cannot be loaded: {check.stderr[-2500:]}")
        result.update(json.loads(check.stdout.splitlines()[-1]))
    return result


def environment_probe(python, run_dir, extra_env=None):
    code = """import json, sys, importlib.util, shutil
result={'python':sys.executable,'version':sys.version,'nvcc':shutil.which('nvcc'),'cl':shutil.which('cl')}
result['modules']={name:importlib.util.find_spec(name) is not None for name in ['torch','torchvision','diff_gaussian_rasterization','diff_surfel_rasterization','simple_knn','pytorch3d','torch_scatter','open3d','wandb','lpips']}
try:
 import torch
 result.update(torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available())
 if torch.cuda.is_available():
  x=torch.ones(8,device='cuda'); y=(x*x).sum(); torch.cuda.synchronize()
  result.update(gpu=torch.cuda.get_device_name(0),gpu_arch=list(torch.cuda.get_device_capability(0)),cuda_kernel_value=y.item())
except Exception as error: result['cuda_error']=repr(error)
print(json.dumps(result))
"""
    try:
        env = child_environment(python, run_dir)
        env.update(extra_env or {})
        result = subprocess.run([python, "-c", code], capture_output=True, text=True, encoding="utf-8",
                                errors="replace", env=env, timeout=45)
        return json.loads(result.stdout.splitlines()[-1])
    except Exception as error:
        return {"python": python, "probe_error": str(error)}


def write_report(run_dir, results, iterations):
    report = {"created_at": datetime.now().astimezone().isoformat(), "synthetic": True,
              "iterations": iterations, "scope": SCOPE, "results": results}
    atomic_write_json(run_dir / "report.json", report)
    rows = ["# 合成输入短训练检查", "", f"目标：{iterations} iterations + 保存并读取真实模型。", "",
            "只验证短训练链路，不能保证30000次迭代或真实大场景训练成功。", "",
            "| 算法 | 状态 | 原因 |", "|---|---|---|"]
    for result in results:
        error = result.get("error", "").replace("|", "/").replace("\n", " ")
        rows.append(f"| {result['id']} {result['name']} | {result['status']} | {error} |")
    (run_dir / "REPORT.md").write_text("\n".join(rows)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=[f"{i:02d}" for i in range(1, 11)])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180, help="Seconds per algorithm command")
    parser.add_argument("--python", default=sys.executable, help="Training Python (current environment by default)")
    parser.add_argument("--env-map", type=Path, help='JSON mapping IDs to Python paths, e.g. {"02":"C:/env/python.exe"}')
    parser.add_argument("--extension-root", type=Path, help="Per-algorithm CUDA packages, with subfolders 01..10")
    parser.add_argument("--output", type=Path, default=ROOT / "output/synthetic_smoke")
    parser.add_argument("--prepare-only", action="store_true", help="Generate inputs and commands; NOT a training pass")
    args = parser.parse_args(argv)
    if args.iterations <= 0 or args.timeout <= 0:
        parser.error("iterations and timeout must be positive")
    selectors = {str(value).zfill(2) for value in args.only}
    definitions = discover_experiments(ROOT)
    if selectors - {item.id for item in definitions}:
        parser.error("Unknown algorithm IDs: " + ", ".join(sorted(selectors)))
    overrides = json.loads(args.env_map.read_text(encoding="utf-8-sig")) if args.env_map else {}
    run_dir = args.output.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Run directory: {run_dir}", flush=True)
    datasets = generate_dataset(run_dir / "dataset")
    results, probes = [], {}
    try:
        for definition in definitions:
            if definition.id not in selectors:
                continue
            python = overrides.get(definition.id, args.python)
            python = shutil.which(python) or python
            output = run_dir / definition.directory
            output.mkdir()
            commands, cwd = build_commands(definition, datasets, output, python, args.iterations)
            atomic_write_json(output / "commands.json", {"cwd": str(cwd), "commands": commands})
            result = {"id": definition.id, "name": definition.name, "status": "planned",
                      "python": python, "output": str(output), "mask_strategy": definition.mask_strategy}
            extra_env = {}
            if args.extension_root:
                overlay = str(args.extension_root.resolve() / definition.id)
                extra_env["PYTHONPATH"] = overlay + os.pathsep + os.environ.get("PYTHONPATH", "")
                result["extension_path"] = overlay
            print(f"\n[{definition.id}] {definition.name}: {args.iterations} iterations + verify saved model", flush=True)
            try:
                if not args.prepare_only:
                    if args.extension_root:
                        result["native_imports"] = validate_native_modules(
                            python, args.extension_root.resolve() / definition.id, definition.id,
                            env={**child_environment(python, output), **extra_env})
                    probe_key = python + "|" + extra_env.get("PYTHONPATH", "")
                    if probe_key not in probes:
                        probes[probe_key] = environment_probe(python, run_dir, extra_env)
                        atomic_write_json(run_dir / "environment.json", probes)
                    for index, command in enumerate(commands, 1):
                        execution = run_command(command, cwd, output / f"train_{index}.log", args.timeout, extra_env)
                        result.setdefault("executions", []).append(execution)
                        if execution["timed_out"]:
                            raise RuntimeError(f"Timed out after {args.timeout}s; inspect train_{index}.log")
                        if execution["exit_code"] != 0:
                            last = [s.strip() for s in execution["tail"].splitlines() if s.strip()]
                            raise RuntimeError(last[-1] if last else f"Exit code {execution['exit_code']}")
                    if definition.id == "04":
                        summary = json.loads((output / "train/sugar_smoke_report.json").read_text(encoding="utf-8"))
                        if summary.get("status") != "passed":
                            raise ValueError("SuGaR full short pipeline not verified; see sugar report")
                        result["model_validation"] = summary
                        if not summary.get("full_pipeline_verified", False):
                            result["scope"] = "SuGaR coarse training only; mesh/refine not verified"
                    else:
                        result["model_validation"] = verify_model(output / "train", definition.id, args.iterations, python)
                    result["status"] = "partial_coarse" if definition.id == "04" and not summary.get("full_pipeline_verified", False) else "passed"
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                result.update(status="failed", error=str(error))
            atomic_write_json(output / "result.json", result)
            results.append(result)
            write_report(run_dir, results, args.iterations)
            print(f"[{definition.id}] {result['status']}: {result.get('error', 'see result.json')}", flush=True)
    except KeyboardInterrupt:
        write_report(run_dir, results, args.iterations)
        print("Interrupted; completed results retained.", flush=True)
        return 130
    passed = sum(item["status"] == "passed" for item in results)
    partial = sum(item["status"].startswith("partial_") for item in results)
    failed = sum(item["status"] == "failed" for item in results)
    print(f"\nTraining + save: {passed} passed, {partial} partial, {failed} failed "
          f"({len(results)} selected). Report: {run_dir / 'REPORT.md'}", flush=True)
    return 0 if args.prepare_only or passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
