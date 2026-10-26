"""Select compatible native packages separately for each algorithm process."""
from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys


def algorithm_environment(method_id, *, extension_root=None, environ=None, cache_dir=None):
    source = os.environ if environ is None else environ
    env = {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "MPLBACKEND": source.get("MPLBACKEND", "Agg")}
    # Child commands named 'python' must use the controller's environment.
    prefix = Path(sys.executable).resolve().parent
    env["PATH"] = str(prefix) + os.pathsep + source.get("PATH", "")
    if cache_dir is not None:
        env["TORCHINDUCTOR_CACHE_DIR"] = source.get("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
    selected = extension_root or source.get("GS_EXTENSION_ROOT")
    if selected is None and sys.platform == "linux":
        default = Path.home() / "3dgs-comparison-extensions"
        if default.is_dir():
            selected = default
    if selected is not None:
        overlay = Path(selected).expanduser().resolve() / str(method_id).zfill(2)
        if not overlay.is_dir():
            raise FileNotFoundError(f"Missing algorithm {method_id} CUDA packages: {overlay}. "
                                    "Run wsl_build_extensions.py first.")
        existing = source.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(overlay) + (os.pathsep + existing if existing else "")
    if sys.platform == "linux":
        cuda = Path(source.get("CUDA_HOME", "/usr/local/cuda-12.8"))
        if (cuda / "bin/nvcc").is_file():
            env["CUDA_HOME"] = str(cuda)
            env["PATH"] = str(cuda / "bin") + os.pathsep + env["PATH"]
        cc = prefix / "x86_64-conda-linux-gnu-cc"
        cxx = prefix / "x86_64-conda-linux-gnu-c++"
        if cc.is_file() and cxx.is_file():
            env.update(CC=source.get("CC", str(cc)), CXX=source.get("CXX", str(cxx)),
                       CUDAHOSTCXX=source.get("CUDAHOSTCXX", str(cxx)))
        env["MAX_JOBS"] = source.get("MAX_JOBS", "1")
    return env


def validate_native_modules(python, overlay, method_id, *, env=None, modules=None):
    """Import native libraries in the selected interpreter and verify origins."""
    overlay = Path(overlay).resolve()
    if not overlay.is_dir():
        raise RuntimeError(f"Missing CUDA extension overlay: {overlay}")
    if modules is None:
        raster = "diff_surfel_rasterization" if str(method_id).zfill(2) == "02" else "diff_gaussian_rasterization"
        modules = [raster + "._C", "simple_knn._C"]
        if str(method_id).zfill(2) == "07":
            modules.append("gaussianpro")
    code = """import importlib,json,sys
from pathlib import Path
import torch
root=Path(sys.argv[1]).resolve()
origins={}
for name in json.loads(sys.argv[2]):
 module=importlib.import_module(name)
 path=Path(module.__file__).resolve()
 if not path.is_relative_to(root):
  raise RuntimeError(f'{name} imported outside selected overlay: {path}')
 origins[name]=str(path)
print(json.dumps(origins))
"""
    merged = dict(os.environ)
    merged.update(env or {})
    existing = merged.get("PYTHONPATH", "")
    merged["PYTHONPATH"] = str(overlay) + (os.pathsep + existing if existing else "")
    result = subprocess.run([python, "-c", code, str(overlay), json.dumps(modules)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=merged, timeout=45)
    if result.returncode:
        raise RuntimeError("CUDA extension import verification failed: " + result.stderr[-2500:])
    return json.loads(result.stdout.splitlines()[-1])
