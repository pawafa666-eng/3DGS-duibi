"""Find an installed nvcc compatible with the active PyTorch CUDA build."""
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


class CudaToolkitError(RuntimeError):
    pass


@dataclass(frozen=True)
class CudaToolkit:
    home: Path
    version: str


def nvcc_version(binary):
    result = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=15)
    match = re.search(r"release\s+(\d+\.\d+)", result.stdout + result.stderr)
    if result.returncode or not match:
        raise CudaToolkitError(f"Cannot run nvcc: {binary}\n{result.stderr[-1000:]}")
    return match.group(1)


def find_cuda_toolkit(requested=None, *, torch_cuda=None, environ=None, prefix=None,
                      search_dirs=None, which=None, version_probe=None):
    env = os.environ if environ is None else environ
    which = which or shutil.which
    version_probe = version_probe or nvcc_version
    prefix = Path(prefix or sys.prefix)
    candidates = []
    if requested is not None:
        candidates.append(Path(requested))
    else:
        candidates.extend(Path(env[key]) for key in ("CUDA_HOME", "CUDA_PATH") if env.get(key))
        located = which("nvcc", path=env.get("PATH", ""))
        if located:
            candidates.append(Path(located).resolve().parent.parent)
        candidates.append(prefix)
        candidates.extend(search_dirs if search_dirs is not None else
                          [Path("/usr/local/cuda"), *sorted(Path("/usr/local").glob("cuda-*"))])
    found, seen, issues = [], set(), []
    for candidate in candidates:
        home = Path(candidate).expanduser().resolve()
        if home in seen:
            continue
        seen.add(home)
        binary = home / "bin/nvcc"
        if not binary.is_file():
            continue
        try:
            found.append(CudaToolkit(home, version_probe(binary)))
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            issues.append(str(error))
    if not found:
        checked = ", ".join(str(path / "bin/nvcc") for path in seen)
        raise CudaToolkitError(f"No usable nvcc compiler found (PyTorch CUDA={torch_cuda}). "
                               f"Checked: {checked}. Set CUDA_HOME or use --cuda <actual toolkit directory>. "
                               + " ".join(issues))
    if torch_cuda:
        exact = [item for item in found if item.version == torch_cuda]
        if exact:
            return exact[0]
        compatible = [item for item in found if item.version.split('.')[0] == torch_cuda.split('.')[0]]
        if compatible:
            return compatible[0]
        installed = ", ".join(f"{item.home} ({item.version})" for item in found)
        raise CudaToolkitError(f"PyTorch CUDA={torch_cuda} and installed nvcc have different major versions: "
                               f"{installed}. Select a matching toolkit with --cuda.")
    return found[0]
