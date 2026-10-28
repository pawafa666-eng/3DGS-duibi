"""Read each bundled rasterizer's Python interface without importing CUDA."""
import ast
import hashlib
from pathlib import Path

from .config import discover_experiments


def rasterizer_contract(method_id):
    root = Path(__file__).resolve().parents[1]
    definition = next(d for d in discover_experiments(root) if d.id == str(method_id).zfill(2))
    repo = root / definition.directory / definition.source_repo_dir
    base = repo / ("gaussian_splatting/submodules" if definition.id == "04" else "submodules")
    package = "diff_surfel_rasterization" if definition.id == "02" else "diff_gaussian_rasterization"
    directory = "diff-surfel-rasterization" if definition.id == "02" else "diff-gaussian-rasterization"
    source = base / directory / package / "__init__.py"
    raw = source.read_bytes().replace(b"\r\n", b"\n")
    tree = ast.parse(raw.decode("utf-8-sig"))
    settings = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "GaussianRasterizationSettings")
    fields = [node.target.id for node in settings.body if isinstance(node, ast.AnnAssign)]
    return {"package": package, "fields": fields, "source_sha256": hashlib.sha256(raw).hexdigest()}
