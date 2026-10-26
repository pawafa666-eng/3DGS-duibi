"""Deserialize fresh Scaffold/Octree network saves with the training Python."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def required_network_files(model_dir):
    path = Path(model_dir) / "cfg_args"
    call = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "Namespace":
        raise ValueError("Invalid saved model configuration")
    config = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
    names = ["opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"]
    if config.get("use_feat_bank", False):
        names.append("feature_bank_mlp.pt")
    if config.get("appearance_dim", 32) > 0:
        names.append("embedding_appearance.pt")
    return names


def verify(model_dir, method_id, iterations):
    import torch
    model_dir = Path(model_dir)
    directory = model_dir / f"point_cloud/iteration_{iterations}"
    loaded = []
    for name in required_network_files(model_dir):
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty MLP checkpoint: {path}")
        network = torch.jit.load(str(path), map_location="cpu")
        parameters = list(network.parameters())
        if not parameters or any(not torch.isfinite(value).all() for value in parameters):
            raise ValueError(f"Empty or non-finite saved network: {name}")
        loaded.append(str(path))
    checkpoint = model_dir / f"chkpnt{iterations}.pth"
    # These are outputs of this fresh owned run, including NumPy metadata.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(saved, tuple) or len(saved) != 2 or saved[1] != iterations:
        raise ValueError("Checkpoint iteration differs from the requested final iteration")
    state = saved[0]
    algorithm = {"06": "Scaffold-GS", "08": "Octree-GS"}[method_id]
    if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("algorithm") != algorithm:
        raise ValueError("Incomplete or incompatible anchor checkpoint schema")
    def finite(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return True
    if not finite(state):
        raise ValueError("Non-finite parameters, statistics, networks or optimizer state")
    if set(state.get("parameters", {})) != {"_anchor", "_offset", "_anchor_feat", "_scaling", "_rotation", "_opacity"}:
        raise ValueError("Incomplete anchor checkpoint parameters")
    steps = [float(item["step"]) for item in state.get("optimizer", {}).get("state", {}).values() if "step" in item]
    if not steps or max(steps) != iterations - 1:
        raise ValueError("Optimizer state does not corroborate the upstream short-training updates")
    return {"mlp_files_loaded": loaded, "checkpoint_loaded": str(checkpoint),
            "checkpoint_iteration": saved[1], "optimizer_max_step": max(steps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--method-id", choices=("06", "08"), required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.model_dir, args.method_id, args.iterations)))


if __name__ == "__main__":
    main()
