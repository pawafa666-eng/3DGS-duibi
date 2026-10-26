"""Exercise upstream checkpoint methods with real CPU parameters and Adam.

Only the CUDA-only constructor/training_setup is replaced. The captured/restored
methods are extracted from the actual patched upstream source, then a second
optimization step must match uninterrupted training exactly.
"""
import ast
from copy import deepcopy
import io
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("06_Scaffold_GS", "08_Octree_GS")
PARAMETERS = ("_anchor", "_offset", "_anchor_feat", "_scaling", "_rotation", "_opacity")


@pytest.fixture(autouse=True)
def isolated_torch_cache(monkeypatch, tmp_path):
    # The host's isolated Windows shell can omit USERNAME; avoid getpass in
    # PyTorch's cache initialization without changing the user's environment.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "torch_cache"))


def model(method, feature_bank=True, appearance=2):
    source = ROOT / method / "runtime/scene/gaussian_model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianModel")
    methods = ast.Module(body=[node for node in cls.body if isinstance(node, ast.FunctionDef)
                              and node.name in ("capture", "restore")], type_ignores=[])
    namespace = {"torch": torch, "nn": nn, "deepcopy": deepcopy}
    exec(compile(methods, str(source), "exec"), namespace)
    instance = SimpleNamespace()
    for name in ("capture", "restore"):
        setattr(instance, name, MethodType(namespace[name], instance))
    for name in PARAMETERS:
        setattr(instance, name, nn.Parameter(torch.rand(4, 3), requires_grad=name not in ("_rotation", "_opacity")))
    for name in ("opacity_accum", "offset_gradient_accum", "offset_denom", "anchor_demon", "max_radii2D"):
        setattr(instance, name, torch.ones(4, 1))
    for name in ("mlp_opacity", "mlp_cov", "mlp_color"):
        setattr(instance, name, nn.Linear(3, 2))
    instance.use_feat_bank = feature_bank
    instance.appearance_dim = appearance
    if feature_bank:
        instance.mlp_feature_bank = nn.Linear(3, 2)
    instance.embedding_appearance = nn.Embedding(4, appearance) if appearance else None
    for name, value in dict(feat_dim=3, n_offsets=2, fork=2, add_level=False,
                            add_opacity_dist=False, add_cov_dist=False, add_color_dist=False,
                            progressive=True, dist2level="round", extend=1.1, base_layer=7,
                            update_depth=3, update_init_factor=16, update_hierachy_factor=4,
                            spatial_lr_scale=1.3, percent_dense=0.01, voxel_size=0.02,
                            levels=2, init_level=1, visible_threshold=0.0, start_step=0,
                            end_step=100, coarse_intervals=[10.0], standard_dist=torch.tensor(4.),
                            init_pos=torch.tensor([-1., -1., -1.]), cam_infos=torch.rand(12, 4),
                            _level=torch.ones(4, 1, dtype=torch.int), _extra_level=torch.zeros(4),
                            _anchor_mask=torch.ones(4, dtype=torch.bool)).items():
        setattr(instance, name, value)

    def setup(self, args):
        parameters = [getattr(self, name) for name in PARAMETERS]
        for name in ("mlp_opacity", "mlp_cov", "mlp_color", "mlp_feature_bank", "embedding_appearance"):
            network = getattr(self, name, None)
            if network is not None:
                parameters.extend(network.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=0.01)
        for name in ("opacity_accum", "offset_gradient_accum", "offset_denom", "anchor_demon"):
            setattr(self, name, torch.zeros(4, 1))
    instance.training_setup = MethodType(setup, instance)
    instance.training_setup(None)
    return instance


def step(instance):
    instance.optimizer.zero_grad(set_to_none=True)
    loss = sum(param.square().sum() for group in instance.optimizer.param_groups
               for param in group["params"] if param.requires_grad)
    loss.backward()
    instance.optimizer.step()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("feature_bank,appearance", [(False, 0), (False, 2), (True, 0), (True, 2)])
def test_checkpoint_roundtrip_continues_same_optimization(method, feature_bank, appearance):
    before = model(method, feature_bank, appearance)
    step(before)
    before.opacity_accum.fill_(5)
    captured = before.capture()
    buffer = io.BytesIO()
    torch.save(captured, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = model(method, feature_bank, appearance)
    restored.restore(saved, None)
    assert torch.equal(before.opacity_accum, restored.opacity_accum)
    assert restored.spatial_lr_scale == before.spatial_lr_scale
    if method.startswith("08"):
        for name in ("_level", "_extra_level", "_anchor_mask", "cam_infos", "init_pos"):
            assert torch.equal(getattr(before, name), getattr(restored, name))
        assert before.coarse_intervals == restored.coarse_intervals
    step(before)
    step(restored)
    for old_group, new_group in zip(before.optimizer.param_groups, restored.optimizer.param_groups):
        for old, new in zip(old_group["params"], new_group["params"]):
            assert torch.equal(old, new)
            assert old.requires_grad == new.requires_grad
    assert before.optimizer.state_dict()["param_groups"] == restored.optimizer.state_dict()["param_groups"]


@pytest.mark.parametrize("method", METHODS)
def test_checkpoint_rejects_incompatible_legacy_tuple(method):
    with pytest.raises(ValueError, match="checkpoint"):
        model(method).restore(tuple(range(11)), None)


@pytest.mark.parametrize("method", METHODS)
def test_checkpoint_after_statistics_cleanup_is_saveable(method):
    instance = model(method)
    del instance.opacity_accum, instance.offset_gradient_accum, instance.offset_denom
    instance.capture()
