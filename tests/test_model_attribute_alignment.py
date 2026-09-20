from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_custom_gaussian_models_keep_normals_aligned_after_topology_changes() -> None:
    for relative_path in (
        Path("01_3DGS/runtime/scene/gaussian_model.py"),
        Path("10_Concrete_GS/runtime/scene/gaussian_model.py"),
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "self._normals = self._normals[valid_points_mask]" in source
        assert "new_normals" in source
        assert "self._normals = torch.cat" in source
