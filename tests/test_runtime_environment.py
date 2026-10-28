import os
from pathlib import Path

import pytest

from common.runtime_env import algorithm_environment, validate_native_modules


def test_extension_overlay_is_isolated_and_keeps_existing_import_path(tmp_path):
    for method_id in ("01", "02"):
        (tmp_path / method_id).mkdir()
    env = {"PYTHONPATH": "existing/import/path"}
    one = algorithm_environment("01", extension_root=tmp_path, environ=env)
    two = algorithm_environment("02", extension_root=tmp_path, environ=env)
    assert one["PYTHONPATH"].split(os.pathsep) == [str(tmp_path / "01"), "existing/import/path"]
    assert two["PYTHONPATH"].split(os.pathsep) == [str(tmp_path / "02"), "existing/import/path"]
    assert env == {"PYTHONPATH": "existing/import/path"}
    assert "02" not in one["PYTHONPATH"]


def test_explicit_missing_overlay_fails_before_training(tmp_path):
    with pytest.raises(FileNotFoundError, match="02"):
        algorithm_environment("02", extension_root=tmp_path, environ={})


def test_without_configured_overlay_existing_environments_keep_working(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    env = algorithm_environment("01", environ={"PYTHONPATH": "custom"})
    assert env.get("PYTHONPATH", "custom") == "custom"


def test_overlay_can_be_selected_through_environment_variable(tmp_path):
    (tmp_path / "03").mkdir()
    env = algorithm_environment("03", environ={"GS_EXTENSION_ROOT": str(tmp_path)})
    assert env["PYTHONPATH"] == str(tmp_path / "03")


def test_native_import_probe_rejects_fallback_outside_overlay(tmp_path):
    import sys
    with pytest.raises(RuntimeError, match="outside"):
        validate_native_modules(sys.executable, tmp_path, "00", modules=["json"])


def test_native_import_probe_accepts_only_selected_import_origin(tmp_path):
    import sys
    module = tmp_path / "unique_smoke_extension.py"
    module.write_text("marker = 1\n")
    result = validate_native_modules(sys.executable, tmp_path, "00", modules=[module.stem])
    assert Path(result[module.stem]) == module


@pytest.mark.parametrize("isolated", [True, False])
def test_mip_rasterizer_cannot_be_used_as_algorithm_01(tmp_path, isolated):
    import sys
    package = tmp_path / "diff_gaussian_rasterization"
    package.mkdir()
    root = Path(__file__).resolve().parents[1]
    source = root / "05_Mip_Splatting/runtime/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py"
    (package / "__init__.py").write_bytes(source.read_bytes())
    (package / "_C.py").write_text("# Stand-in only for import-path/contract inspection, never training\n")
    with pytest.raises(RuntimeError, match="kernel_size"):
        validate_native_modules(sys.executable, tmp_path if isolated else None, "01",
                                env={"PYTHONPATH": str(tmp_path)},
                                modules=["diff_gaussian_rasterization._C"])
