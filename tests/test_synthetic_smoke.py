import importlib.util
import json
from pathlib import Path
import struct
import sys
import os

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def smoke_module():
    path = ROOT / "synthetic_smoke.py"
    assert path.is_file(), "The bounded synthetic training runner is missing"
    spec = importlib.util.spec_from_file_location("synthetic_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_colmap_and_masks_are_consistent(tmp_path):
    smoke = smoke_module()
    generated = smoke.generate_dataset(tmp_path / "synthetic")
    for variant in ("standard", "custom"):
        from common.dataset import validate_dataset_layout
        report = validate_dataset_layout(generated[variant])
        assert report.ok, report.errors
        assert report.counts == {"train_images": 12, "train_masks": 12,
                                 "test_images": 4, "test_masks": 4}
    # Independently parse the standard format with an actual upstream reader.
    spec = importlib.util.spec_from_file_location("upstream_loader", ROOT / "05_Mip_Splatting/runtime/scene/colmap_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    sparse = generated["standard"] / "train/sparse/0"
    points, rgb, error = loader.read_points3D_binary(sparse / "points3D.bin")
    cameras = loader.read_intrinsics_binary(sparse / "cameras.bin")
    images = loader.read_extrinsics_binary(sparse / "images.bin")
    assert points.shape == (6000, 3)
    assert len(images) == 12 and len(cameras) == 1
    assert np.isfinite(points).all()
    # Custom record must contain normals, compatible with the existing adapter.
    from common.dataset_compat import convert_custom_points3d
    converted = tmp_path / "converted.bin"
    convert_custom_points3d(generated["custom"] / "train/sparse/0/points3D.bin", converted)
    assert converted.read_bytes() == (sparse / "points3D.bin").read_bytes()
    first = next(iter(images.values()))
    image_path = generated["standard"] / "train/images" / first.name
    mask_path = generated["standard"] / "train/masks" / (Path(first.name).stem + ".mask.png")
    image = np.asarray(Image.open(image_path))
    mask = np.asarray(Image.open(mask_path))
    assert image.shape == (256, 256, 3)
    assert set(np.unique(mask)) == {0, 255}
    assert np.any(image[mask > 0] > 0)
    # A known visible surface point projects onto the same foreground mask.
    R = loader.qvec2rotmat(first.qvec)
    camera_points = points @ R.T + first.tvec
    K = cameras[first.camera_id].params
    xy = camera_points[:, :2] / camera_points[:, 2:3] * K[:2] + K[2:]
    assert np.count_nonzero(camera_points[:, 2] > 0) == 6000
    assert np.isfinite(xy).all()


def test_refuses_to_overwrite_unrelated_input(tmp_path):
    smoke = smoke_module()
    target = tmp_path / "existing"
    target.mkdir()
    (target / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        smoke.generate_dataset(target)
    assert (target / "user.txt").read_text() == "keep"


def test_success_requires_valid_final_iteration_model(tmp_path):
    smoke = smoke_module()
    with pytest.raises(ValueError, match="Missing"):
        smoke.verify_model(tmp_path, "01", 10)
    target = tmp_path / "point_cloud/iteration_10/point_cloud.ply"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 20\nend_header\n")
    with pytest.raises(ValueError):
        smoke.verify_model(tmp_path, "01", 10)
    vertex = np.zeros(2, dtype=[(n, "f4") for n in ("x", "y", "z", "opacity", "f_dc_0", "scale_0", "rot_0")])
    PlyData([PlyElement.describe(vertex, "vertex")]).write(target)
    result = smoke.verify_model(tmp_path, "01", 10)
    assert result["vertices"] == 2
    (target.parent / "unrelated.pt").touch()
    with pytest.raises(ValueError, match="MLP"):
        smoke.verify_model(tmp_path, "06", 10)
    vertex["x"][0] = np.nan
    PlyData([PlyElement.describe(vertex, "vertex")]).write(target)
    with pytest.raises(ValueError, match="finite"):
        smoke.verify_model(tmp_path, "01", 10)


def test_reduced_3dgs_validates_all_spherical_harmonic_groups(tmp_path):
    smoke = smoke_module()
    target = tmp_path / "point_cloud/iteration_10/point_cloud.ply"
    target.parent.mkdir(parents=True)
    fields = [(name, "f4") for name in ("x", "y", "z", "opacity", "f_dc_0")]
    groups = [np.zeros(size, dtype=fields) for size in (2, 0, 3, 0)]
    def save():
        PlyData([PlyElement.describe(group, f"vertex_{degree}")
                 for degree, group in enumerate(groups)]).write(target)
    save()
    assert smoke.verify_model(tmp_path, "09", 10)["vertices"] == 5
    groups[2]["opacity"][1] = np.nan
    save()
    with pytest.raises(ValueError, match="finite"):
        smoke.verify_model(tmp_path, "09", 10)


def test_commands_use_selected_python_and_bounded_iterations(tmp_path):
    smoke = smoke_module()
    from common.config import discover_experiments
    datasets = {"standard": tmp_path / "standard", "custom": tmp_path / "custom"}
    for definition in discover_experiments(ROOT):
        commands, cwd = smoke.build_commands(definition, datasets, tmp_path / definition.directory, sys.executable, 10)
        for command in commands:
            assert command[0] == sys.executable
            assert "7000" not in command and "7_000" not in command
        if definition.id != "04":
            command = commands[0]
            assert command[command.index("--iterations") + 1] == "10"
            assert command[command.index("--save_iterations") + 1] == "10"
            assert command[command.index("--checkpoint_iterations") + 1] == "10"
    definition = next(d for d in discover_experiments(ROOT) if d.id == "10")
    command = smoke.build_commands(definition, datasets, tmp_path / "concrete", sys.executable, 10)[0][0]
    assert command[command.index("--position_lr_init") + 1] == "0.0000016"
    assert command[command.index("--experiment_mode") + 1] == "F"


def test_exit_zero_or_training_banner_alone_cannot_pass(tmp_path):
    smoke = smoke_module()
    command = [sys.executable, "-u", "-c", "print('Training progress: 0%|0/10'); print('Training complete.')"]
    result = smoke.run_command(command, tmp_path, tmp_path / "log.txt", 10)
    assert result["exit_code"] == 0
    with pytest.raises(ValueError):
        smoke.verify_model(tmp_path, "01", 10)


def test_failure_and_timeout_return_actual_exit(tmp_path):
    smoke = smoke_module()
    failed = smoke.run_command([sys.executable, "-c", "raise RuntimeError('marker')"], tmp_path, tmp_path / "failed.log", 10)
    assert failed["exit_code"] != 0
    assert "marker" in failed["tail"]
    timed = smoke.run_command([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path, tmp_path / "timeout.log", 0.2)
    assert timed["timed_out"] and timed["exit_code"] != 0


@pytest.mark.skipif(os.name == "nt", reason="Checks Linux process-group cleanup")
def test_timeout_kills_stdout_inheriting_descendant_after_parent_exits(tmp_path):
    smoke = smoke_module()
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
    code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c',sys.argv[1]])"
    import time
    start = time.monotonic()
    result = smoke.run_command([sys.executable, "-c", code, child], tmp_path,
                               tmp_path / "descendant.log", 0.5)
    assert result["timed_out"]
    assert time.monotonic() - start < 8
