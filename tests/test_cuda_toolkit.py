from pathlib import Path
import pytest

from common.cuda_toolkit import find_cuda_toolkit, CudaToolkitError


def toolkit(tmp_path, name, version, versions):
    root = tmp_path / name
    binary = root / "bin/nvcc"
    binary.parent.mkdir(parents=True)
    binary.write_text("placeholder")
    versions[str(binary.resolve())] = version
    return root


def find(tmp_path, versions, **kwargs):
    return find_cuda_toolkit(prefix=tmp_path / "env", search_dirs=[],
                             which=lambda *a, **k: None,
                             version_probe=lambda binary: versions[str(binary)], **kwargs)


def test_existing_cuda_home_118_is_used_without_hardcoded_128(tmp_path):
    versions = {}
    root = toolkit(tmp_path, "cuda-11.8", "11.8", versions)
    actual = find(tmp_path, versions, torch_cuda="11.8", environ={"CUDA_HOME": str(root)})
    assert actual.home == root and actual.version == "11.8"


def test_stale_cuda_home_does_not_hide_working_path_compiler(tmp_path):
    versions = {}
    root = toolkit(tmp_path, "cuda-11.8", "11.8", versions)
    actual = find_cuda_toolkit(torch_cuda="11.8", environ={"CUDA_HOME": str(tmp_path / "missing")},
        prefix=tmp_path / "env", search_dirs=[], which=lambda *a, **k: str(root / "bin/nvcc"),
        version_probe=lambda binary: versions[str(binary)])
    assert actual.home == root


def test_exact_torch_version_is_preferred_over_other_installed_major(tmp_path):
    versions = {}
    newer = toolkit(tmp_path, "cuda-12.8", "12.8", versions)
    matching = toolkit(tmp_path, "cuda-11.8", "11.8", versions)
    actual = find_cuda_toolkit(torch_cuda="11.8", environ={"CUDA_HOME": str(newer)},
        prefix=tmp_path / "env", search_dirs=[matching], which=lambda *a, **k: None,
        version_probe=lambda binary: versions[str(binary)])
    assert actual.home == matching


def test_explicit_missing_toolkit_fails_instead_of_silently_using_another(tmp_path):
    with pytest.raises(CudaToolkitError, match="nvcc"):
        find(tmp_path, {}, requested=tmp_path / "missing", torch_cuda="11.8", environ={})


def test_cross_major_mismatch_has_clear_error_before_package_build(tmp_path):
    versions = {}
    root = toolkit(tmp_path, "cuda-12.8", "12.8", versions)
    with pytest.raises(CudaToolkitError, match="11.8"):
        find(tmp_path, versions, torch_cuda="11.8", environ={"CUDA_HOME": str(root)})


def test_no_compiler_explains_compiler_requirement(tmp_path):
    with pytest.raises(CudaToolkitError, match="nvcc"):
        find(tmp_path, {}, torch_cuda="11.8", environ={})


def test_builder_stops_once_before_pip_when_compiler_is_unavailable(monkeypatch, capsys):
    import wsl_build_extensions as builder
    monkeypatch.setattr(builder.sys, "platform", "linux")
    def missing(*args, **kwargs):
        raise CudaToolkitError("nvcc is missing")
    def unexpected(*args, **kwargs):
        raise AssertionError("No package builds should run without nvcc")
    monkeypatch.setattr(builder, "find_cuda_toolkit", missing)
    monkeypatch.setattr(builder.subprocess, "run", unexpected)
    assert builder.main(["--only", "01"]) == 2
    assert "before any package build" in capsys.readouterr().err
