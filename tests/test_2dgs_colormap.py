"""CPU-only rendering checks for the actual 2DGS colormap function."""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch


SOURCE = (Path(__file__).resolve().parents[1]
          / "02_2DGS/runtime/utils/general_utils.py")


@pytest.fixture(scope="module")
def general_utils():
    spec = importlib.util.spec_from_file_location("two_dgs_general_utils", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("shape", [(256, 384), (384, 256)])
def test_colormap_returns_finite_rgb_float_tensor_at_original_size(general_utils, shape):
    image = np.linspace(-2, 3, np.prod(shape), dtype=np.float32).reshape(shape)
    result = general_utils.colormap(image)

    assert result.shape == (3, *shape)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.isfinite(result).all()
    assert result.min().item() >= 0
    assert result.max().item() <= 1
    assert not torch.equal(result[0], result[2])


@pytest.mark.parametrize("cmap", ["jet", "viridis"])
def test_colormap_preserves_legacy_figure_layout_and_rgb_order(general_utils, cmap):
    image = np.linspace(0, 1, 480 * 640, dtype=np.float32).reshape(480, 640)
    image[200:210, 300:310] = np.nan
    # Keep the original pyplot layout as a reference. print_to_buffer is a
    # separate supported Agg API, so the expected pixels do not use the
    # production function's RGB extraction path.
    figure, axis = plt.subplots(1, figsize=(640 / 300, 480 / 300), dpi=300)
    try:
        artist = axis.imshow(image, cmap=cmap)
        axis.set_axis_off()
        figure.colorbar(artist, ax=axis)
        figure.tight_layout()
        rgba_bytes, (width, height) = figure.canvas.print_to_buffer()
        expected = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(height, width, 4)
        expected = (expected[..., :3] / 255).astype(np.float32).transpose(2, 0, 1)
    finally:
        plt.close(figure)

    actual = general_utils.colormap(image, cmap=cmap)

    np.testing.assert_array_equal(actual.numpy(), expected)


def test_colormap_keeps_callers_figures_and_does_not_leak_figures(general_utils):
    caller_figure = plt.figure()
    try:
        before = plt.get_fignums()
        for _ in range(2):
            general_utils.colormap(np.arange(256 * 384).reshape(256, 384))
            assert plt.get_fignums() == before
            assert plt.gcf() is caller_figure
    finally:
        plt.close(caller_figure)
