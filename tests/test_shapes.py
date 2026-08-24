"""End-to-end shape assertions using a small real backbone (85 MB)."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from vit_xray import XrayResult, inspect

SMALL = "vit_small_patch14_dinov2.lvd142m"


@pytest.fixture(scope="module")
def result_and_grid(tmp_path_factory):
    from PIL import Image
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(140, 40, (560, 560, 3)), 0, 255).astype(np.uint8)
    path = tmp_path_factory.mktemp("img") / "test.png"
    Image.fromarray(img).save(path)
    return inspect(path, model=SMALL, device="cpu")


@pytest.mark.slow
def test_result_type(result_and_grid):
    assert isinstance(result_and_grid, XrayResult)
    assert result_and_grid.model == SMALL


@pytest.mark.slow
def test_panel_shapes(result_and_grid):
    res = result_and_grid
    grid_h, grid_w = res.grid
    assert res.norm_map.shape == (grid_h, grid_w)
    assert res.pca_rgb.shape == (grid_h, grid_w, 3)
    assert res.attention.shape == (grid_h, grid_w)


@pytest.mark.slow
def test_value_ranges(result_and_grid):
    res = result_and_grid
    assert isinstance(res.artifact_score, float)
    assert 0.0 <= res.artifact_score <= 1.0
    assert res.pca_rgb.min() >= 0.0 and res.pca_rgb.max() <= 1.0
    assert (res.norm_map > 0).all()
    # Attention is a softmax row restricted to patch keys, so it sums to at most 1.
    assert 0.0 < res.attention.sum() <= 1.0


@pytest.mark.slow
def test_plot_returns_a_figure(result_and_grid):
    fig = result_and_grid.plot()
    assert fig.__class__.__name__ == "Figure"
    assert len(fig.axes) >= 4
