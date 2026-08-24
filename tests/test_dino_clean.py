"""The regression test that guards the whole metric: DINO must come out clean.

Darcet et al. single out DINO as the exception among modern ViTs -- if this ever starts
reporting artifacts, the measurement is wrong and nothing downstream is trustworthy.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from vit_xray import inspect


@pytest.fixture(scope="module")
def image(tmp_path_factory):
    from PIL import Image
    rng = np.random.default_rng(0)
    img = np.full((448, 448, 3), 210, np.uint8)
    img[112:336, 112:336] = [50, 100, 170]
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    path = tmp_path_factory.mktemp("img") / "test.png"
    Image.fromarray(img).save(path)
    return path


@pytest.mark.slow
def test_dino_scores_near_zero(image):
    res = inspect(image, model="vit_base_patch16_224.dino", device="cpu")
    assert res.artifact_score == 0.0
    assert res.artifact_info["bimodal"] is False


@pytest.mark.slow
def test_dino_norms_are_not_heavy_tailed(image):
    # The qualitative form of the same claim: no token towers over the bulk.
    res = inspect(image, model="vit_base_patch16_224.dino", device="cpu")
    norms = res.norm_map.ravel()
    assert norms.max() / np.median(norms) < 3.0
