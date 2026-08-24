"""Shared fixtures. Test images are synthesized, never read from assets/ -- that
directory is gitignored, so a fresh clone has none and such a test would fail for
anyone but us."""

import numpy as np
import pytest
from PIL import Image


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run tests that need real pretrained weights")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs pretrained weights; pass --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def image_path(tmp_path):
    # A structured image, not pure noise: some models refuse to produce anything
    # interesting on white noise, and we want the panels to be meaningful.
    rng = np.random.default_rng(0)
    h = w = 560
    img = np.full((h, w, 3), 200, np.uint8)
    img[h // 4:3 * h // 4, w // 4:3 * w // 4] = [40, 90, 160]   # a block in the middle
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)

    path = tmp_path / "test.png"
    Image.fromarray(img).save(path)
    return path


class FakeModel:
    """Stands in for a timm model where all we need is num_prefix_tokens."""

    def __init__(self, num_prefix_tokens):
        self.num_prefix_tokens = num_prefix_tokens
