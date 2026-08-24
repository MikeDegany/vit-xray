"""Prefix-token handling per model family.

pretrained=False downloads nothing, but timm still reports the real architecture, so
these run offline in about a second.
"""

import pytest
import timm
import torch

from tests.conftest import FakeModel
from vit_xray.core import drop_prefix_tokens, patch_grid


@pytest.mark.parametrize("name, expected_prefix", [
    ("vit_base_patch14_dinov2.lvd142m", 1),
    ("vit_giant_patch14_dinov2.lvd142m", 1),
    ("vit_base_patch16_224.dino", 1),
    ("vit_base_patch16_clip_224.laion2b", 1),
    ("deit3_base_patch16_224.fb_in22k_ft_in1k", 1),
    ("vit_base_patch16_224.mae", 1),
    # The ones that matter: CLS + 4 registers. Assuming 1 here silently corrupts the
    # grid for exactly the models that best demonstrate the phenomenon.
    ("vit_base_patch14_reg4_dinov2.lvd142m", 5),
    ("vit_giant_patch14_reg4_dinov2.lvd142m", 5),
])
def test_num_prefix_tokens(name, expected_prefix):
    model = timm.create_model(name, pretrained=False, num_classes=0)
    assert model.num_prefix_tokens == expected_prefix


@pytest.mark.parametrize("name", [
    "vit_base_patch14_dinov2.lvd142m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch16_224.dino",
    "deit3_base_patch16_224.fb_in22k_ft_in1k",
])
def test_grid_matches_token_count(name):
    # The grid derived from patch size must account for exactly the non-prefix tokens.
    model = timm.create_model(name, pretrained=False, num_classes=0)
    cfg = timm.data.resolve_model_data_config(model)
    _, height, width = cfg["input_size"]

    x = torch.zeros(1, 3, height, width)
    grid_h, grid_w = patch_grid(model, x)

    n_prefix = model.num_prefix_tokens
    tokens = torch.zeros(1, n_prefix + grid_h * grid_w, model.embed_dim)
    kept = drop_prefix_tokens(tokens, model, grid_h, grid_w, "test")
    assert kept.shape[1] == grid_h * grid_w


def test_wrong_prefix_count_raises():
    # A reg4 model read as if it were CLS-only leaves 4 tokens too many. That must blow
    # up loudly rather than reshape into a wrong-but-plausible map.
    tokens = torch.zeros(1, 5 + 1369, 768)
    with pytest.raises(SystemExit, match="does not match the patch grid"):
        drop_prefix_tokens(tokens, FakeModel(num_prefix_tokens=1), 37, 37, "pre-norm")


def test_correct_prefix_count_passes():
    tokens = torch.zeros(1, 5 + 1369, 768)
    kept = drop_prefix_tokens(tokens, FakeModel(num_prefix_tokens=5), 37, 37, "pre-norm")
    assert kept.shape == (1, 1369, 768)


def test_non_multiple_resolution_raises():
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    x = torch.zeros(1, 3, 520, 520)  # 520 is not a multiple of 14
    with pytest.raises(SystemExit, match="not a multiple of the patch size"):
        patch_grid(model, x)
