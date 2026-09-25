"""The gallery config. Parses and resolves without downloading or running anything."""

from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "gallery.yaml"


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG.read_text())


def test_config_exists_where_the_readme_says_it_does():
    assert CONFIG.is_file()


def test_every_alias_used_by_a_figure_is_defined(cfg):
    # A typo'd alias would otherwise surface as a KeyError several minutes into a run,
    # after the weights have already downloaded.
    for fig in cfg["figures"]:
        for alias in fig["models"]:
            assert alias in cfg["models"], f"{fig['name']} refers to unknown model {alias}"
        for img in fig.get("images") or [fig["image"]]:
            assert img in cfg["images"], f"{fig['name']} refers to unknown image {img}"


def test_every_model_is_a_real_timm_name(cfg):
    import timm

    known = set(timm.list_models(pretrained=True))
    for alias, name in cfg["models"].items():
        assert name in known, f"{alias} -> {name} is not a pretrained timm model"


def test_figures_have_a_known_type(cfg):
    for fig in cfg["figures"]:
        assert fig["type"] in {"zoo_grid", "panels"}
        assert fig["name"].endswith(".png")


def test_required_pairs_covers_every_figure(cfg):
    from vit_xray.gallery import required_pairs

    pairs = required_pairs(cfg)
    assert pairs
    for fig in cfg["figures"]:
        for alias in fig["models"]:
            for img in fig.get("images") or [fig["image"]]:
                assert (alias, img) in pairs


RAW = "https://raw.githubusercontent.com/MikeDegany/vit-xray/main/"


def test_readme_figures_all_exist(cfg):
    # The README embeds these; a missing one renders as a broken image.
    import re

    root = CONFIG.resolve().parent.parent
    readme = (root / "README.md").read_text()

    # Both spellings: markdown ![](path) and the HTML <img src="path"> the styled README
    # uses for width control.
    embedded = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    embedded += re.findall(r'<img[^>]+src="([^"]+)"', readme)

    # Figures have to be absolute URLs. PyPI renders the README shipped inside the
    # package and has nothing to resolve docs/hero.png against, so a relative path
    # works on GitHub and shows up broken on the PyPI page -- for that release, forever.
    relative = [e for e in embedded if not e.startswith("http")]
    assert not relative, f"relative image paths break on PyPI: {relative}"

    ours = [e[len(RAW):] for e in embedded if e.startswith(RAW)]
    assert ours, "no repo figures referenced -- did the README lose its images?"
    for path in ours:
        assert (root / path).is_file(), f"README references missing figure {path}"

    for fig in cfg["figures"]:
        assert f"docs/{fig['name']}" in readme or fig["name"] == "panels.png"


def test_feature_pca_is_deterministic():
    """PCA must give the same answer twice.

    At these shapes sklearn's `auto` solver picks randomized SVD, so an unpinned
    random_state makes the PCA panel differ on every run and the gallery stops being
    reproducible. Only the figures containing a PCA panel are affected, which makes it
    easy to miss.
    """
    import torch

    from vit_xray.core import feature_pca

    torch.manual_seed(0)
    tokens = torch.randn(1, 196, 384)
    first = feature_pca(tokens, 14, 14, False)
    second = feature_pca(tokens, 14, 14, False)
    assert (first == second).all()
