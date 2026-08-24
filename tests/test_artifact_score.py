"""The artifact score, on synthetic distributions. No models, no network."""

import numpy as np

from vit_xray.core import ARTIFACT_MAX_FRACTION, _component_crossover, artifact_score


def test_unimodal_scores_zero():
    # A plain Gaussian has no outlier population. A 2-component GMM will still fit it and
    # happily cut it in half, so this is really testing that the overlap gate holds.
    rng = np.random.default_rng(0)
    norms = rng.normal(50, 5, 1369)
    out = artifact_score(norms)
    assert out["score"] == 0.0
    assert out["bimodal"] is False


def test_bimodal_finds_the_tail():
    # 2% of tokens at ~10x the bulk, which is the phenomenon we are measuring.
    rng = np.random.default_rng(0)
    bulk = rng.normal(50, 5, 1341)
    tail = rng.normal(500, 20, 28)
    out = artifact_score(np.concatenate([bulk, tail]))

    assert out["bimodal"] is True
    assert 0.015 < out["score"] < 0.030
    assert 50 < out["threshold"] < 500  # lands in the gap, not on top of either mode


def test_huge_high_population_is_rejected():
    # Two well-separated but evenly sized modes is image content, not artifacts. Without
    # the fraction cap this is a false positive: an image made of two flat colour fields
    # makes even a clean backbone fit with low overlap, reporting ~18% "artifacts".
    rng = np.random.default_rng(0)
    low = rng.normal(20, 1, 800)
    high = rng.normal(60, 1, 569)
    out = artifact_score(np.concatenate([low, high]))

    assert out["score"] == 0.0
    assert out["bimodal"] is False
    assert out["rejected_fraction"] > ARTIFACT_MAX_FRACTION


def test_crossover_sits_between_equal_components():
    # Equal weights and equal widths -> the boundary is exactly halfway.
    x = _component_crossover((0.5, 0.5), (0.0, 10.0), (1.0, 1.0))
    assert np.isclose(x, 5.0)


def test_crossover_shifts_toward_the_rarer_mode():
    # A rare high component means the boundary should move up, away from the bulk.
    even = _component_crossover((0.5, 0.5), (0.0, 10.0), (1.0, 1.0))
    rare = _component_crossover((0.98, 0.02), (0.0, 10.0), (1.0, 1.0))
    assert rare > even
