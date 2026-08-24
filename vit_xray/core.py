# vit-xray -- MIT License, Copyright (c) 2026 Mike Degany <mike.degany@unt.edu>
"""Measurement side: load a backbone, pull its patch tokens, turn them into panels."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from vit_xray.defaults import (  # noqa: F401  (re-exported for convenience)
    ARTIFACT_MAX_FRACTION,
    ARTIFACT_OVERLAP_GATE,
    DEFAULT_MODEL,
    SHOWCASE_MODEL,
)


def pick_device(requested) -> torch.device:
    # "auto" grabs the GPU if there is one, otherwise CPU. CPU works fine, just slower.
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def load_image(path: Path, model) -> tuple[torch.Tensor, dict]:
    # Let timm build the transform. Rolling our own is how you silently end up with the
    # wrong resolution or normalization and wonder why the maps look off.
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Image not found: {path}")

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    image = Image.open(path).convert("RGB")
    x = transform(image).unsqueeze(0)  # (1, 3, H, W)
    return x, data_config


def patch_grid(model, x: torch.Tensor) -> tuple[int, int]:
    # Ask the model for its patch size. Hardcoding 14x14 or 16x16 works right up until
    # you swap backbones, and then the reshape quietly produces garbage.
    patch_h, patch_w = model.patch_embed.patch_size  # e.g. (14, 14) for DINOv2
    _, _, height, width = x.shape

    if height % patch_h or width % patch_w:
        msg = (
            f"Input resolution {height}x{width} is not a multiple of the patch size "
            f"{patch_h}x{patch_w}. Resize the input to a multiple of the patch size "
            f"(DINOv2 in particular requires this)."
        )
        raise SystemExit(msg)
    return height // patch_h, width // patch_w


def drop_prefix_tokens(tokens: torch.Tensor, model, grid_h, grid_w, label) -> torch.Tensor:
    # Always ask the model how many prefix tokens it has. DINOv2-with-registers has 5
    # (CLS + 4 registers), so assuming 1 corrupts the grid for the very model that best
    # demonstrates the phenomenon.
    n_prefix = model.num_prefix_tokens
    patch_tokens = tokens[:, n_prefix:]  # (1, N, C)

    n_tokens = patch_tokens.shape[1]
    expected = grid_h * grid_w
    if n_tokens != expected:
        got = f"{tokens.shape[1]} total, {n_prefix} prefix, {n_tokens} patch"
        want = f"{grid_h} x {grid_w} = {expected}"
        msg = (
            f"[{label}] Patch-token count does not match the patch grid.\n"
            f"  tokens from model : {got}\n"
            f"  grid expects      : {want}\n"
            f"Refusing to reshape -- a mismatched reshape here yields a wrong but "
            f"plausible-looking map."
        )
        raise SystemExit(msg)
    return patch_tokens


def token_norms(patch_tokens: torch.Tensor, grid_h, grid_w) -> np.ndarray:
    # L2 norm per token, laid back out on the grid.
    norms = patch_tokens[0].float().norm(dim=-1)  # (N,)
    return norms.reshape(grid_h, grid_w).cpu().numpy()  # (grid_h, grid_w)


def _component_crossover(weights, means, sigmas):
    # Optimal decision boundary between the two fitted components.
    # w1*N(x;m1,s1) == w2*N(x;m2,s2) -> take logs -> ax^2 + bx + c = 0, solve for x.
    (w1, w2), (m1, m2), (s1, s2) = weights, means, sigmas
    k = np.log(w1 / w2) + np.log(s2 / s1)
    a = 1 / (2 * s2**2) - 1 / (2 * s1**2)
    b = m1 / s1**2 - m2 / s2**2
    c = m2**2 / (2 * s2**2) - m1**2 / (2 * s1**2) + k

    if abs(a) < 1e-12:  # equal variances -> the quadratic degenerates to a line
        return -c / b if abs(b) > 1e-12 else (m1 + m2) / 2

    disc = b * b - 4 * a * c
    if disc < 0:  # no real crossing, fall back to something sane
        return (m1 + m2) / 2

    roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
    inside = [r for r in roots if m1 < r < m2]
    return inside[0] if inside else (m1 + m2) / 2


def artifact_score(norm_map):
    # Fraction of tokens in the high-norm outlier population, or 0 if there isn't one.
    # Careful: a 2-component GMM will happily cut a perfectly unimodal distribution in
    # half and claim 50% artifacts, which is what the two gates below are guarding.
    norms = norm_map.ravel().astype(np.float64)

    # Standardize first -- norms run from ~23 (DINO) to ~640 (DeiT3) and EM behaves much
    # better on a common scale. The threshold gets mapped back at the end.
    mu, sigma = norms.mean(), norms.std() + 1e-12
    z = ((norms - mu) / sigma).reshape(-1, 1)  # (N, 1)

    gmm = GaussianMixture(n_components=2, n_init=10, random_state=0).fit(z)
    order = np.argsort(gmm.means_.ravel())  # component 0 = bulk, 1 = high-norm
    means = gmm.means_.ravel()[order]
    sigmas = np.sqrt(gmm.covariances_.ravel()[order])
    weights = gmm.weights_[order]

    # Shared area under the two densities: 0 = cleanly separated, 1 = indistinguishable.
    grid = np.linspace(means[0] - 5 * sigmas[0], means[1] + 5 * sigmas[1], 4000)
    densities = [w * np.exp(-0.5 * ((grid - m) / sg) ** 2) / (sg * np.sqrt(2 * np.pi))
                 for w, m, sg in zip(weights, means, sigmas)]
    overlap = float(np.trapezoid(np.minimum(densities[0], densities[1]), grid))

    blank = {"score": 0.0, "threshold": None, "overlap": overlap,
             "bimodal": False, "components": None, "rejected_fraction": None}

    if overlap > ARTIFACT_OVERLAP_GATE:
        return blank

    thresh_z = _component_crossover(weights, means, sigmas)
    frac = float((z.ravel() > thresh_z).mean())
    if frac > ARTIFACT_MAX_FRACTION:
        # Separable, but far too big to be artifacts -- this is image content.
        return {**blank, "rejected_fraction": frac}

    return {
        "score": frac,
        "threshold": float(thresh_z * sigma + mu),  # back to raw norm units
        "overlap": overlap,
        "bimodal": True,
        "components": (means * sigma + mu, sigmas * sigma, weights),
        "rejected_fraction": None,
    }


def feature_pca(patch_tokens: torch.Tensor, grid_h, grid_w, foreground) -> np.ndarray:
    # Patch tokens projected onto 3 principal components and shown as RGB.
    tokens = patch_tokens[0].float().cpu().numpy()  # (N, C)
    # random_state matters: at these shapes sklearn picks the randomized SVD solver,
    # so without it the PCA panel comes out slightly different on every run and the
    # gallery stops being reproducible.
    comps = PCA(n_components=3, random_state=0).fit_transform(tokens)  # (N, 3)

    # Clip to 2nd/98th percentile, not min/max. The artifact tokens are the outliers, so
    # a min/max stretch burns the whole colour range on ~2% of tokens and the panel
    # washes out on exactly the models you opened this tool to look at.
    rgb = np.zeros_like(comps)
    for i in range(3):
        lo, hi = np.percentile(comps[:, i], [2, 98])
        rgb[:, i] = np.clip((comps[:, i] - lo) / max(hi - lo, 1e-8), 0.0, 1.0)

    if foreground:
        # PC1 tends to split object from background, but its sign is arbitrary, so keep
        # whichever side has fewer tokens. Breaks when the subject fills the frame.
        pc1 = comps[:, 0]
        mask = pc1 > 0
        if mask.sum() > mask.size / 2:
            mask = ~mask
        rgb[~mask] = 0.0

    return rgb.reshape(grid_h, grid_w, 3)


def cls_attention(model, x: torch.Tensor, grid_h, grid_w, layer) -> np.ndarray | None:
    # CLS-to-patch attention, averaged over heads. Returns None instead of raising, so a
    # backbone we can't read still gets its other panels drawn.
    block = model.blocks[layer]
    attn_module = getattr(block, "attn", None)
    hookable = (attn_module is not None
                and hasattr(attn_module, "attn_drop")
                and hasattr(attn_module, "fused_attn"))
    if not hookable:
        print(f"  warning: block {layer} has no hookable attention module; skipping attention panel")
        return None
    if model.num_prefix_tokens == 0:
        # Nothing to attend from -- SigLIP-style models pool instead of using a CLS.
        print("  warning: model has no prefix/CLS token, so there is no CLS attention; skipping panel")
        return None

    captured = {}

    def grab_attn(module, inputs, output):
        captured["attn"] = output.detach()  # (1, heads, N, N), post-softmax

    # The catch: timm runs fused scaled_dot_product_attention by default, and the fused
    # kernel never builds the weight matrix, so there is nothing to hook. Flip it off on
    # this one module (not globally) to get the eager path, and put it back afterwards.
    was_fused = attn_module.fused_attn
    attn_module.fused_attn = False
    handle = attn_module.attn_drop.register_forward_hook(grab_attn)
    try:
        with torch.inference_mode():
            model.forward_features(x)
    finally:
        handle.remove()
        attn_module.fused_attn = was_fused

    attn = captured.get("attn")
    if attn is None:
        print("  warning: attention hook never fired; skipping attention panel")
        return None

    n_prefix = model.num_prefix_tokens
    # Row 0 is the CLS query; columns from n_prefix skip CLS and any registers.
    # (1, heads, N, N) -> (heads, n_patches) -> (n_patches,)
    cls_to_patch = attn[0, :, 0, n_prefix:].mean(dim=0)

    expected = grid_h * grid_w
    if cls_to_patch.numel() != expected:
        msg = (f"  warning: CLS attention covers {cls_to_patch.numel()} patch keys but the "
               f"grid expects {grid_h}x{grid_w}={expected}; skipping attention panel")
        print(msg)
        return None
    return cls_to_patch.reshape(grid_h, grid_w).float().cpu().numpy()


def denormalize(x: torch.Tensor, data_config) -> np.ndarray:
    # Undo the input normalization so the preprocessed image is actually viewable.
    mean = np.array(data_config["mean"]).reshape(3, 1, 1)
    std = np.array(data_config["std"]).reshape(3, 1, 1)
    img = x[0].cpu().numpy() * std + mean  # (3, H, W)
    return np.clip(img.transpose(1, 2, 0), 0.0, 1.0)  # (H, W, 3)


def analyze(model_name, image_path: Path, device: torch.device, layer, pca_foreground=False):
    # Run one model on one image and hand back every panel plus its stats.
    # num_classes=0 drops the classifier head; we only ever want features.
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.eval().to(device)

    if not hasattr(model, "blocks"):
        msg = (f"{model_name}: no .blocks attribute, so the final-block hook cannot be "
               f"attached. This tool currently supports plain ViT backbones.")
        raise SystemExit(msg)

    x, data_config = load_image(image_path, model)
    # Keep this in fp32. Autocasting here would distort the very magnitudes we measure.
    x = x.to(device)  # (1, 3, H, W)
    grid_h, grid_w = patch_grid(model, x)

    # forward_features ends with self.norm, and that LayerNorm flattens the norm spread
    # we're measuring (9.4x -> 1.08x on ViT-g), so hook the block and read it before.
    captured = {}

    def grab_block_output(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            msg = (f"{model_name}: expected block to return a Tensor, got {type(output)}. "
                   f"This model's block signature differs; the hook needs adjusting.")
            raise SystemExit(msg)
        captured["pre_norm"] = output.detach()  # (1, P + N, C)

    handle = model.blocks[layer].register_forward_hook(grab_block_output)
    try:
        with torch.inference_mode():
            post_norm = model.forward_features(x)  # (1, P + N, C), self.norm applied
    finally:
        handle.remove()

    # Every panel comes off the same tokens, same block, same prefix stripping.
    patch_tokens = drop_prefix_tokens(captured["pre_norm"], model, grid_h, grid_w, "pre-norm")
    post_tokens = drop_prefix_tokens(post_norm, model, grid_h, grid_w, "post-norm")
    pre_map = token_norms(patch_tokens, grid_h, grid_w)
    post_map = token_norms(post_tokens, grid_h, grid_w)

    # PCA wants the other side of the LayerNorm: it fits variance, and pre-norm
    # magnitudes span 53..500, so the components track magnitude rather than meaning
    # and the panel comes out as speckle. Normalized tokens leave direction behind.
    pca_rgb = feature_pca(post_tokens, grid_h, grid_w, pca_foreground)  # (H, W, 3)
    attention = cls_attention(model, x, grid_h, grid_w, layer)  # (H, W) or None

    score = artifact_score(pre_map)

    norms = pre_map.ravel()
    median = float(np.median(norms))
    result = {
        "model": model_name,
        "norm_map": pre_map,
        "post_map": post_map,
        "pca_rgb": pca_rgb,
        "attention": attention,
        "artifact": score,
        "image": denormalize(x, data_config),
        "median": median,
        "max_ratio": float(norms.max()) / median,
        # Rough cut kept for the panel title; artifact_score is the real measure.
        "outlier_frac": float((norms > 3 * median).sum()) / norms.size,
        "grid": (grid_h, grid_w),
        "input_size": (x.shape[2], x.shape[3]),
        "n_prefix": model.num_prefix_tokens,
        "depth": len(model.blocks),
    }

    # Drop the weights before the next backbone loads -- two ViT-g's won't fit at once.
    del model, captured, post_norm, x, patch_tokens, post_tokens
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@dataclass
class XrayResult:
    """What one backbone did to one image.

    Attributes:
        norm_map: (H, W) L2 norm of each patch token at the read block, pre-LayerNorm.
        pca_rgb: (H, W, 3) patch tokens projected to 3 principal components, in [0, 1].
        attention: (H, W) CLS-to-patch attention averaged over heads, or None if the
            model's attention could not be read.
        artifact_score: fraction of patch tokens in the high-norm outlier population.
            0.0 means the norms are not meaningfully bimodal, i.e. the backbone is clean.
        artifact_info: the fit behind that score -- threshold, overlap, fitted components.
        model: the timm model name.
        grid: (grid_h, grid_w) patch grid.
    """

    norm_map: np.ndarray
    pca_rgb: np.ndarray
    attention: np.ndarray | None
    artifact_score: float
    artifact_info: dict
    model: str
    grid: tuple
    raw: dict = field(repr=False, default_factory=dict)

    def plot(self, histogram=False):
        """Render the panels and return the matplotlib Figure."""
        from vit_xray.plotting import build_figure

        caption = f"{self.model}  |  {self.grid[0]}x{self.grid[1]} patches"
        return build_figure([self.raw], caption, "per-image", histogram)


def inspect(image, model=DEFAULT_MODEL, device="auto", layer=-1, pca_foreground=False):
    """Run one backbone over one image and return an XrayResult.

    Args:
        image: path to an image file.
        model: timm model name. Defaults to DINOv2 ViT-g/14, the smallest DINOv2 that
            still shows the artifact phenomenon at the final block.
        device: "auto", "cpu" or "cuda".
        layer: which block to read, as an index into model.blocks. -1 is the final block.
        pca_foreground: mask the PCA panel's background using the sign of PC1.
    """
    res = analyze(model, Path(image), pick_device(device), layer, pca_foreground)
    return XrayResult(
        norm_map=res["norm_map"],
        pca_rgb=res["pca_rgb"],
        attention=res["attention"],
        artifact_score=res["artifact"]["score"],
        artifact_info=res["artifact"],
        model=res["model"],
        grid=res["grid"],
        raw=res,
    )
