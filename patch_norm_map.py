"""
MIT License

Copyright (c) 2026 Mike Degany <mike.degany@unt.edu>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Agg backend just to save figs (not screen)

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# ViT-g is the only DINOv2 size still showing artifacts at the final block:
# 9.4x max/median, against 1.2x for both ViT-B and ViT-L.
DEFAULT_MODEL = "vit_giant_patch14_dinov2.lvd142m"

# Above this much overlap the two fitted Gaussians aren't really separate populations.
ARTIFACT_OVERLAP_GATE = 0.02

# Separability alone isn't enough: two flat colour fields make a clean model look bimodal.
# Real artifacts are always a small minority of tokens, so cap it.
ARTIFACT_MAX_FRACTION = 0.10


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


def describe(norm_map, label):
    # One-line summary of the norm distribution. The heavy tail is the artifact signal.
    norms = norm_map.ravel()
    median = float(np.median(norms))
    p99 = float(np.percentile(norms, 99))
    peak = float(norms.max())
    n_outliers = int((norms > 3 * median).sum())

    stats = f"median={median:7.2f}  p99={p99:7.2f}  max={peak:7.2f}"
    ratio = f"max/median={peak / median:6.2f}"
    tail = f"tokens>3x median: {n_outliers:>4} ({100 * n_outliers / norms.size:4.1f}%)"
    print(f"  {label:<28} {stats}  {ratio}  {tail}")


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
    comps = PCA(n_components=3).fit_transform(tokens)  # (N, 3)

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


def draw_panels(axes, res, image, norm_vmin=None, norm_vmax=None):
    # Draw the 4 panels for a single model's row
    axes[0].imshow(image)
    axes[0].set_title("input", fontsize=9)

    # Patch-norm heatmap
    im = axes[1].imshow(res["norm_map"], cmap="inferno", interpolation="nearest",
                        vmin=norm_vmin, vmax=norm_vmax)

    # Build title string first so it's not a nightmare to read
    ratio = res.get("max_ratio", 0.0)
    outliers = 100 * res.get("outlier_frac", 0.0)
    norm_title = f"patch-norm  |  {ratio:.1f}x max/med  |  {outliers:.1f}%"
    axes[1].set_title(norm_title, fontsize=9)

    # Classic StackOverflow magic numbers to make colorbar height match the plot
    axes[1].figure.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Feature PCA
    axes[2].imshow(res["pca_rgb"], interpolation="nearest")
    axes[2].set_title("feature PCA (RGB)", fontsize=9)

    # CLS Attention (if available for this model)
    attention = res.get("attention")
    if attention is None:
        axes[3].text(0.5, 0.5, "attention\nunavailable", ha="center", va="center",
                     fontsize=10, color="0.4", transform=axes[3].transAxes)
        axes[3].set_title("CLS attention", fontsize=9)
    else:
        # Note: using viridis here instead of inferno for contrast
        att_im = axes[3].imshow(attention, cmap="viridis", interpolation="nearest")
        axes[3].set_title("CLS attention (head mean)", fontsize=9)
        axes[3].figure.colorbar(att_im, ax=axes[3], fraction=0.046, pad=0.04)

    # Clean up axes
    for ax in axes[:4]:
        ax.axis("off")


def draw_histogram(ax, res):
    # Plot token-norm histogram overlayed with fitted mixture and threshold
    vals = res["norm_map"].ravel()

    # Grab artifact info defensively
    art_info = res.get("artifact", {})

    ax.hist(vals, bins=60, color="0.6", log=True)
    ax.set_xlabel("patch-token L2 norm", fontsize=8)
    ax.set_ylabel("count (log)", fontsize=8)
    ax.tick_params(labelsize=7)

    if not art_info.get("bimodal"):
        overlap = art_info.get("overlap", 0.0)
        title_str = (f"norms  |  not bimodal (overlap {overlap:.3f})\n"
                     "-> artifact-free, score 0")
        ax.set_title(title_str, fontsize=9)
        return

    means, sigmas, weights = art_info["components"]
    grid = np.linspace(vals.min(), vals.max(), 500)

    # Scale to match counts instead of density
    scale = len(vals) * (vals.max() - vals.min()) / 60

    for m, sg, w, color in zip(means, sigmas, weights, ("tab:blue", "tab:red")):
        # Standard Gaussian formula
        curve = scale * w * np.exp(-0.5 * ((grid - m) / sg) ** 2) / (sg * np.sqrt(2 * np.pi))

        # Gaussian tails reach ~1e-280, which crushes the log axis scale into nothing.
        # Trick to fix it: hide the curve when it drops below half a count.
        curve_clipped = np.where(curve < 0.5, np.nan, curve)
        ax.plot(grid, curve_clipped, color=color, lw=1.4)

    ax.axvline(art_info["threshold"], color="k", ls="--", lw=1.2)
    ax.set_ylim(bottom=0.5)

    # Build title string cleanly
    thresh = art_info["threshold"]
    score = 100 * art_info.get("score", 0.0)
    title_str = f"norms  |  threshold {thresh:.0f}  |  score {score:.2f}%"
    ax.set_title(title_str, fontsize=9)


def save_panels(results, caption, out_path: Path, norm_scale, histogram=False):
    # One row per model, plus the histogram column when asked for.
    vmin = vmax = None
    if norm_scale == "shared":
        # Only meaningful for models of the same width -- token norm scales with embedding
        # dim, so a shared scale across families renders the narrow ones nearly black.
        vmin = min(res["norm_map"].min() for res in results)
        vmax = max(res["norm_map"].max() for res in results)

    n_rows = len(results)
    n_cols = 5 if histogram else 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 4.3 * n_rows), squeeze=False)

    for row, res in zip(axes, results):
        draw_panels(row, res, res["image"], vmin, vmax)
        if histogram:
            draw_histogram(row[4], res)

        grid_h, grid_w = res.get("grid", (0, 0))
        row_title = f"{short_name(res['model'])}\n{grid_h}x{grid_w} patches"
        row[0].set_title(row_title, fontsize=9)

    fig.suptitle(caption, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def short_name(model_name):
    # Shorten a timm name so it fits in a panel title. Keep the tag after the dot --
    # vit_base_patch16_224.dino and .mae are the same arch and only differ by it.
    arch, _, tag = model_name.partition(".")
    arch = arch.replace("vit_", "").replace("_patch", "/p")
    if len(tag) > 14:
        tag = tag[:13] + "\u2026"
    return f"{arch}.{tag}" if tag else arch


def describe_score(art_info):
    # Turn the artifact dict into the one line we print under each model.
    if art_info.get("bimodal"):
        score = 100 * art_info.get("score", 0.0)
        detail = f"(GMM threshold {art_info['threshold']:.1f}, overlap {art_info['overlap']:.4f})"
        return f"  artifact score: {score:5.2f}%   {detail}"

    rejected = art_info.get("rejected_fraction")
    if rejected is not None:
        why = (f"high-norm population is {100 * rejected:.1f}% of tokens, "
               f"too large to be artifacts -- that is image content")
    else:
        overlap = art_info.get("overlap", 0.0)
        why = f"norms not bimodal, overlap {overlap:.3f} > {ARTIFACT_OVERLAP_GATE}"
    return f"  artifact score:  0.00%   ({why} -> artifact-free)"


def main():
    parser = argparse.ArgumentParser(
        description="Look inside a vision transformer: patch-token norms, feature PCA, "
                    "CLS attention, and an artifact score."
    )
    parser.add_argument("image", type=Path, help="path to an input image")
    parser.add_argument(
        "--model", action="append", default=None,
        help=f"timm model name; repeat for a comparison grid (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--out", type=Path, default=Path("out/norm_map.png"), help="output PNG path")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--layer", type=int, default=-1,
        help="which block to read, as an index into model.blocks (default: -1, the final block). "
             "Artifacts can appear mid-network and be erased by the last block, so this is how "
             "you tell 'no artifacts' apart from 'artifacts removed later'.",
    )
    parser.add_argument(
        "--histogram", action="store_true",
        help="add the raw token-norm histogram with the fitted mixture as a fifth panel.",
    )
    parser.add_argument(
        "--pca-foreground", action="store_true",
        help="mask the PCA panel's background using the sign of the first component.",
    )
    parser.add_argument(
        "--norm-scale", default="per-image", choices=["shared", "per-image"],
        help="colour scale across grid panels. 'shared' only compares like with like: token "
             "norm scales with embedding width, so it is right for same-width pairs and "
             "misleading across model families (default: per-image).",
    )
    args = parser.parse_args()

    models = args.model or [DEFAULT_MODEL]
    device = pick_device(args.device)
    print(f"device: {device}")
    print(f"image : {args.image}")
    print(f"layer : {args.layer}\n")

    results = []
    for name in models:
        res = analyze(name, args.image, device, args.layer, args.pca_foreground)
        results.append(res)

        grid_h, grid_w = res["grid"]
        in_h, in_w = res["input_size"]
        shape_line = (f"  depth={res['depth']:<3} prefix={res['n_prefix']}  "
                      f"input={in_h}x{in_w} -> {grid_h}x{grid_w}={grid_h * grid_w} tokens")
        print(name)
        print(shape_line)
        print(describe_score(res["artifact"]))
        describe(res["norm_map"], f"block {args.layer} (pre-norm)")
        describe(res["post_map"], "forward_features (post-norm)")
        print()

    caption = (f"block {args.layer} (pre-LayerNorm)  |  {args.image.name}  |  "
               f"norm-scale={args.norm_scale}")
    save_panels(results, caption, args.out, args.norm_scale, args.histogram)
    print(f"wrote : {args.out}")


if __name__ == "__main__":
    main()
