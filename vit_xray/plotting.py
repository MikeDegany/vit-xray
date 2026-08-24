# vit-xray -- MIT License, Copyright (c) 2026 Mike Degany <mike.degany@unt.edu>
"""Rendering side: turn analysis results into a figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Note: no matplotlib.use("Agg") here on purpose. The CLI sets it, but a library that
# hijacks the backend breaks anyone plotting interactively in a notebook.


def short_name(model_name):
    # Shorten a timm name so it fits in a panel title. Keep the tag after the dot --
    # vit_base_patch16_224.dino and .mae are the same arch and only differ by it.
    arch, _, tag = model_name.partition(".")
    arch = arch.replace("vit_", "").replace("_patch", "/p")
    if len(tag) > 14:
        tag = tag[:13] + "…"
    return f"{arch}.{tag}" if tag else arch


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


def build_figure(results, caption, norm_scale="per-image", histogram=False):
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
    return fig


def build_zoo_grid(results, model_names, image_names, caption):
    # Rows are images, columns are the input plus one model each. Laid out this way round
    # because a README hero wants to be wide, not tall -- transposed it is seven rows deep
    # and nothing but the first model is above the fold.
    #
    # Scale is per-cell on purpose: token norm scales with embedding width, so a shared
    # scale across a 1536-dim ViT-g and a 768-dim ViT-B would render the narrow models
    # black and compare quantities that aren't comparable.
    n_rows = len(image_names)
    n_cols = len(model_names) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.35 * n_cols, 2.5 * n_rows), squeeze=False)

    for row, img_name in enumerate(image_names):
        ax = axes[row][0]
        ax.imshow(results[(model_names[0], img_name)]["image"])
        ax.set_ylabel(img_name, fontsize=10)
        if row == 0:
            ax.set_title("input", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        for col, model_name in enumerate(model_names, start=1):
            ax = axes[row][col]
            res = results[(model_name, img_name)]
            ax.imshow(res["norm_map"], cmap="inferno", interpolation="nearest")
            score = 100 * res["artifact"].get("score", 0.0)
            label = f"{score:.1f}%" if score else "clean"
            colour = "0.35" if score else "tab:green"
            if row == 0:
                ax.set_title(f"{model_name}\n{label}", fontsize=9, color=colour)
            else:
                ax.set_title(label, fontsize=9, color=colour)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(caption, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # leave the suptitle its own strip
    return fig


def save_figure(fig, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_panels(results, caption, out_path: Path, norm_scale, histogram=False):
    # Build the figure and write it out.
    fig = build_figure(results, caption, norm_scale, histogram)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
