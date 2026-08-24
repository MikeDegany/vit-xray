# vit-xray -- MIT License, Copyright (c) 2026 Mike Degany <mike.degany@unt.edu>
"""Command line entry point for `vit-xray`.

All the presentation lives here on purpose -- core.py and plotting.py stay plain so the
measurement code is readable on its own.
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vit_xray import __version__
from vit_xray.defaults import DEFAULT_MODEL, SHOWCASE_MODEL

# numpy, matplotlib, torch and timm are imported inside the command instead of up here.
# They cost about six seconds between them, and `--help` and `--version` need none of it.

console = Console()


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


class NormScale(str, Enum):
    per_image = "per-image"
    shared = "shared"


HELP = """
🩻 [bold]vit-xray[/bold] · look inside any vision transformer backbone in one command 🩻

Renders the patch-token [bold]norm map[/bold], a [bold]feature PCA[/bold], and the [bold]CLS attention[/bold] for a backbone and an image, then scores how much of its token budget the model has quietly repurposed as scratch space.

[bold]Examples:[/bold]

[dim]# The default backbone is small enough for any laptop (329 MB)[/dim]
$ vit-xray photo.jpg 🖼

[dim]# The clearest picture, if you can spare 4.3 GB and ~10 GB of RAM on CPU[/dim]
$ vit-xray photo.jpg --model vit_giant_patch14_dinov2.lvd142m

[dim]# Compare backbones side by side, one row per model[/dim]
$ vit-xray photo.jpg --model vit_base_patch16_224.dino --model vit_giant_patch14_dinov2.lvd142m

[dim]# Every panel, including the norm histogram with the fitted threshold[/dim]
$ vit-xray photo.jpg --all-panels -o out.png 📊

[dim]# Registers on vs off, on one shared colour scale[/dim]
$ vit-xray photo.jpg -m vit_giant_patch14_dinov2.lvd142m -m vit_giant_patch14_reg4_dinov2.lvd142m --norm-scale shared

[dim]# Artifacts can appear mid-network and be erased by the last block. Look earlier:[/dim]
$ vit-xray photo.jpg --model deit3_base_patch16_224.fb_in22k_ft_in1k --layer 6
"""

app = typer.Typer(
    rich_markup_mode="rich",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version(value: bool):
    if value:
        console.print(f"🩻 [bold]vit-xray[/bold] {__version__}")
        raise typer.Exit()


def max_over_median(norm_map):
    import numpy as np

    norms = norm_map.ravel()
    return float(norms.max()) / float(np.median(norms))


def verdict(art_info):
    # Short human verdict plus the reason, for the summary table.
    if art_info.get("bimodal"):
        return "⚠️  artifacts", f"threshold {art_info['threshold']:.0f}"

    rejected = art_info.get("rejected_fraction")
    if rejected is not None:
        return "✅ clean", f"{100 * rejected:.0f}% high-norm = content"
    return "✅ clean", f"overlap {art_info.get('overlap', 0.0):.3f}, unimodal"


def showcase_hint(models):
    # Nudge toward the model that makes the best picture, but only when the user took the
    # default. If they named a model they have already made this choice. Returns None so
    # the caller stays a one-liner.
    if models != [DEFAULT_MODEL]:
        return None
    return (
        "💡 That was the lightweight default. The clearest picture comes from "
        f"[bold]DINOv2 ViT-g/14[/bold], a 37×37 grid with ~20 outlier tokens:\n"
        f"   [cyan]--model {SHOWCASE_MODEL}[/cyan]  "
        "[dim](4.3 GB download, ~10 GB RAM on CPU)[/dim]"
    )


def summary_table(results, layer):
    import numpy as np

    from vit_xray.plotting import short_name

    table = Table(title=f"🩻 patch-token analysis · block {layer} (pre-LayerNorm)",
                  title_style="bold", header_style="bold cyan", expand=False)
    table.add_column("backbone", style="bold")
    table.add_column("grid", justify="right")
    table.add_column("prefix", justify="right")
    table.add_column("median", justify="right")
    table.add_column("max", justify="right")
    table.add_column("pre", justify="right")
    table.add_column("post", justify="right", style="dim")
    table.add_column("score", justify="right", style="bold")
    table.add_column("verdict")
    table.add_column("detail", style="dim")

    for res in results:
        grid_h, grid_w = res["grid"]
        art_info = res["artifact"]
        mark, why = verdict(art_info)
        score = 100 * art_info.get("score", 0.0)
        score_cell = f"[yellow]{score:.2f}%[/yellow]" if score else "[green]0.00%[/green]"

        norms = res["norm_map"].ravel()
        table.add_row(
            short_name(res["model"]), f"{grid_h}×{grid_w}", str(res["n_prefix"]),
            f"{float(np.median(norms)):.1f}", f"{float(norms.max()):.1f}",
            f"{max_over_median(res['norm_map']):.2f}x",
            f"{max_over_median(res['post_map']):.2f}x",
            score_cell, mark, why,
        )
    return table


# "pre" and "post" are the max/median ratio either side of the final LayerNorm. The gap
# between them is why the norm map is read from a hook before self.norm rather than from
# forward_features -- on ViT-g it is 9.4x against 1.08x.


@app.command(help=HELP)
def xray(
    image: Optional[Path] = typer.Argument(
        None, exists=True, dir_okay=False, readable=True, metavar="IMAGE",
        help="🖼  Image to run through the backbone",
    ),
    model: Optional[List[str]] = typer.Option(
        None, "--model", "-m", metavar="NAME",
        help=f"timm model name; repeat it to compare backbones  [dim](default: {DEFAULT_MODEL})[/dim]",
        rich_help_panel="🧠 Model & device",
    ),
    device: Device = typer.Option(
        Device.auto, "--device", "-d", help="Where to run. CPU works, just slower",
        rich_help_panel="🧠 Model & device",
    ),
    layer: int = typer.Option(
        -1, "--layer", "-l", metavar="N",
        help="Which block to read, as an index into [italic]model.blocks[/italic]. Artifacts "
             "can appear mid-network and be erased by the last block, so this is how you tell "
             "[italic]no artifacts[/italic] from [italic]artifacts removed later[/italic]",
        rich_help_panel="🧠 Model & device",
    ),
    all_panels: bool = typer.Option(
        False, "--all-panels", "-a", help="📊 Every panel, i.e. the three plus the histogram",
        rich_help_panel="🎨 Panels",
    ),
    histogram: bool = typer.Option(
        False, "--histogram", help="Add the norm histogram with the fitted mixture",
        rich_help_panel="🎨 Panels",
    ),
    pca_foreground: bool = typer.Option(
        False, "--pca-foreground", help="Mask the PCA panel's background using the sign of PC1",
        rich_help_panel="🎨 Panels",
    ),
    norm_scale: NormScale = typer.Option(
        NormScale.per_image, "--norm-scale",
        help="Colour scale across rows. [bold]shared[/bold] only compares like with like: "
             "token norm scales with embedding width, so it suits same-width pairs and "
             "misleads across model families",
        rich_help_panel="🎨 Panels",
    ),
    out: Path = typer.Option(
        Path("out/norm_map.png"), "--out", "-o", metavar="PATH", help="💾 Where to write the figure",
        rich_help_panel="💾 Output",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only print the output path",
        rich_help_panel="💾 Output",
    ),
    gallery: Optional[Path] = typer.Option(
        None, "--gallery", exists=True, dir_okay=False, metavar="CONFIG",
        help="🖼  Regenerate every README figure from a gallery config, e.g. "
             "[cyan]configs/gallery.yaml[/cyan]. Ignores the other options.",
        rich_help_panel="💾 Output",
    ),
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True,
        help="Show the version and exit", rich_help_panel="💾 Output",
    ),
):
    import matplotlib

    matplotlib.use("Agg")  # Agg backend just to save figs (not screen)

    from vit_xray.core import analyze, pick_device
    from vit_xray.plotting import save_panels

    if gallery is not None:
        # Imported here, not at module scope: gallery.py pulls in yaml and torch.
        from vit_xray.gallery import run as run_gallery

        console.print(Panel(f"regenerating figures from [bold]{gallery}[/bold]",
                            title="🩻 vit-xray gallery", border_style="cyan", expand=False))
        written = run_gallery(gallery, device.value, console)
        console.print(f"\n💾 wrote [bold green]{len(written)}[/bold green] figures")
        return

    if image is None:
        raise typer.BadParameter("give an IMAGE, or --gallery CONFIG to rebuild the figures")

    models = list(model) if model else [DEFAULT_MODEL]
    show_histogram = histogram or all_panels
    torch_device = pick_device(device.value)

    if not quiet:
        header = (f"[bold]{image.name}[/bold]   ·   device [cyan]{torch_device}[/cyan]   ·   "
                  f"block [cyan]{layer}[/cyan]   ·   [dim]{len(models)} backbone(s)[/dim]")
        console.print(Panel(header, title="🩻 vit-xray", border_style="cyan", expand=False))

    # Only spin on a real terminal. Piped into a file, Live would write thousands of
    # spinner frames and bury the actual output.
    spin = not quiet and console.is_terminal

    results = []
    for name in models:
        if spin:
            with console.status(f"[cyan]{name}[/cyan] …", spinner="dots"):
                results.append(analyze(name, image, torch_device, layer, pca_foreground))
        else:
            results.append(analyze(name, image, torch_device, layer, pca_foreground))
        if not quiet:
            console.print(f"  [green]✓[/green] {name}")

    caption = (f"block {layer} (pre-LayerNorm)  |  {image.name}  |  "
               f"norm-scale={norm_scale.value}")
    save_panels(results, caption, out, norm_scale.value, show_histogram)

    if not quiet:
        console.print()
        console.print(summary_table(results, layer))
        console.print()
    console.print(f"💾 wrote [bold green]{out}[/bold green]")

    hint = showcase_hint(models)
    if hint and not quiet:
        console.print()
        console.print(hint)


def main():
    app()


if __name__ == "__main__":
    main()
