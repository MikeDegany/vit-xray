# vit-xray -- MIT License, Copyright (c) 2026 Mike Degany <mike.degany@unt.edu>
"""Regenerate every README figure from configs/gallery.yaml.

Kept out of the CLI's import path -- it pulls in yaml and torch, and `--help` should not
pay for either.
"""

import urllib.request
from pathlib import Path

import yaml


def fetch_images(cfg, root: Path, console=None):
    # Download any image we don't have yet. assets/ is gitignored, so a fresh clone starts
    # empty and this is what makes the gallery reproducible without committing photos.
    image_dir = root / cfg.get("image_dir", "assets")
    image_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, spec in cfg["images"].items():
        dest = image_dir / f"{spec['id']}.jpg"
        if not dest.exists():
            url = cfg["image_url"].format(id=spec["id"])
            if console:
                console.print(f"  ⬇  {name} [dim]{url}[/dim]")
            urllib.request.urlretrieve(url, dest)
        paths[name] = dest
    return paths


def required_pairs(cfg):
    # Which (model, image) analyses each figure needs, so nothing is computed twice.
    pairs = set()
    for fig in cfg["figures"]:
        images = fig.get("images") or [fig["image"]]
        for alias in fig["models"]:
            for img in images:
                pairs.add((alias, img))
    return pairs


def _print_scores(cfg, results, console):
    # The numbers the README's support table quotes, printed by the same command that
    # draws the figures -- so nobody has to trust a hand-copied percentage.
    from rich.table import Table

    image_names = list(cfg["images"])
    table = Table(title="🩻 artifact score by model and image", title_style="bold",
                  header_style="bold cyan")
    table.add_column("model", style="bold")
    table.add_column("grid", justify="right")
    for name in image_names:
        table.add_column(name, justify="right")
    table.add_column("attention")

    for alias in cfg["models"]:
        row = [alias]
        first = results.get((alias, image_names[0]))
        if first is None:
            continue
        row.append("{}×{}".format(*first["grid"]))
        for img in image_names:
            res = results.get((alias, img))
            score = 100 * res["artifact"].get("score", 0.0) if res else 0.0
            row.append(f"[yellow]{score:.1f}%[/yellow]" if score else "[green]clean[/green]")
        row.append("—" if first["attention"] is None else "✓")
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()


def run(config_path, device="auto", console=None):
    from vit_xray.core import analyze, pick_device
    from vit_xray.plotting import build_figure, build_zoo_grid, save_figure

    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parent.parent

    image_paths = fetch_images(cfg, root, console)
    out_dir = root / cfg.get("out_dir", "docs")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_device = pick_device(device)

    # Group by model. analyze() reloads weights every call, and the zoo holds two 4.3 GB
    # ViT-g checkpoints -- doing all of one model's images together at least keeps that
    # file in the OS page cache instead of re-reading it from disk between models.
    pairs = required_pairs(cfg)
    results = {}
    for alias in cfg["models"]:
        wanted = sorted(img for a, img in pairs if a == alias)
        if not wanted:
            continue
        model_name = cfg["models"][alias]
        for img in wanted:
            if console:
                console.print(f"  [cyan]{alias}[/cyan] · {img}")
            results[(alias, img)] = analyze(model_name, image_paths[img], torch_device, -1)

    if console:
        _print_scores(cfg, results, console)

    written = []
    for fig_spec in cfg["figures"]:
        name = fig_spec["name"]
        models = fig_spec["models"]

        if fig_spec["type"] == "zoo_grid":
            images = fig_spec["images"]
            caption = "patch-token L2 norm at the final block · brighter is a higher-norm token"
            fig = build_zoo_grid(results, models, images, caption)
        else:
            img = fig_spec["image"]
            rows = [results[(m, img)] for m in models]
            note = cfg["images"][img].get("note", "")
            caption = f"{img}: {note}"
            fig = build_figure(rows, caption,
                               fig_spec.get("norm_scale", "per-image"),
                               fig_spec.get("histogram", False))

        save_figure(fig, out_dir / name)
        written.append(out_dir / name)
        if console:
            console.print(f"  [green]✓[/green] {name}")
    return written
