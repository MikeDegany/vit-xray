# vit-xray -- MIT License, Copyright (c) 2026 Mike Degany <mike.degany@unt.edu>
"""Constants that are worth reading without dragging torch in behind them.

cli.py needs DEFAULT_MODEL at import time to build its --help text, and reaching into
core.py for it would cost ~6 seconds of timm import before printing a help page.
"""

# Two different jobs, so two different models.
#
# The default has to fit on whatever laptop someone just ran `pip install vit-xray` on.
# OpenCLIP ViT-B/16 is 329 MB, ~6 s and ~1.4 GB of RAM on CPU, and still shows the
# phenomenon clearly (3.6% of tokens on the test images). It is also the OpenCLIP from
# Fig 2 of the registers paper, so it is a fair thing to be looking at.
DEFAULT_MODEL = "vit_base_patch16_clip_224.laion2b"

# The showcase is what the README gallery uses and what the CLI points people at. ViT-g
# is the only DINOv2 size still showing artifacts at the final block -- 9.4x max/median,
# against 1.2x for both ViT-B and ViT-L -- and its 37x37 grid makes a far better picture
# than the default's 14x14. It costs 4.3 GB on disk and ~9.6 GB of RAM on CPU, which is
# exactly why it isn't the default.
SHOWCASE_MODEL = "vit_giant_patch14_dinov2.lvd142m"

# Above this much overlap the two fitted Gaussians aren't really separate populations.
ARTIFACT_OVERLAP_GATE = 0.02

# Separability alone isn't enough: two flat colour fields make a clean model look bimodal.
# Real artifacts are always a small minority of tokens, so cap it.
ARTIFACT_MAX_FRACTION = 0.10


# The comparison set that matters, in the order the README table lists them. DINO and MAE
# are the controls that must come out clean; the registers pair is the controlled result.
MODEL_ZOO = [
    "vit_base_patch16_224.dino",
    "vit_giant_patch14_dinov2.lvd142m",
    "vit_giant_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
    "vit_base_patch16_siglip_224.webli",
    "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "vit_base_patch16_224.mae",
]
