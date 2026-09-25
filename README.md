<div align="center">

# 🩻 vit-xray

### **Look inside any vision transformer backbone in one command**

[![PyPI](https://img.shields.io/pypi/v/vit-xray?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/vit-xray/) [![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![timm](https://img.shields.io/badge/timm-any%20ViT-FFD21E?logo=huggingface&logoColor=black)](https://github.com/huggingface/pytorch-image-models) [![License](https://img.shields.io/badge/license-MIT-22c55e)](https://github.com/MikeDegany/vit-xray/blob/main/LICENSE)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2309.16588-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2309.16588) -->

```bash
vit-xray photo.jpg
```

<br>

<img src="https://raw.githubusercontent.com/MikeDegany/vit-xray/main/docs/hero.png" alt="Patch-token norm maps across seven backbones and four images" width="100%">

<sub><b>Patch-token norms at the final block, seven pretrained backbones.</b> Most of them quietly repurpose a couple of percent
of their patch tokens as scratch space; the bright dots, sitting in sky, walls and flat ground.<br>
DINO, MAE, and DINOv2-with-registers do not.</sub>

</div>

---

## ⚡ Quickstart

```bash
conda create -n vitxray python=3.11
conda activate vitxray
pip install vit-xray

vit-xray photo.jpg
```

> [!TIP]
> **No GPU? No problem.** The default backbone is a 329 MB CLIP. The whole command takes about
> ten seconds on CPU and peaks around 1.5 GB of RAM; it works on a laptop.

> [!NOTE]
> On Linux, `pip` pulls the CUDA build of PyTorch by default, which is several GB. On a machine
> without a GPU, install the CPU build first and `vit-xray` will use it:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

For the clearest picture, use the model the figures above lead with:

```bash
vit-xray photo.jpg --model vit_giant_patch14_dinov2.lvd142m --all-panels
```

That's DINOv2 ViT-g/14, a 37×37 grid instead of 14×14, and roughly twenty outlier tokens instead
of seven. It costs a 4.3 GB download and ~10 GB of RAM on CPU, which is exactly why it isn't the
default.

---

## 🔍 What am I looking at

Vision transformers cut an image into patches and give each patch a token. You'd expect every
token to describe its own patch. Mostly they do, but in a trained ViT, a handful of tokens in
boring regions get quietly repurposed.

The model notices that a patch of empty sky looks exactly like its neighbours, decides it has
nothing worth storing, and reuses its token as scratch space for global information about the
whole image. Those tokens end up with an **L2 norm about ten times larger** than everything around
them, and they no longer describe the patch they came from.

That's the phenomenon Darcet et al. named and measured in
**[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)** (ICLR 2024): roughly
2% of tokens, only in large enough models trained long enough, and fixable by giving the model
dedicated scratch tokens *registers* so it stops stealing patch tokens for the job.

`vit-xray` gives you three views of it, plus a score:

<table>
<tr>
<td width="25%" valign="top">

**📊 Patch-norm map**

L2 norm of every patch token. Artifacts are isolated bright dots.

</td>
<td width="25%" valign="top">

**🎨 Feature PCA**

Patch tokens projected to 3 components as RGB, the pseudo-segmentation view.

</td>
<td width="25%" valign="top">

**👁 CLS attention**

What the class token attends to. Artifacts show up as spikes.

</td>
<td width="25%" valign="top">

**🩻 Artifact score**

Fraction of tokens in the high-norm population.

</td>
</tr>
</table>

The score fits a two-component Gaussian mixture to the norms and thresholds at the crossover.
**There is no hardcoded cutoff**: the paper's own was hand-picked for one model, and it says
outright that it varies. If the norms aren't meaningfully bimodal, the score is zero and the
backbone is reported clean.

<br>

### 🎯 Registers fix it

Same architecture, same width, same training data, one shared colour scale. Registers are the
only difference:

<div align="center">
<img src="https://raw.githubusercontent.com/MikeDegany/vit-xray/main/docs/registers.png" alt="DINOv2 ViT-g with and without register tokens" width="100%">
</div>

> [!NOTE]
> The **attention** panel is the part to look at. Without registers it spikes on the artifacts;
> with them it traces the cats.

<br>

### 📈 The norm distribution

<div align="center">
<img src="https://raw.githubusercontent.com/MikeDegany/vit-xray/main/docs/histograms.png" alt="Token-norm histograms with fitted thresholds" width="100%">
</div>

A backbone with artifacts has a visibly bimodal norm distribution, a bulk, a gap, and a small
high-norm cluster. That gap is where the fitted threshold lands. DINO has no gap.

---

## 💡 Why you should care

If you're freezing a backbone and reading its patch tokens, **semantic segmentation, monocular
depth, feature fields, open-vocabulary detection, robotic semantic mapping**, then artifact
tokens are corrupted inputs to your dense head. They carry global information, not local, so
whatever sits under them is described wrongly.

Darcet et al. show this measurably hurts dense prediction and breaks unsupervised object
discovery: DINOv2 was *worse* than supervised baselines with LOST until registers were added.

It costs one command to find out which kind of backbone you have. If yours scores clean, ignore
all of this. If it doesn't, pick a register-trained variant, or look at the training-free
mitigation in
**[Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010)**.

---

## 🧠 Model support

Measured by `vit-xray --gallery configs/gallery.yaml`, which prints this table as it runs.
Percentages are the artifact score on each of the four gallery images.

| model | size | grid | cats | skaters | frisbee | bathroom | attn |
|:---|---:|---:|---:|---:|---:|---:|:---:|
| `vit_base_patch16_224.dino` | 328 MB | 14×14 | 🟢 clean | 🟢 clean | 🟢 clean | 🟢 clean | ✅ |
| `vit_giant_patch14_dinov2.lvd142m` | 4.3 GB | 37×37 | 🟡 1.7% | 🟡 1.8% | 🟡 1.9% | 🟡 1.6% | ✅ |
| `vit_giant_patch14_reg4_dinov2.lvd142m` | 4.3 GB | 37×37 | 🟢 clean | 🟢 clean | 🟢 clean | 🟢 clean | ✅ |
| `vit_base_patch16_clip_224.laion2b` | 329 MB | 14×14 | 🟡 3.6% | 🟡 4.6% | 🟡 4.6% | 🟡 3.1% | ✅ |
| `vit_base_patch16_siglip_224.webli` | 355 MB | 14×14 | 🟡 1.0% | 🟡 1.0% | 🟡 1.0% | 🟡 1.5% | — |
| `deit3_base_patch16_224.fb_in22k_ft_in1k` | 331 MB | 14×14 | 🟡 2.0% | 🟢 clean | 🟡 5.1% | 🟡 2.0% | ✅ |
| `vit_base_patch16_224.mae` | 328 MB | 14×14 | 🟢 clean | 🟢 clean | 🟢 clean | 🟢 clean | ✅ |

Any [timm](https://github.com/huggingface/pytorch-image-models) ViT works, not just these. The
patch grid and prefix-token count are read from the model, never hardcoded, so
DINOv2-with-registers (5 prefix tokens) and plain ViTs (1) both come out right.

<details>
<summary><b>Two honest caveats in that table</b></summary>

<br>

**SigLIP has no CLS token.** It pools instead, so there's nothing to compute CLS attention
*from*. The other panels render and the attention panel says so rather than failing.

**DeiT3 reads `clean` on one image and 5.1% on another.** Its artifacts are real and stable, the
same four border tokens sit at ~1500 from block 3 onward, but its *bulk* norms inflate over the
last few blocks until they nearly reach the outliers, so separability at the final block depends
on the image. `--layer 6` shows it unambiguously.

</details>

---

## 🐍 Library

```python
from vit_xray import inspect

res = inspect("photo.jpg", model="vit_base_patch16_clip_224.laion2b")

res.norm_map        # (H, W)
res.pca_rgb         # (H, W, 3)
res.attention       # (H, W), or None if the model has no CLS token
res.artifact_score  # float
res.plot()          # matplotlib Figure
```

---

## 🎛 Useful flags

| command | what it does |
|:---|:---|
| `vit-xray photo.jpg --all-panels` | add the norm histogram |
| `vit-xray photo.jpg -m modelA -m modelB` | compare backbones, one row each |
| `vit-xray photo.jpg -m a -m b --norm-scale shared` | shared colour scale *(same-width models only)* |
| `vit-xray photo.jpg --layer 6` | read a mid-network block |
| `vit-xray photo.jpg --pca-foreground` | mask the PCA background |
| `vit-xray --gallery configs/gallery.yaml` | rebuild every figure in this README |

> [!WARNING]
> `--layer` matters more than it looks. **Artifacts can appear mid-network and be erased by the
> last block**, so a clean final-block reading is not always the same as no artifacts.

<details>
<summary><b>What one run actually puts on screen</b></summary>

<br>

<img src="https://raw.githubusercontent.com/MikeDegany/vit-xray/main/docs/panels.png" alt="All panels for a single model" width="100%">

</details>

---

## 🔗 Complementary tools

`vit-xray` answers one question: **are this backbone's patch features clean?** For other kinds of
looking-inside, use these instead, complementary, and deliberately not reimplemented here:

| tool | what it's for |
|:---|:---|
| **[vit-explain](https://github.com/jacobgil/vit-explain)** | Attention rollout and gradient rollout, aggregating attention across *all* layers rather than reading one. |
| **[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)** | Grad-CAM and the wider family of 2D class-activation methods, *why this prediction*, rather than *are these features clean*. |

---

## 🔁 Reproducing the figures

The gallery config lives in the repo, so this one needs a checkout:

```bash
git clone https://github.com/MikeDegany/vit-xray && cd vit-xray
pip install -e ".[dev]"
vit-xray --gallery configs/gallery.yaml
```

Downloads the four COCO val2017 images into `assets/` (gitignored), runs the zoo, and rewrites
everything in `docs/`. Images are chosen for large low-information regions, sky, walls, flat
ground, because that's where artifacts concentrate. Output is deterministic: same figures,
pixel for pixel, every run.

<details>
<summary><b>Citation</b></summary>

<br>

```bibtex
@inproceedings{darcet2024vision,
  title     = {Vision Transformers Need Registers},
  author    = {Darcet, Timoth\'ee and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
```

</details>

---

<div align="center">
<sub>🩻 <b>vit-xray</b> · MIT licensed · built on <a href="https://github.com/huggingface/pytorch-image-models">timm</a></sub>
</div>
