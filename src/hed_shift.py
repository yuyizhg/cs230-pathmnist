"""Stain-shifted (HED) validation sets, and evaluation of saved checkpoints on them.

Put this file in cs230-pathmnist/src/ and use it from a notebook:

    from src.hed_shift import make_val_versions, show_grid, evaluate_runs

    versions, labels = make_val_versions(sigmas=(0.15, 0.30))
    show_grid(*versions.values())      # look at the images before trusting numbers
    table = evaluate_runs(RUNS, versions, labels)
    table

Needs scikit-image:  pip install scikit-image
No training happens here; it only loads the saved best-epoch checkpoints.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.color import hed2rgb, rgb2hed
from torch.utils.data import DataLoader

from .data import DATA_ROOT, GPUPreprocess, PathMNISTArrays, load_split
from .engine import CKPT_DIR, run_epoch
from .models import build_model


def hed_perturb(images, sigma=0.15, seed=0, chunk=1000):
    """Return a stain-shifted uint8 copy of `images` (N, H, W, 3, uint8).

    Each image is unmixed into its stain channels (H = hematoxylin,
    E = eosin), each stain amount is scaled independently, and the
    result is mixed back into RGB:
        stain' = alpha * stain,   alpha ~ U(1 - sigma, 1 + sigma)
    so sigma=0.15 means each stain gets up to 15% more or less dye.

    Only scaling, no additive shift: skimage's stain values are tiny
    (about 0.001-0.07), so even a small added constant tints white
    background green or teal, which no real slide looks like.

    The same seed always gives every image the same change, so every
    model is evaluated on identical shifted images.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    alpha = rng.uniform(1 - sigma, 1 + sigma, size=(n, 1, 1, 3))
    # Third channel is DAB, a stain H&E slides don't use: leave it unchanged.
    alpha[..., 2] = 1.0

    out = np.empty_like(images)
    for s in range(0, n, chunk):          # chunks keep memory use small
        hed = rgb2hed(images[s:s + chunk]) * alpha[s:s + chunk]
        rgb = np.clip(hed2rgb(hed), 0, 1)
        out[s:s + chunk] = (rgb * 255).round().astype(np.uint8)
    return out


def make_val_versions(sigmas=(0.15, 0.30), size=64, seed=0, root=DATA_ROOT):
    """Clean val images plus one HED-shifted copy per sigma.

    Returns ({"clean": imgs, "hed_0.05": imgs, ...}, labels).
    """
    images, labels = load_split(root, size, "val")
    images = np.asarray(images)
    versions = {"clean": images}
    for s in sigmas:
        versions[f"hed_{s:.2f}"] = hed_perturb(images, sigma=s, seed=seed)
    return versions, labels


def show_grid(*versions, n=8, seed=0):
    """Show the same n images side by side, one row per version."""
    idx = np.random.default_rng(seed).choice(len(versions[0]), n, replace=False)
    fig, axes = plt.subplots(len(versions), n, figsize=(1.6 * n, 1.7 * len(versions)))
    axes = np.atleast_2d(axes)
    for r, imgs in enumerate(versions):
        for c, i in enumerate(idx):
            axes[r, c].imshow(imgs[i])
            axes[r, c].axis("off")
    plt.tight_layout()
    return fig


def evaluate_runs(runs, versions, labels, batch_size=256, ckpt_dir=CKPT_DIR):
    """Load each run's best checkpoint and report val macro-F1 on every version.

    Uses the same preprocessing as training (including the 64 -> 224 resize
    for the upsampling run), read from the config saved in the checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in runs:
        ckpt = torch.load(os.path.join(ckpt_dir, f"{name}.pt"), map_location=device)
        cfg = ckpt["config"]
        net = build_model(cfg["model"]).to(device)
        net.load_state_dict(ckpt["model_state"])
        prep = GPUPreprocess(device, out_size=cfg["input_size"])

        row = {"run": name, "model": cfg["model"], "input": cfg["input_size"] or cfg["data_size"],
               "train HED": cfg.get("hed_sigma", 0.0), "seed": cfg["seed"], "best epoch": ckpt["epoch"]}
        for vname, imgs in versions.items():
            loader = DataLoader(PathMNISTArrays(imgs, labels), batch_size=batch_size,
                                shuffle=False, num_workers=2)
            m = run_epoch(net, loader, prep, device, train=False)
            row[f"{vname} F1"] = round(m["macro_f1"], 4)
        rows.append(row)
        print(f"done: {name}")

    table = pd.DataFrame(rows).set_index("run")
    for vname in versions:
        if vname != "clean":
            table[f"{vname} drop"] = (table["clean F1"] - table[f"{vname} F1"]).round(4)
    return table


def summarize_seeds(table):
    """Mean +/- standard deviation over seeds, one row per (model, input, train HED)."""
    cols = [c for c in table.columns if c.endswith(" F1") or c.endswith(" drop")]
    g = table.groupby(["model", "input", "train HED"])[cols]
    mean, std, n = g.mean(), g.std(), g.size()
    out = pd.DataFrame(index=mean.index)
    out["seeds"] = n
    for c in cols:
        out[c] = [f"{m:.4f}" if np.isnan(s) else f"{m:.4f} ± {s:.4f}"
                  for m, s in zip(mean[c], std[c])]
    return out
