"""Tables and plots built from the results JSON files written by run_experiment()."""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import CLASS_NAMES
from .engine import RESULTS_DIR


def load_results(names=None, results_dir=RESULTS_DIR):
    """Load results by run name, or every results file if names is None."""
    if names is None:
        paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    else:
        paths = [os.path.join(results_dir, f"{n}.json") for n in names]
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def results_table(results):
    rows = []
    for r in results:
        cfg = r["config"]
        best = r.get("best_val", {})
        rows.append({
            "run": r["name"],
            "model": cfg["model"],
            "data": cfg["data_size"],
            "net input": cfg["input_size"] or cfg["data_size"],
            "params (M)": round(r["params"] / 1e6, 2),
            "best epoch": r.get("best_epoch"),
            "epochs run": r.get("epochs_run", len(r["history"])),
            "val acc": round(best.get("accuracy", float("nan")), 4),
            "val macro-F1": round(best.get("macro_f1", float("nan")), 4),
            "train s/epoch": round(r.get("mean_train_seconds_per_epoch", float("nan")), 1),
            "inference img/s": round(best.get("images_per_sec", float("nan"))),
            "status": r["status"],
        })
    return pd.DataFrame(rows).set_index("run")


def plot_history(results):
    """Train vs. validation loss and macro-F1 per epoch, one row per run.

    A large gap between train and val curves points to variance (overfitting);
    both curves plateauing low points to bias (underfitting).
    """
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 3.2 * len(results)), squeeze=False)
    for row, r in zip(axes, results):
        h = pd.DataFrame(r["history"])
        row[0].plot(h["epoch"], h["train_loss"], label="train")
        row[0].plot(h["epoch"], h["val_loss"], label="val")
        row[0].set_title(f"{r['name']}: loss")
        row[1].plot(h["epoch"], h["train_f1"], label="train")
        row[1].plot(h["epoch"], h["val_f1"], label="val")
        row[1].set_title(f"{r['name']}: macro-F1")
        if r.get("best_epoch"):
            for ax in row:
                ax.axvline(r["best_epoch"], color="gray", linestyle=":", linewidth=1)
        for ax in row:
            ax.set_xlabel("epoch")
            ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion(result, normalize=True):
    """Validation confusion matrix of the best checkpoint.

    normalize=True divides each row by its total, so the diagonal is each
    class's recall and off-diagonal cells show where that class gets confused.
    """
    cm = np.array(result["best_val"]["confusion_matrix"], dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, val, ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.5 * cm.max() else "black")
    ax.set_title(f"{result['name']}: validation confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


def per_class_table(result):
    best = result["best_val"]
    return pd.DataFrame(
        {"precision": best["precision"], "recall": best["recall"], "F1": best["f1"]},
        index=CLASS_NAMES,
    ).round(3)
