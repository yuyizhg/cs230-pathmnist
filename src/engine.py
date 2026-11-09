"""Training and evaluation.

run_experiment() is the one function the notebooks call. It trains one model
on one resolution, tracks train/val metrics every epoch, saves the best
checkpoint (by validation macro-F1), stops early when validation stalls, and
writes a results JSON that the report helpers can plot.
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .data import DATA_ROOT, NUM_CLASSES, GPUPreprocess, make_loaders
from .models import build_model, count_params

CKPT_DIR = "/home/ec2-user/SageMaker/checkpoints"  # outside the repo: weights are too big for Git
RESULTS_DIR = "results"  # inside the repo: small JSON files, worth committing


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def confusion_matrix(preds, targets, num_classes=NUM_CLASSES):
    """Rows = true class, columns = predicted class."""
    idx = targets * num_classes + preds
    return torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_cm(cm):
    """Accuracy, macro-F1, and per-class precision/recall/F1 from a confusion matrix.

    Macro-F1 is the unweighted mean of the per-class F1 scores (same definition
    as sklearn's f1_score(average="macro")), so every class counts equally.
    """
    cm = cm.double()
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp(min=1)
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        "accuracy": (tp.sum() / cm.sum()).item(),
        "macro_f1": f1.mean().item(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }


# ----------------------------------------------------------------------------
# One pass over the data
# ----------------------------------------------------------------------------

def run_epoch(model, loader, prep, device, train, optimizer=None, scaler=None, amp=True, limit_batches=None):
    """One pass over `loader`. Training if train=True, otherwise evaluation.

    Training-pass metrics are measured with augmentation and dropout active,
    so they slightly understate how well the model fits the training set.
    """
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    use_amp = amp and device.type == "cuda"
    total_loss = torch.zeros((), device=device)
    n_seen = 0
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    start = time.time()

    with torch.set_grad_enabled(train):
        for b, (x, y) in enumerate(loader):
            if limit_batches is not None and b >= limit_batches:
                break
            x = prep(x.to(device, non_blocking=True), train=train)
            y = y.to(device, non_blocking=True)

            # Mixed precision: runs most of the math in float16 on the T4's tensor cores (much faster).
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)

            if train:
                # PyTorch's version of GradientTape + apply_gradients:
                optimizer.zero_grad(set_to_none=True)  # clear last batch's gradients
                scaler.scale(loss).backward()          # backprop (scaled to avoid float16 underflow)
                scaler.step(optimizer)                 # Adam update
                scaler.update()

            total_loss += loss.detach() * y.size(0)
            n_seen += y.size(0)
            cm += confusion_matrix(logits.argmax(1), y)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    metrics = metrics_from_cm(cm.cpu())
    metrics["loss"] = (total_loss / max(n_seen, 1)).item()
    metrics["seconds"] = elapsed
    metrics["images_per_sec"] = n_seen / max(elapsed, 1e-9)
    metrics["confusion_matrix"] = cm.cpu().tolist()
    return metrics


# ----------------------------------------------------------------------------
# A full experiment
# ----------------------------------------------------------------------------

def run_experiment(
    name,
    model,
    data_size=64,
    input_size=None,
    hed_sigma=0.0,
    epochs=30,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,
    lr_decay_factor=0.3,
    lr_patience=2,
    early_stop_patience=6,
    seed=0,
    amp=True,
    num_workers=4,
    limit_batches=None,
    save=True,
    data_root=DATA_ROOT,
    ckpt_dir=CKPT_DIR,
    results_dir=RESULTS_DIR,
):
    """Train `model` ("baseline_cnn", "resnet18_scratch" or "resnet18_pretrained").

    data_size:   which PathMNIST file to read (64, 128 or 224).
    input_size:  size the network actually sees. None = same as data_size.
                 input_size=224 with data_size=64 is the upsampling experiment.
    hed_sigma:   stain augmentation strength for training images (0 = off).
                 0.2 means each image gets up to 20% more or less of each dye.
    lr decay:    learning rate is multiplied by lr_decay_factor whenever val
                 macro-F1 hasn't improved for lr_patience epochs.
    early stop:  training ends after early_stop_patience epochs without a
                 new best val macro-F1; the best checkpoint is kept.
    L2:          weight_decay adds an L2 penalty to Adam's gradients.
    limit_batches / save=False: quick smoke tests.
    """
    config = {k: v for k, v in locals().items()}
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(data_size, batch_size, num_workers, root=data_root)
    prep = GPUPreprocess(device, out_size=input_size, hed_sigma=hed_sigma)
    net = build_model(model).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_decay_factor, patience=lr_patience
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
    results_path = os.path.join(results_dir, f"{name}.json")

    result = {
        "name": name,
        "config": config,
        "params": count_params(net),
        "history": [],
        "status": "running",
    }
    seen = input_size or data_size
    print(f"=== {name}: {model}, data {data_size}x{data_size}, network input {seen}x{seen}, "
          f"HED aug {hed_sigma}, seed {seed}, {result['params'] / 1e6:.2f}M params, device {device} ===")

    best_f1, best_epoch, epochs_since_best = -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        tr = run_epoch(net, train_loader, prep, device, True, optimizer, scaler, amp, limit_batches)
        va = run_epoch(net, val_loader, prep, device, False, amp=amp, limit_batches=limit_batches)
        scheduler.step(va["macro_f1"])

        improved = va["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, epochs_since_best = va["macro_f1"], epoch, 0
            if save:
                torch.save({"model_state": net.state_dict(), "config": config, "epoch": epoch}, ckpt_path)
        else:
            epochs_since_best += 1

        result["history"].append({
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": tr["loss"], "train_acc": tr["accuracy"], "train_f1": tr["macro_f1"],
            "val_loss": va["loss"], "val_acc": va["accuracy"], "val_f1": va["macro_f1"],
            "train_seconds": tr["seconds"], "val_seconds": va["seconds"],
        })
        print(f"ep {epoch:02d} | lr {lr_now:.1e} | "
              f"train loss {tr['loss']:.3f} acc {tr['accuracy']:.3f} F1 {tr['macro_f1']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['accuracy']:.3f} F1 {va['macro_f1']:.3f} | "
              f"{tr['seconds'] + va['seconds']:.0f}s{'  *best' if improved else ''}")

        if save:  # write progress every epoch, so it survives a disconnected browser tab
            _write_json(result, results_path)
        if epochs_since_best >= early_stop_patience:
            print(f"Early stopping: no val improvement for {early_stop_patience} epochs.")
            break

    # Final validation pass with the best weights: per-class metrics, confusion matrix, inference speed.
    if save:
        net.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    best = run_epoch(net, val_loader, prep, device, False, amp=amp, limit_batches=limit_batches)
    hist = result["history"]
    result.update({
        "status": "done",
        "best_epoch": best_epoch,
        "best_val": best,
        "epochs_run": len(hist),
        "mean_train_seconds_per_epoch": float(np.mean([h["train_seconds"] for h in hist])),
        "total_train_minutes": sum(h["train_seconds"] + h["val_seconds"] for h in hist) / 60,
    })
    if save:
        _write_json(result, results_path)
    print(f"Best epoch {best_epoch}: val acc {best['accuracy']:.4f}, val macro-F1 {best['macro_f1']:.4f}, "
          f"{result['mean_train_seconds_per_epoch']:.0f}s/epoch train, "
          f"{best['images_per_sec']:.0f} img/s inference. Saved: {results_path if save else '(not saved)'}\n")
    return result


def _write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
