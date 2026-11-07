"""PathMNIST data loading.

Reads the MedMNIST .npz files directly with numpy (no medmnist package needed
for training). Images stay uint8 on the CPU side; conversion to float,
augmentation, normalization and optional resizing happen on the GPU in
`GPUPreprocess`, which keeps the 4-vCPU instance from becoming the bottleneck.
"""
import functools
import os
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA_ROOT = "/home/ec2-user/SageMaker/data"

CLASS_NAMES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet statistics, which the pretrained ResNet-18 weights expect.
# Every model uses them so all runs see identically normalized inputs.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def npz_path(root, size):
    return os.path.join(root, f"pathmnist_{size}.npz")


def extract_npz(root, size):
    """Unzip pathmnist_{size}.npz into a folder of .npy files (only once).

    An .npz file is a zip of .npy files. Extracting them lets us memory-map
    the arrays, so the 224x224 training set (~13.5 GB) never has to fit in RAM.
    A marker file records that extraction finished, so an interrupted run
    gets redone instead of leaving half-written files.
    """
    out_dir = os.path.join(root, f"pathmnist_{size}")
    marker = os.path.join(out_dir, "_extracted")
    if not os.path.exists(marker):
        os.makedirs(out_dir, exist_ok=True)
        print(f"Extracting {npz_path(root, size)} -> {out_dir} (one time) ...")
        with zipfile.ZipFile(npz_path(root, size)) as zf:
            zf.extractall(out_dir)
        open(marker, "w").close()
    return out_dir


def load_split(root, size, split, mmap=None, allow_test=False):
    """Return (images uint8 [N, H, W, 3], labels int64 [N]).

    mmap=None picks automatically: in-memory for 64x64, memory-mapped for 128+.
    The test split is locked unless allow_test=True, so it stays untouched
    until all experimental choices are final (as the proposal requires).
    Results are cached, so several experiments in one session load data once.
    """
    if split == "test" and not allow_test:
        raise ValueError(
            "The test split is locked until final evaluation. "
            "Pass allow_test=True only for the final, pre-registered runs."
        )
    if mmap is None:
        mmap = size >= 128
    return _load_split_cached(root, int(size), split, bool(mmap))


@functools.lru_cache(maxsize=None)
def _load_split_cached(root, size, split, mmap):
    if mmap:
        folder = extract_npz(root, size)
        images = np.load(os.path.join(folder, f"{split}_images.npy"), mmap_mode="r")
        labels = np.load(os.path.join(folder, f"{split}_labels.npy"))
    else:
        with np.load(npz_path(root, size)) as f:
            images = f[f"{split}_images"]
            labels = f[f"{split}_labels"]
    return images, labels.reshape(-1).astype(np.int64)


class PathMNISTArrays(Dataset):
    """Wraps image/label arrays. Returns uint8 CHW tensors (like ToTensor without the /255)."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = torch.from_numpy(np.array(self.images[i]))  # copy; also works for memory-mapped arrays
        return img.permute(2, 0, 1), self.labels[i]  # HWC -> CHW


def make_loaders(size, batch_size=128, num_workers=4, root=DATA_ROOT, mmap=None):
    """Train and validation DataLoaders (Keras analogue: tf.data.Dataset(...).shuffle().batch())."""
    tr_x, tr_y = load_split(root, size, "train", mmap)
    va_x, va_y = load_split(root, size, "val", mmap)
    common = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    train_loader = DataLoader(
        PathMNISTArrays(tr_x, tr_y), batch_size=batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(
        PathMNISTArrays(va_x, va_y), batch_size=batch_size * 2, shuffle=False, **common
    )
    return train_loader, val_loader


def random_flips(x):
    """Independent random horizontal and vertical flip for each image in the batch."""
    n = x.shape[0]
    h = torch.rand(n, 1, 1, 1, device=x.device) < 0.5
    x = torch.where(h, x.flip(3), x)
    v = torch.rand(n, 1, 1, 1, device=x.device) < 0.5
    return torch.where(v, x.flip(2), x)


class GPUPreprocess:
    """Per-batch preprocessing on the GPU.

    uint8 [0, 255] -> float [0, 1] -> random flips (training only)
    -> optional bilinear resize (e.g. 64 -> 224 for the upsampling experiment)
    -> per-channel normalization with ImageNet statistics.
    Validation batches get the same steps minus the flips (deterministic).
    """

    def __init__(self, device, out_size=None):
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.out_size = out_size

    def __call__(self, x, train):
        x = x.float().div_(255)
        if train:
            x = random_flips(x)
        if self.out_size is not None and x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std
