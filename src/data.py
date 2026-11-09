"""PathMNIST data loading.

Reads the MedMNIST .npz files directly with numpy (no medmnist package needed
for training). Images stay uint8 on the CPU side; conversion to float,
augmentation, normalization and optional resizing happen on the GPU in
`GPUPreprocess`, which keeps the 4-vCPU instance from becoming the bottleneck.
"""
import functools
import math
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

# H&E stain matrix (Ruifrok & Johnston, 2001): rows are the RGB absorbance of
# hematoxylin, eosin and DAB. Same values scikit-image uses in rgb2hed/hed2rgb,
# so the training augmentation below matches src/hed_shift.py exactly.
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]])
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
_LOG_ADJUST = math.log(1e-6)


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


def hed_jitter(x, sigma, hed_from_rgb, rgb_from_hed, alpha=None):
    """Random stain augmentation on a GPU batch, x: float [N, 3, H, W] in [0, 1].

    Unmixes each image into hematoxylin (H) and eosin (E) amounts, scales each
    by its own random factor in [1 - sigma, 1 + sigma], and mixes back to RGB.
    This is the GPU version of hed_perturb() in src/hed_shift.py (same math as
    scikit-image's rgb2hed -> scale -> hed2rgb). Pass `alpha` [N or 1, 3, 1, 1]
    to use fixed factors instead of random ones (used for the sanity check).
    """
    od = torch.log(x.clamp(min=1e-6)) / _LOG_ADJUST
    hed = torch.einsum("nchw,cd->ndhw", od, hed_from_rgb).clamp(min=0)
    if alpha is None:
        alpha = 1 + (torch.rand(x.shape[0], 3, 1, 1, device=x.device) * 2 - 1) * sigma
        alpha[:, 2] = 1.0  # DAB channel: H&E slides have no DAB, leave it alone
    hed = hed * alpha
    rgb = torch.exp(-torch.einsum("nchw,cd->ndhw", hed * -_LOG_ADJUST, rgb_from_hed))
    return rgb.clamp(0, 1)


class GPUPreprocess:
    """Per-batch preprocessing on the GPU.

    uint8 [0, 255] -> float [0, 1] -> optional HED stain jitter (training only)
    -> random flips (training only)
    -> optional bilinear resize (e.g. 64 -> 224 for the upsampling experiment)
    -> per-channel normalization with ImageNet statistics.
    Validation batches get the same steps minus the augmentation (deterministic).

    hed_sigma=0 (default) turns stain augmentation off and draws no extra random
    numbers, so earlier runs reproduce exactly.
    """

    def __init__(self, device, out_size=None, hed_sigma=0.0):
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.out_size = out_size
        self.hed_sigma = hed_sigma
        self.hed_from_rgb = torch.tensor(HED_FROM_RGB, dtype=torch.float32, device=device)
        self.rgb_from_hed = torch.tensor(RGB_FROM_HED, dtype=torch.float32, device=device)

    def __call__(self, x, train):
        x = x.float().div_(255)
        if train:
            if self.hed_sigma > 0:
                x = hed_jitter(x, self.hed_sigma, self.hed_from_rgb, self.rgb_from_hed)
            x = random_flips(x)
        if self.out_size is not None and x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std
