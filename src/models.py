"""Model definitions.

All models return raw logits. nn.CrossEntropyLoss applies the softmax
internally, the same idea as from_logits=True in Keras.
"""
import torch.nn as nn
from torchvision import models as tvm

from .data import NUM_CLASSES


def conv_block(c_in, c_out):
    """Conv(3x3) -> BatchNorm -> ReLU -> MaxPool(2).

    padding=1 keeps height and width unchanged through the conv
    (Keras padding='same'), so only the max-pool halves the size.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),  # bias is redundant before BatchNorm
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """Stacked conv-ReLU-pool blocks, global average pooling, linear classifier.

    Global average pooling (instead of Flatten -> Dense) averages each feature
    map down to one number, so the same model runs unchanged at 64, 128 and
    224. Feature map sizes with 4 blocks: 64 -> 4x4, 128 -> 8x8, 224 -> 14x14.
    """

    def __init__(self, num_classes=NUM_CLASSES, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        blocks, c_in = [], 3
        for w in widths:
            blocks.append(conv_block(c_in, w))
            c_in = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def resnet18(num_classes=NUM_CLASSES, pretrained=False):
    """Standard torchvision ResNet-18 with a new 9-class final layer.

    pretrained=True loads ImageNet weights for every layer except the new
    final layer, which starts from random initialization.
    """
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name, num_classes=NUM_CLASSES):
    if name == "baseline_cnn":
        return BaselineCNN(num_classes)
    if name == "resnet18_scratch":
        return resnet18(num_classes, pretrained=False)
    if name == "resnet18_pretrained":
        return resnet18(num_classes, pretrained=True)
    raise ValueError(f"Unknown model name: {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
