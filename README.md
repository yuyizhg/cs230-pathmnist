# Colorectal Histopathology Classification with CNNs: A Multi-Resolution Study on PathMNIST

CS230 (Deep Learning), Stanford University — course project
Team: Ximing Gao, Yuyi Zhang

## Overview
We classify colorectal tissue patches into 9 tissue types and compare:
- a baseline CNN (conv–ReLU–pool blocks + softmax classifier)
- ResNet-18 trained from scratch
- ResNet-18 with ImageNet-pretrained weights

Each model is trained at 64×64, 128×128, and 224×224 to test whether lower
resolutions can reach accuracy comparable to 224×224 at a fraction of the compute.

## Dataset
[PathMNIST](https://medmnist.com/) from MedMNIST, derived from Kather et al. (2019).
- 107,180 patches, 9 classes (CC BY 4.0)
- Official splits: 89,996 train / 10,004 val / 7,180 test
- The test set comes from a different clinical center, so evaluation also probes
  robustness to staining and imaging differences.

## Evaluation
- Primary metric: macro-F1
- Also: accuracy, per-class precision/recall, training and inference runtime
- Validation split for model selection; the test set stays untouched until all
  experimental choices are final.

## Repository layout
- `00_gpu_check.ipynb`: environment and GPU check
- Data and model checkpoints are kept outside the repo and are not tracked by Git.

## Environment
- AWS SageMaker notebook instance (`ml.g4dn.xlarge`, NVIDIA T4), `conda_pytorch` kernel
- Google Colab for lighter experiments
- Install the dataset package with `pip install medmnist`

## Status
In progress.

## References
- Kather et al. Predicting survival from colorectal cancer histology slides using deep learning. *PLOS Medicine*, 2019.
- Yang et al. MedMNIST v2. *Scientific Data*, 2023.
- He et al. Deep residual learning for image recognition. *CVPR*, 2016.