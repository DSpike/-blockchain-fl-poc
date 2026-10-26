"""
SVHN (Street View House Numbers) loader with Dirichlet non-IID partitioning.

SVHN: 73,257 training samples, 26,032 test, 10 digit classes (0-9),
      colour 32x32 images.  Uses ResNet8 (same as CIFAR-10).

Normalisation values match standard SVHN pre-processing:
  mean = (0.4377, 0.4438, 0.4728)
  std  = (0.1980, 0.2010, 0.1970)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from data.cifar10_loader import dirichlet_partition


def get_svhn_transforms():
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                             (0.1980, 0.2010, 0.1970)),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                             (0.1980, 0.2010, 0.1970)),
    ])
    return train, test


def build_client_loaders(alpha, num_clients, batch_size,
                         data_root="./data/cache", val_size=2000):
    train_tf, test_tf = get_svhn_transforms()

    train_ds = datasets.SVHN(data_root, split="train",
                              download=True, transform=train_tf)
    test_ds  = datasets.SVHN(data_root, split="test",
                              download=True, transform=test_tf)

    # SVHN stores labels as .labels (numpy array), not .targets
    all_labels = np.array(train_ds.labels, dtype=np.int64)
    partitions  = dirichlet_partition(train_ds, num_clients, alpha,
                                      targets_override=all_labels)

    client_loaders = [
        DataLoader(Subset(train_ds, idxs), batch_size=batch_size,
                   shuffle=True, num_workers=2, pin_memory=True,
                   persistent_workers=True)
        for idxs in partitions
    ]

    val_indices      = list(range(val_size))
    val_loader       = DataLoader(Subset(test_ds, val_indices), batch_size=256,
                                  shuffle=False, num_workers=2, pin_memory=True,
                                  persistent_workers=True)
    full_val_loader  = DataLoader(test_ds, batch_size=256,
                                  shuffle=False, num_workers=2, pin_memory=True,
                                  persistent_workers=True)

    return client_loaders, val_loader, partitions, full_val_loader, all_labels
