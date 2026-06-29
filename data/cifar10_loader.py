"""CIFAR-10 loader with Dirichlet non-IID partitioning across clients."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar10_transforms():
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return train, test


def dirichlet_partition(dataset, num_clients, alpha, seed=42):
    """
    Partition dataset indices across clients using Dirichlet(alpha).
    Low alpha = extreme non-IID; high alpha ~ IID.
    Returns list of index arrays, one per client.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    for c_idx in class_indices:
        rng.shuffle(c_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(c_idx)).astype(int)
        # fix rounding so all samples are assigned
        proportions[-1] = len(c_idx) - proportions[:-1].sum()
        splits = np.split(c_idx, np.cumsum(proportions[:-1]))
        for i, s in enumerate(splits):
            client_indices[i].extend(s.tolist())

    return client_indices


def build_client_loaders(alpha, num_clients, batch_size, data_root="./data/cache"):
    train_tf, test_tf = get_cifar10_transforms()
    train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)

    partitions = dirichlet_partition(train_ds, num_clients, alpha)

    client_loaders = [
        DataLoader(Subset(train_ds, idxs), batch_size=batch_size, shuffle=True, num_workers=2)
        for idxs in partitions
    ]
    # shared validation loader (used for contribution scoring)
    val_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    return client_loaders, val_loader, partitions
