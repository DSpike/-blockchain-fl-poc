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


def dirichlet_partition(dataset, num_clients, alpha, seed=42,
                        targets_override=None):
    """
    Partition dataset indices across clients using Dirichlet(alpha).
    Low alpha = extreme non-IID; high alpha ~ IID.
    Returns list of index arrays, one per client.
    targets_override: pass a numpy array of labels when dataset.targets is
                      unavailable (e.g. torchvision SVHN uses .labels).
    """
    rng = np.random.default_rng(seed)
    targets = (np.array(targets_override) if targets_override is not None
               else np.array(dataset.targets))
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


def build_client_loaders(alpha, num_clients, batch_size, data_root="./data/cache",
                         val_size=2000, num_workers=2):
    train_tf, test_tf = get_cifar10_transforms()
    train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)

    partitions = dirichlet_partition(train_ds, num_clients, alpha)

    pw = num_workers > 0
    client_loaders = [
        DataLoader(Subset(train_ds, idxs), batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, persistent_workers=pw)
        for idxs in partitions
    ]
    # fixed subset for contribution scoring — faster per-round evaluation
    val_indices = list(range(val_size))
    val_loader = DataLoader(Subset(test_ds, val_indices), batch_size=256,
                            shuffle=False, num_workers=num_workers, pin_memory=True,
                            persistent_workers=pw)
    # full test loader for final accuracy reporting
    full_val_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                                 num_workers=num_workers, pin_memory=True,
                                 persistent_workers=pw)

    targets = np.array(train_ds.targets)
    return client_loaders, val_loader, partitions, full_val_loader, targets
