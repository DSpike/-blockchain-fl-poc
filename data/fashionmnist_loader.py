"""FashionMNIST loader with Dirichlet non-IID partitioning across clients."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from data.cifar10_loader import dirichlet_partition


def get_fashionmnist_transforms():
    train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    return train, test


def build_client_loaders(alpha, num_clients, batch_size,
                         data_root="./data/cache", val_size=2000):
    train_tf, test_tf = get_fashionmnist_transforms()
    train_ds = datasets.FashionMNIST(data_root, train=True,
                                     download=True, transform=train_tf)
    test_ds  = datasets.FashionMNIST(data_root, train=False,
                                     download=True, transform=test_tf)

    partitions = dirichlet_partition(train_ds, num_clients, alpha)

    client_loaders = [
        DataLoader(Subset(train_ds, idxs), batch_size=batch_size,
                   shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        for idxs in partitions
    ]
    val_indices    = list(range(val_size))
    val_loader     = DataLoader(Subset(test_ds, val_indices), batch_size=256,
                                shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    full_val_loader = DataLoader(test_ds, batch_size=256,
                                 shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    targets = np.array(train_ds.targets)
    return client_loaders, val_loader, partitions, full_val_loader, targets
