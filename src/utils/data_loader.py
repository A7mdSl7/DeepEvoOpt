'''Data loader & preprocessing utilities'''

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(dataset_name="fashion-mnist",
                    data_dir="data/raw",
                    batch_size=64,
                    val_split=0.15,
                    test_split=0.15,
                    num_workers=2,
                    resize=None,
                    seed=42):
    """
    Returns: train_loader, val_loader, test_loader, classes
    """
    set_seed(seed)

    os.makedirs(data_dir, exist_ok=True)

    # transforms: normalize to mean/std recommended for grayscale
    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # scale to [-1,1]
    ]
    transform = transforms.Compose(transform_list)

    if dataset_name.lower() in ["fashion-mnist", "fashion_mnist"]:
        full_train = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform)
        classes = full_train.classes
    elif dataset_name.lower() in ["mnist", "mnist"]:
        full_train = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform)
        classes = full_train.classes
    else:
        raise ValueError(
            "Unsupported dataset. Use 'fashion-mnist' or 'mnist' for now.")

    # create train/val split from full_train
    n_total = len(full_train)
    n_test = len(test_set)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_set, val_set = random_split(full_train, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(seed))

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, classes
