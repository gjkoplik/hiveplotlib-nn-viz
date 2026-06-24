"""MNIST / Fashion-MNIST loading and the frozen evaluation sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_DATASETS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}


def load_tensors(
    name: str = "mnist",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a dataset fully into memory: ``(train_x, train_y, test_x, test_y)``.

    Images are flattened to ``[n, 784]`` and standard-normalized.
    """
    if name not in _DATASETS:
        msg = f"unknown dataset {name!r}; expected one of {sorted(_DATASETS)}"
        raise ValueError(msg)
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ctor = _DATASETS[name]
    train = ctor(DATA_DIR, train=True, download=True, transform=tfm)
    test = ctor(DATA_DIR, train=False, download=True, transform=tfm)
    train_x, train_y = _stack(train)
    test_x, test_y = _stack(test)
    return train_x, train_y, test_x, test_y


def _stack(ds: torch.utils.data.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate an entire dataset into ``([n, 784], [n])`` tensors."""
    loader = DataLoader(ds, batch_size=len(ds))  # type: ignore[arg-type]
    x, y = next(iter(loader))
    return x.view(x.shape[0], -1), y


def train_loader(
    train_x: torch.Tensor, train_y: torch.Tensor, batch_size: int, seed: int
) -> DataLoader:
    """Shuffled DataLoader over the training tensors with a seeded generator."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )


def frozen_sample(test_y: torch.Tensor, per_class: int, seed: int) -> np.ndarray:
    """Pick a class-balanced, fixed sample of test indices for the P2CP mapping.

    Returns up to ``per_class * 10`` sorted indices (fewer if a class is short).
    """
    rng = np.random.default_rng(seed)
    y = test_y.numpy()
    picks = []
    for c in range(10):
        idx = np.where(y == c)[0]
        take = min(per_class, idx.size)
        picks.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(picks))
