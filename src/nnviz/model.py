"""Tiny MLP for MNIST, with per-layer activation capture."""

from __future__ import annotations

import torch
from torch import nn

# The three trainable layers, in order. These double as the hive-plot axes.
LAYER_NAMES = ("hidden1", "hidden2", "output")


class MLP(nn.Module):
    """A small MLP: ``784 -> hidden1 -> hidden2 -> 10`` with ReLU activations."""

    def __init__(self, hidden1: int = 64, hidden2: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return output logits for a batch of images."""
        return self.activations(x)["output"]

    def activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Post-ReLU activations for both hidden layers plus the output logits.

        Keys: ``"hidden1"`` (size ``hidden1``), ``"hidden2"`` (size ``hidden2``),
        ``"output"`` (size 10, pre-softmax logits).
        """
        x = x.view(x.shape[0], -1)
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        out = self.fc3(h2)
        return {"hidden1": h1, "hidden2": h2, "output": out}
