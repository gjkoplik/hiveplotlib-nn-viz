"""Compute the per-checkpoint statistics we log for each mapping."""

from __future__ import annotations

import numpy as np
import torch

from nnviz.model import LAYER_NAMES, MLP


@torch.no_grad()
def evaluate(
    model: MLP, x: torch.Tensor, y: torch.Tensor, batch_size: int = 2048
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run ``model`` over ``x`` in batches.

    Returns ``(acts, preds)`` where ``acts`` maps each layer name to a
    ``[n, n_neurons]`` array and ``preds`` is a ``[n]`` array of argmax predictions.
    """
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in LAYER_NAMES}
    preds = []
    for start in range(0, x.shape[0], batch_size):
        acts = model.activations(x[start : start + batch_size])
        for name in LAYER_NAMES:
            chunks[name].append(acts[name].cpu().numpy())
        preds.append(acts["output"].argmax(dim=1).cpu().numpy())
    acts_out = {name: np.concatenate(chunks[name]) for name in LAYER_NAMES}
    return acts_out, np.concatenate(preds)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """10x10 confusion counts; rows are true class, columns predicted class."""
    m = np.zeros((10, 10), dtype=np.int64)
    np.add.at(m, (y_true, y_pred), 1)
    return m


def class_mean_activations(
    acts: dict[str, np.ndarray], y_true: np.ndarray
) -> dict[str, np.ndarray]:
    """Mean activation vector per class for each layer: ``[10, n_neurons]`` each."""
    out = {}
    for name, a in acts.items():
        means = np.zeros((10, a.shape[1]), dtype=np.float64)
        for c in range(10):
            mask = y_true == c
            if mask.any():
                means[c] = a[mask].mean(axis=0)
        out[name] = means
    return out
