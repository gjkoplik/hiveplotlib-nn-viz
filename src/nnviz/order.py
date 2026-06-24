"""Freeze a meaningful, stable neuron ordering for the hive-plot axes."""

from __future__ import annotations

import numpy as np


def selectivity_order(
    class_activations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Order each layer's neurons by the class they respond to most strongly.

    Every neuron is assigned its argmax class over the per-class mean activations,
    then neurons are sorted by (preferred class ascending, strength descending).
    Computed once from the final trained model and frozen for all frames, so each
    digit's pathway occupies a contiguous arc and the layout never jumps.

    Returns a mapping from layer name to an array of neuron indices (the new order).
    """
    order = {}
    for name, means in class_activations.items():
        # means: [10 classes, n_neurons]
        preferred = means.argmax(axis=0)  # preferred class per neuron
        strength = means.max(axis=0)  # activation at that preferred class
        order[name] = np.lexsort((-strength, preferred))
    return order
