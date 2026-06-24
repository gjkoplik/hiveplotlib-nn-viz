# hiveplotlib-nn-viz

Prototype exploring whether [hiveplotlib](https://gitlab.com/hiveplotlib/hiveplotlib)
hive plots and P2CPs make a compelling animated view of a small neural network learning
MNIST. Throwaway and experimental; not part of hiveplotlib.

The bet: a hive plot's layout is deterministic, so in a training movie every bit of
motion is real signal, not the frame-to-frame wobble you get from t-SNE or UMAP.

See [PLAN.md](PLAN.md) for the design, the three mappings, and milestones.

## Setup

Requires `uv` and Python 3.12.

    make install      # uv venv + deps (CPU torch)
    make train        # train once, log params, metrics, and per-checkpoint artifacts to mlflow
    make ui           # browse runs in the mlflow UI
    make pathways     # mapping 2: per-class activation pathways over training
