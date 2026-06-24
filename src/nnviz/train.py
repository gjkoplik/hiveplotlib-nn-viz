"""Train the MLP and log everything the hive-plot mappings need to mlflow.

One run captures, per checkpoint: the model state, a test-set confusion matrix,
per-class mean activations per layer, and per-image activations for a frozen image
sample (hidden layers + output logits). The neuron ordering and that image sample are
frozen once and logged at run level so every frame and every mapping share them.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

from nnviz import capture, data, order, tracking
from nnviz.model import LAYER_NAMES, MLP


def parse_args() -> argparse.Namespace:
    """Parse training hyperparameters from the command line."""
    p = argparse.ArgumentParser(description="Train MNIST MLP and log to mlflow.")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden1", type=int, default=64)
    p.add_argument("--hidden2", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num-checkpoints",
        type=int,
        default=60,
        help="target number of log-spaced checkpoints (dense early, sparse late)",
    )
    p.add_argument(
        "--sample-per-class",
        type=int,
        default=200,
        help="held-out test images per class frozen for per-image mappings",
    )
    p.add_argument("--experiment", default="mnist-hiveplot")
    return p.parse_args()


def checkpoint_steps(total_steps: int, num: int) -> set[int]:
    """Log-spaced checkpoint steps, dense early. Always includes 0 and ``total_steps``.

    MNIST converges in the first few hundred steps, so linear spacing wastes most of the
    movie on the flat tail; geometric spacing puts the frames where the action is.
    """
    spaced = np.geomspace(1, total_steps, num=num).round().astype(int)
    return {0, total_steps, *spaced.tolist()}


def log_checkpoint(
    model: MLP,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    sample_idx: np.ndarray,
    sample_y: np.ndarray,
    step: int,
) -> dict[str, np.ndarray]:
    """Compute and log per-checkpoint artifacts plus the test-accuracy metric.

    Returns the per-class mean activations (reused at the end to freeze the ordering).
    """
    acts, preds = capture.evaluate(model, test_x, test_y)
    y_true = test_y.numpy()
    test_acc = float((preds == y_true).mean())
    conf = capture.confusion(y_true, preds)
    class_acts = capture.class_mean_activations(acts, y_true)
    # per-image activations for the frozen sample, sliced from the full test pass
    sample_acts = {layer: acts[layer][sample_idx] for layer in LAYER_NAMES}

    mlflow.log_metric("test_acc", test_acc, step=step)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        torch.save(model.state_dict(), d / "model.pt")
        np.save(d / "confusion.npy", conf)
        np.savez(d / "class_activations.npz", **class_acts)
        np.savez(d / "sample_activations.npz", labels=sample_y, **sample_acts)
        mlflow.log_artifacts(str(d), artifact_path=f"checkpoints/step_{step:06d}")
    return class_acts


def main() -> None:
    """Run training end to end, logging metrics and per-checkpoint artifacts."""
    args = parse_args()
    torch.manual_seed(args.seed)

    train_x, train_y, test_x, test_y = data.load_tensors(args.dataset)
    sample_idx = data.frozen_sample(test_y, args.sample_per_class, args.seed)
    sample_y = test_y[sample_idx].numpy()

    model = MLP(args.hidden1, args.hidden2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = data.train_loader(train_x, train_y, args.batch_size, args.seed)
    ckpt_set = checkpoint_steps(args.epochs * len(loader), args.num_checkpoints)

    tracking.configure()
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))

        step = 0
        # initial checkpoint (random init) so the movie starts before any learning
        class_acts = log_checkpoint(model, test_x, test_y, sample_idx, sample_y, step)

        for _epoch in range(args.epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                step += 1
                mlflow.log_metric("train_loss", float(loss.item()), step=step)
                if step in ckpt_set:
                    class_acts = log_checkpoint(
                        model, test_x, test_y, sample_idx, sample_y, step
                    )
                    model.train()

        # run-level frozen artifacts: neuron ordering (final model) + sample indices
        neuron_order = order.selectivity_order(class_acts)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            np.savez(d / "neuron_order.npz", **neuron_order)
            np.save(d / "sample_indices.npy", sample_idx)
            mlflow.log_artifacts(str(d))

        print(f"run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
