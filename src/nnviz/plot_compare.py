"""Head-to-head: the same per-image activation edges drawn two ways.

Left column is the datashaded hive plot (hiveplotlib); right column is straight parallel
coordinates (plain matplotlib). Built to answer whether hiveplotlib earns its place over
ordinary parallel coordinates for the feedforward pathway figure.

For each digit we take its frozen-sample images, and for each image draw paths from its
top-p active hidden1 neurons through its top-q active hidden2 neurons to its predicted
output node. Both layouts get the exact same set of paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import datashader as ds  # noqa: E402
import datashader.transfer_functions as tf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from hiveplotlib.viz.datashader import datashade_edges_mpl  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from nnviz import tracking  # noqa: E402
from nnviz.plot_pathways import _mark_output_node, build_base  # noqa: E402

FRAMES_DIR = Path("frames")
# same colormap hiveplotlib's datashader viz uses by default, so both panels match
DS_CMAP = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)


def layer_positions(order: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map each layer's neuron index to its normalized [0, 1] position from the order."""
    pos = {}
    for layer, idx_order in order.items():
        n = len(idx_order)
        rank = np.empty(n, dtype=float)
        rank[idx_order] = np.arange(n)
        pos[layer] = rank / max(n - 1, 1)
    return pos


def image_paths(
    h1: np.ndarray, h2: np.ndarray, out: np.ndarray, p: int, q: int
) -> list:
    """Per-image (h1, h2, predicted-out) triples from the top-p / top-q active neurons."""
    top1 = [a for a in np.argsort(h1)[::-1][:p] if h1[a] > 0]
    top2 = [b for b in np.argsort(h2)[::-1][:q] if h2[b] > 0]
    pred = int(np.argmax(out))
    return [(int(a), int(b), pred) for a in top1 for b in top2]


def class_paths(sa: dict[str, np.ndarray], digit: int, p: int, q: int) -> list:
    """All per-image paths for frozen-sample images whose true label is ``digit``."""
    mask = sa["labels"] == digit
    h1, h2, out = sa["hidden1"][mask], sa["hidden2"][mask], sa["output"][mask]
    paths = []
    for i in range(h1.shape[0]):
        paths.extend(image_paths(h1[i], h2[i], out[i], p, q))
    return paths


def render_hive(ax: plt.Axes, base, paths: list, digit: int) -> None:
    """Datashaded hive plot of the given paths (two hops as Bezier arcs)."""
    hp = base.copy()
    e1 = np.array([[f"hidden1:{a}", f"hidden2:{b}"] for a, b, _ in paths], dtype=object)
    e2 = np.array([[f"hidden2:{b}", f"output:{c}"] for _, b, c in paths], dtype=object)
    if len(e1):
        hp.connect_axes(
            edges=e1, axis_id_1="hidden1", axis_id_2="hidden2", a2_to_a1=False
        )
    if len(e2):
        hp.connect_axes(
            edges=e2, axis_id_1="hidden2", axis_id_2="output", a2_to_a1=False
        )
    datashade_edges_mpl(hp, fig=ax.figure, ax=ax, pixel_spread=2)
    _mark_output_node(ax, base, digit)
    ax.set_title(f"hive  ·  digit {digit}", fontsize=12)


def render_pcp(
    ax: plt.Axes, pos: dict[str, np.ndarray], paths: list, digit: int
) -> None:
    """Datashaded straight parallel-coordinates plot of the same paths (same engine)."""
    xs: list[float] = []
    ys: list[float] = []
    for a, b, c in paths:
        xs += [0.0, 1.0, 2.0, np.nan]
        ys += [pos["hidden1"][a], pos["hidden2"][b], pos["output"][c], np.nan]
    df = pd.DataFrame({"x": xs, "y": ys})
    cvs = ds.Canvas(
        plot_width=300, plot_height=440, x_range=(-0.1, 2.1), y_range=(-0.03, 1.03)
    )
    shaded = tf.shade(cvs.line(df, "x", "y", agg=ds.count()), cmap=DS_CMAP, how="log")
    img = tf.spread(shaded, px=2)  # match the hive panel's pixel_spread
    ax.imshow(
        img.to_pil(), extent=(-0.1, 2.1, -0.03, 1.03), aspect="auto", origin="upper"
    )
    for x in (0, 1, 2):
        ax.axvline(x, color="0.6", lw=1, zorder=2)
    ax.scatter(
        [2],
        [pos["output"][digit]],
        s=70,
        color="crimson",
        edgecolors="white",
        linewidths=0.8,
        zorder=10,
    )
    ax.set_xlim(-0.15, 2.15)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["h1", "h2", "out"])
    ax.set_yticks([])
    ax.set_title(f"parallel coords  ·  digit {digit}", fontsize=12)


def parse_args() -> argparse.Namespace:
    """Parse options for the hive-vs-PCP comparison."""
    p = argparse.ArgumentParser(
        description="Compare hive plot vs parallel coordinates."
    )
    p.add_argument("--run-id", default=None, help="default: latest run")
    p.add_argument("--step", type=int, default=None, help="default: final checkpoint")
    p.add_argument("--digits", type=int, nargs="+", default=[0, 4, 8])
    p.add_argument(
        "--p", type=int, default=3, help="top active hidden1 neurons per image"
    )
    p.add_argument(
        "--q", type=int, default=2, help="top active hidden2 neurons per image"
    )
    return p.parse_args()


def main() -> None:
    """Render the hive-vs-PCP comparison grid for a few digits at one checkpoint."""
    args = parse_args()
    tracking.configure()
    run_id = args.run_id or tracking.latest_run_id()
    client = MlflowClient()
    steps = sorted(
        int(a.path.split("_")[-1]) for a in client.list_artifacts(run_id, "checkpoints")
    )
    step = args.step if args.step is not None else steps[-1]

    order = dict(
        np.load(
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="neuron_order.npz"
            )
        )
    )
    sa = dict(
        np.load(
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=f"checkpoints/step_{step:06d}/sample_activations.npz",
            )
        )
    )
    base = build_base(order)
    pos = layer_positions(order)

    fig, axes = plt.subplots(len(args.digits), 2, figsize=(8, 3.6 * len(args.digits)))
    axes = np.atleast_2d(axes)
    for r, digit in enumerate(args.digits):
        paths = class_paths(sa, digit, args.p, args.q)
        render_hive(axes[r, 0], base, paths, digit)
        render_pcp(axes[r, 1], pos, paths, digit)

    fig.suptitle(f"same per-image edges, two layouts  ·  step {step}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FRAMES_DIR / f"compare_step_{step:06d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
