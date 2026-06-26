"""Mapping 2: per-class activation pathways through the network, over training.

For each digit, a hive plot with three axes (hidden1, hidden2, output). Neurons are
ordered once by class selectivity and frozen, so node positions never move; only the
edges change. Edges for digit k connect neurons that co-activate strongly for class-k
inputs (thresholded to the top few), so each panel shows that digit's sparse pathway.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from hiveplotlib import BaseHivePlot, HivePlot, NodeCollection  # noqa: E402
from hiveplotlib.viz.matplotlib import hive_plot_viz  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from nnviz import animate, tracking  # noqa: E402
from nnviz.model import LAYER_NAMES  # noqa: E402

LAYER_LABELS = {"hidden1": "hidden 1", "hidden2": "hidden 2", "output": "output"}
# rotation places hidden1 at the top (90 deg); the 3 axes spread evenly to 210 / 330.
AXIS_ROTATION = 90
FRAMES_DIR = Path("frames")
MOVIES_DIR = Path("movies")
EDGE_COLOR = "#2c3e50"


def build_plot(
    order: dict[str, np.ndarray],
    edges: np.ndarray | None = None,
    *,
    num_steps: int = 100,
    **edge_kwargs,
) -> HivePlot:
    """One hive plot for a panel: layer axes, frozen node order, and the given edges.

    Neurons are partitioned onto a hidden1 / hidden2 / output axis and frozen by
    selectivity rank (``pos``). ``edges`` is an ``(n, 2)`` array of ``"layer:idx"`` id
    pairs; ``HivePlot`` routes each pair to the correct axis pair from its endpoints'
    layers, so the two hops (and any wrap-around) need not be split by the caller.
    Uniform ``edge_kwargs`` (color, alpha, lw, ...) style every edge.
    """
    rows = [
        {"unique_id": f"{layer}:{int(neuron)}", "layer": layer, "pos": rank}
        for layer in LAYER_NAMES
        for rank, neuron in enumerate(order[layer])
    ]
    nodes = NodeCollection(data=pd.DataFrame(rows), unique_id_column="unique_id")
    if edges is None or len(edges) == 0:
        edges = np.empty((0, 2), dtype=object)
    return HivePlot(
        nodes=nodes,
        edges=np.asarray(edges, dtype=object),
        partition_variable="layer",
        sorting_variables="pos",
        axes_order=list(LAYER_NAMES),
        rotation=AXIS_ROTATION,
        num_steps_per_edge=num_steps,
        all_edge_kwargs=edge_kwargs or None,
        axis_kwargs={
            layer: {"start": 1, "end": 5, "long_name": LAYER_LABELS[layer]}
            for layer in LAYER_NAMES
        },
    )


def top_edges(weights: np.ndarray, src: str, dst: str, k_top: int) -> np.ndarray:
    """Return the top-``k_top`` positive co-activation edges as an (n, 2) id array."""
    flat = weights.ravel()
    pos = np.flatnonzero(flat > 0)
    if pos.size == 0:
        return np.empty((0, 2), dtype=object)
    keep = pos[np.argsort(flat[pos])[::-1][:k_top]]
    si, di = np.unravel_index(keep, weights.shape)
    return np.column_stack([[f"{src}:{i}" for i in si], [f"{dst}:{j}" for j in di]])


def selectivity(class_act: dict[str, np.ndarray], layer: str, k: int) -> np.ndarray:
    """How much each neuron in ``layer`` fires for class ``k`` above its all-class mean.

    Clipped at 0, so only class-preferring neurons contribute. Subtracting the baseline
    is what makes each digit's panel diverge; raw activation is dominated by neurons that
    fire for everything, which looks identical in every panel.
    """
    a = class_act[layer]
    return np.clip(a[k] - a.mean(axis=0), 0, None)


def pathway_edges(
    class_act: dict[str, np.ndarray], k: int, top_h1h2: int, top_h2o: int
) -> tuple[np.ndarray, np.ndarray]:
    """Class-selective co-activation pathway for digit ``k`` across the two transitions."""
    s1 = selectivity(class_act, "hidden1", k)
    s2 = selectivity(class_act, "hidden2", k)
    so = selectivity(class_act, "output", k)
    e1 = top_edges(np.outer(s1, s2), "hidden1", "hidden2", top_h1h2)
    e2 = top_edges(np.outer(s2, so), "hidden2", "output", top_h2o)
    return e1, e2


def _mark_output_node(ax: plt.Axes, base: BaseHivePlot, k: int) -> None:
    """Mark the correct output node for digit ``k`` so the convergence target is obvious."""
    placements = base.axes["output"].node_placements
    row = placements[placements["unique_id"] == f"output:{k}"]
    ax.scatter(
        row["x"],
        row["y"],
        s=80,
        color="crimson",
        edgecolors="white",
        linewidths=0.8,
        zorder=10,
    )


def render_frame(
    order: dict[str, np.ndarray],
    class_act: dict[str, np.ndarray],
    step: int,
    out_path: Path,
    top_h1h2: int,
    top_h2o: int,
) -> None:
    """Render the 2x5 grid of per-digit pathways for one checkpoint."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    for k, ax in enumerate(axes.flat):
        e1, e2 = pathway_edges(class_act, k, top_h1h2, top_h2o)
        hp = build_plot(
            order, np.vstack([e1, e2]), color=EDGE_COLOR, alpha=0.45, lw=1.0
        )
        hive_plot_viz(
            hp,
            fig=fig,
            ax=ax,
            show_axes_labels=False,
            node_kwargs={"color": "lightgray", "s": 5},
        )
        _mark_output_node(ax, hp, k)
        ax.set_title(str(k), fontsize=14)
    fig.suptitle(f"activation pathways  ·  step {step}", fontsize=18)
    fig.text(
        0.5,
        0.01,
        "each panel: hidden1 (top) -> hidden2 (lower-left) -> output (lower-right)",
        ha="center",
        fontsize=11,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse plotting options from the command line."""
    p = argparse.ArgumentParser(description="Render mapping 2 (activation pathways).")
    p.add_argument("--run-id", default=None, help="default: latest run")
    p.add_argument("--step", type=int, default=None, help="single checkpoint to render")
    p.add_argument("--animate", action="store_true", help="render all frames + movie")
    p.add_argument("--top-h1h2", type=int, default=40)
    p.add_argument("--top-h2o", type=int, default=12)
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


def main() -> None:
    """Render mapping 2 for one checkpoint (default: final) or the whole run."""
    args = parse_args()
    tracking.configure()
    run_id = args.run_id or tracking.latest_run_id()
    client = MlflowClient()

    order_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="neuron_order.npz"
    )
    order = dict(np.load(order_path))

    steps = sorted(
        int(a.path.split("_")[-1]) for a in client.list_artifacts(run_id, "checkpoints")
    )
    if not steps:
        raise SystemExit("no checkpoints found in run")
    targets = (
        steps if args.animate else [args.step if args.step is not None else steps[-1]]
    )

    out_paths = []
    for step in targets:
        ca_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=f"checkpoints/step_{step:06d}/class_activations.npz",
        )
        render_frame(
            order,
            dict(np.load(ca_path)),
            step,
            FRAMES_DIR / f"pathways_step_{step:06d}.png",
            args.top_h1h2,
            args.top_h2o,
        )
        out_paths.append(FRAMES_DIR / f"pathways_step_{step:06d}.png")
        print(f"wrote {out_paths[-1]}")

    if args.animate:
        print(
            f"wrote {animate.stitch(out_paths, MOVIES_DIR / 'pathways.mp4', args.fps)}"
        )


if __name__ == "__main__":
    main()
