"""Mapping 2, PCP variant: the same per-image selectivity edges as straight datashaded
parallel coordinates, one panel per digit, over training.

Built to re-test in MOTION whether the hive layout actually beats parallel coordinates.
The earlier "hive wins" verdict came from a single still; since motion rescued the dense
hive (lock-in vs bounce), the PCP verdict deserves the same scrutiny. Same selectivity
edges and same global density scale as the hive movie, so only the layout differs.
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
from mlflow.tracking import MlflowClient  # noqa: E402

from nnviz import animate, tracking  # noqa: E402
from nnviz.plot_compare import DS_CMAP, layer_positions  # noqa: E402
from nnviz.plot_dense import _sample_path, baselines, selective_paths  # noqa: E402

FRAMES_DIR = Path("frames")
MOVIES_DIR = Path("movies")
SPREAD = 3  # PCP straight lines are sparse/isolated, so they need more spread than the
# hive's dense Bezier bundles (which were fine at 1) to avoid looking choppy.
EXTENT = (-0.1, 3.1, -0.03, 1.03)
CANVAS = {
    "plot_width": 520,
    "plot_height": 520,
    "x_range": (-0.1, 3.1),
    "y_range": (-0.03, 1.03),
}


def panel_agg(pos: dict, paths: list):
    """Datashader line aggregate of one digit's paths as 3-axis polylines."""
    xs: list[float] = []
    ys: list[float] = []
    for a, b, c in paths:
        # close the loop back to h1 (repeated axis) so the h1<->output relationship draws;
        # straight PC needs a repeated axis to do this, which polar closes for free.
        xs += [0.0, 1.0, 2.0, 3.0, np.nan]
        ys += [
            pos["hidden1"][a],
            pos["hidden2"][b],
            pos["output"][c],
            pos["hidden1"][a],
            np.nan,
        ]
    agg = ds.Canvas(**CANVAS).line(
        pd.DataFrame({"x": xs, "y": ys}), "x", "y", agg=ds.count()
    )
    return tf.spread(
        agg, px=SPREAD
    )  # spread the counts, then shade (overlaps stay additive)


def render_panel(ax: plt.Axes, pos: dict, paths: list, digit: int, vmax: float) -> None:
    """Straight datashaded parallel-coordinates panel for one digit (fixed density span)."""
    img = tf.shade(panel_agg(pos, paths), cmap=DS_CMAP, how="log", span=[1, vmax])
    ax.imshow(img.to_pil(), extent=EXTENT, aspect="auto", origin="upper")
    for x in (0, 1, 2, 3):
        ax.axvline(x, color="0.6", lw=1, zorder=2)
    ax.scatter(
        [2],
        [pos["output"][digit]],
        s=60,
        color="crimson",
        edgecolors="white",
        linewidths=0.8,
        zorder=10,
    )
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["h1", "h2", "out", "h1"], fontsize=8)
    ax.set_yticks([])
    ax.set_title(str(digit), fontsize=13)


def probe_vmax(pos: dict, sa: dict) -> float:
    """Global max line-count over the final frame's panels, for a fixed density span."""
    bl = baselines(sa)
    m = 0.0
    for k in range(10):
        agg = panel_agg(pos, selective_paths(sa, k, bl))
        m = max(m, float(np.nanmax(agg.to_numpy())) if agg.size else 0.0)
    return max(m, 1.0)


def render_frame(pos: dict, sa: dict, step: int, out_path: Path, vmax: float) -> None:
    """Render the 2x5 grid of datashaded PCP panels for one checkpoint."""
    bl = baselines(sa)
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for k, ax in enumerate(axes.flat):
        render_panel(ax, pos, selective_paths(sa, k, bl), k, vmax)
    fig.suptitle(
        f"co-activation, straight parallel coordinates (h1-h2-out-h1)  ·  step {step}",
        fontsize=16,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse plotting options."""
    p = argparse.ArgumentParser(description="Render mapping 2 as datashaded PCP.")
    p.add_argument("--run-id", default=None, help="default: latest run")
    p.add_argument("--step", type=int, default=None, help="single checkpoint to render")
    p.add_argument("--animate", action="store_true", help="render all frames + movie")
    p.add_argument("--vmax", type=float, default=None, help="fixed density ceiling")
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


def main() -> None:
    """Render mapping 2 (PCP) for one checkpoint (default final) or the whole run."""
    args = parse_args()
    tracking.configure()
    run_id = args.run_id or tracking.latest_run_id()
    client = MlflowClient()

    order = dict(
        np.load(
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="neuron_order.npz"
            )
        )
    )
    pos = layer_positions(order)
    steps = sorted(
        int(a.path.split("_")[-1]) for a in client.list_artifacts(run_id, "checkpoints")
    )

    vmax = args.vmax
    if vmax is None:
        vmax = probe_vmax(pos, dict(np.load(_sample_path(run_id, steps[-1]))))
        print(f"global vmax = {vmax:.0f}")

    targets = (
        steps if args.animate else [args.step if args.step is not None else steps[-1]]
    )
    out_paths = []
    for step in targets:
        sa = dict(np.load(_sample_path(run_id, step)))
        out = FRAMES_DIR / f"dense_pcp_step_{step:06d}.png"
        render_frame(pos, sa, step, out, vmax)
        out_paths.append(out)
        print(f"wrote {out}")

    if args.animate:
        print(
            f"wrote {animate.stitch(out_paths, MOVIES_DIR / 'pathways_dense_pcp.mp4', args.fps)}"
        )


if __name__ == "__main__":
    main()
