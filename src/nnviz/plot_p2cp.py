"""Mapping 3: per-class P2CPs of the output probabilities, over training.

Ten panels, one per true label. Each panel datashades that class's images as loops over
10 probability axes (one per class). Early the loops hug the uncertain center; trained,
they bloom into a single petal on that class's own axis (marked). One color, datashaded,
with a standardized density scale so the bloom is comparable across panels and over time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from hiveplotlib import p2cp_n_axes  # noqa: E402
from hiveplotlib.viz.datashader import datashade_edges_mpl  # noqa: E402
from hiveplotlib.viz.matplotlib import axes_viz  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from nnviz import animate, tracking  # noqa: E402
from nnviz.plot_compare import DS_CMAP  # noqa: E402
from nnviz.plot_dense import _sample_path  # noqa: E402

FRAMES_DIR = Path("frames")
MOVIES_DIR = Path("movies")
PROB_COLS = [str(i) for i in range(10)]
DPI = 150
PIXEL_SPREAD = 2

# datashader colormap with an alpha ramp: sparse density fades out while dense petals stay
# solid, so the tight curves read cleanly instead of full color everywhere.
_ramp = DS_CMAP(np.linspace(0, 1, 256))
_ramp[:, 3] = np.clip(np.linspace(0, 1, 256) / 0.3, 0, 1)  # alpha hits 1 by ~30% up
ALPHA_CMAP = ListedColormap(_ramp)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def build_df(sa: dict) -> pd.DataFrame:
    """Per-image class probabilities (columns '0'..'9') plus the true label."""
    df = pd.DataFrame(_softmax(sa["output"]), columns=PROB_COLS)
    df["true_label"] = sa["labels"].astype(int)
    return df


def class_p2cp(df: pd.DataFrame, k: int):
    """P2CP of only the images whose true label is ``k`` (axes fixed to [0, 1])."""
    df_k = df[df["true_label"] == k]
    return p2cp_n_axes(df_k, axes=PROB_COLS, vmins=[0.0] * 10, vmaxes=[1.0] * 10)


def render_panel(ax: plt.Axes, p2cp, k: int, vmax: float | None):
    """Draw the 10 axes, datashade class-k loops on top, mark the home axis."""
    axes_viz(
        p2cp,
        fig=ax.figure,
        ax=ax,
        show_axes_labels=False,
        zorder=6,
        color="0.55",
        lw=0.8,
    )
    _, _, im = datashade_edges_mpl(
        p2cp,
        fig=ax.figure,
        ax=ax,
        vmax=vmax,
        pixel_spread=PIXEL_SPREAD,
        cmap=ALPHA_CMAP,
    )
    # recolor the home axis (class k) red over the gray axes; thin and slightly alpha so
    # it points to the petal without dominating.
    home = p2cp.axes[str(k)]["axis"]
    ax.plot(
        [home.start[0], home.end[0]],
        [home.start[1], home.end[1]],
        color="red",
        lw=1.5,
        alpha=0.5,
        zorder=7,
    )
    ax.set_title(str(k), fontsize=13)
    return im


def probe_vmax(df: pd.DataFrame) -> float | None:
    """Global density ceiling across the 10 class panels of the final frame."""
    fig, axes = plt.subplots(2, 5, dpi=DPI)
    vmaxes = []
    for k, ax in enumerate(axes.flat):
        im = render_panel(ax, class_p2cp(df, k), k, None)
        if im is not None and im.norm.vmax is not None:
            vmaxes.append(float(im.norm.vmax))
    plt.close(fig)
    return max(vmaxes) if vmaxes else None


def render_frame(df: pd.DataFrame, step: int, out_path: Path, vmax, acc: float) -> None:
    """Render the 2x5 grid of per-class probability P2CPs for one checkpoint."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 9), dpi=DPI)
    for k, ax in enumerate(axes.flat):
        render_panel(ax, class_p2cp(df, k), k, vmax)
    fig.suptitle(
        f"output-probability P2CP by true class  ·  step {step}  ·  test acc {acc:.1%}",
        fontsize=17,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    sm = ScalarMappable(norm=LogNorm(vmin=1, vmax=vmax), cmap=ALPHA_CMAP)
    cax = fig.add_axes((0.945, 0.15, 0.012, 0.7))
    fig.colorbar(sm, cax=cax).set_label("image density (log scale)", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse plotting options."""
    p = argparse.ArgumentParser(description="Render mapping 3 (per-class P2CP grid).")
    p.add_argument("--run-id", default=None, help="default: latest run")
    p.add_argument("--step", type=int, default=None, help="single checkpoint to render")
    p.add_argument("--animate", action="store_true", help="render all frames + movie")
    p.add_argument("--vmax", type=float, default=None, help="fixed density ceiling")
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


def main() -> None:
    """Render mapping 3 for one checkpoint (default final) or the whole run."""
    args = parse_args()
    tracking.configure()
    run_id = args.run_id or tracking.latest_run_id()
    client = MlflowClient()
    steps = sorted(
        int(a.path.split("_")[-1]) for a in client.list_artifacts(run_id, "checkpoints")
    )

    def acc_of(sa: dict) -> float:
        return float((_softmax(sa["output"]).argmax(axis=1) == sa["labels"]).mean())

    vmax = args.vmax
    if vmax is None:
        vmax = probe_vmax(build_df(dict(np.load(_sample_path(run_id, steps[-1])))))
        print(f"global vmax = {vmax:.0f}")

    targets = (
        steps if args.animate else [args.step if args.step is not None else steps[-1]]
    )
    out_paths = []
    for step in targets:
        sa = dict(np.load(_sample_path(run_id, step)))
        out = FRAMES_DIR / f"p2cp_step_{step:06d}.png"
        render_frame(build_df(sa), step, out, vmax, acc_of(sa))
        out_paths.append(out)
        print(f"wrote {out}")

    if args.animate:
        print(f"wrote {animate.stitch(out_paths, MOVIES_DIR / 'p2cp.mp4', args.fps)}")


if __name__ == "__main__":
    main()
