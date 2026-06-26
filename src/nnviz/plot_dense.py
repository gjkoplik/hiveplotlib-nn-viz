"""Mapping 2: per-image *selectivity* pathways as datashaded hive plots.

One panel per digit, animated over training. For each image we keep the neurons firing
above their baseline (their average activation over the whole sample = the "average
image"): the smallest set covering ``MASS_FRAC`` of that image's selectivity, capped per
layer. Edges run kept-hidden1 -> kept-hidden2 -> predicted output. Datashaded with one
global ``vmax`` so density is comparable across digits and across training time.

Keying off selectivity (not raw activation) is the point: raw activation is distributed
and roughly static over training, while selectivity is concentrated and sharpens as the
net learns, which is what brings the per-digit pathways and the condensation out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
from hiveplotlib.viz.datashader import datashade_edges_mpl  # noqa: E402
from hiveplotlib.viz.matplotlib import axes_viz  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from nnviz import animate, tracking  # noqa: E402
from nnviz.plot_pathways import _mark_output_node, build_plot  # noqa: E402

FRAMES_DIR = Path("frames")
MOVIES_DIR = Path("movies")
NUM_STEPS = 30
PIXEL_SPREAD = 1
DPI = 150
MASS_FRAC = 0.8  # keep smallest neuron set covering this much of an image's selectivity
CAP = {
    "hidden1": 4,
    "hidden2": 2,
}  # bound on kept neurons per layer (edge-count control)
HIDDEN = ("hidden1", "hidden2")


def baselines(sa: dict) -> dict:
    """Per-neuron mean activation over all sample images (the 'average image')."""
    return {layer: sa[layer].mean(axis=0) for layer in HIDDEN}


def _selective(vec: np.ndarray, base: np.ndarray, cap: int) -> np.ndarray:
    """Above-baseline neurons: smallest set covering MASS_FRAC of selectivity, capped."""
    sel = np.clip(vec - base, 0, None)
    order = np.argsort(sel)[::-1]
    total = sel.sum()
    if total <= 0:
        return order[:1]
    keep = int(np.searchsorted(np.cumsum(sel[order]), MASS_FRAC * total) + 1)
    return order[: min(keep, cap)]


def selective_paths(sa: dict, digit: int, bl: dict) -> list:
    """Per-image (h1, h2, predicted-out) triples over each image's selective neurons."""
    mask = sa["labels"] == digit
    h1, h2, out = sa["hidden1"][mask], sa["hidden2"][mask], sa["output"][mask]
    paths = []
    for i in range(h1.shape[0]):
        s1 = _selective(h1[i], bl["hidden1"], CAP["hidden1"])
        s2 = _selective(h2[i], bl["hidden2"], CAP["hidden2"])
        pred = int(out[i].argmax())
        paths.extend((int(a), int(b), pred) for a in s1 for b in s2)
    return paths


def render_panel(ax: plt.Axes, order: dict, paths: list, vmax: float | None):
    """Build one digit's selectivity hive plot, then datashade its per-image paths."""
    e_h1h2 = [[f"hidden1:{a}", f"hidden2:{b}"] for a, b, _ in paths]
    e_h2o = [[f"hidden2:{b}", f"output:{c}"] for _, b, c in paths]
    # h1 <-> output: the observed input/output co-activation that closes the triangle
    # (a real measurement, not a wire), which is what makes the radial layout earn it.
    e_h1o = [[f"hidden1:{a}", f"output:{c}"] for a, _, c in paths]
    edges = e_h1h2 + e_h2o + e_h1o
    hp = build_plot(
        order, np.array(edges, dtype=object) if edges else None, num_steps=NUM_STEPS
    )
    axes_viz(
        hp, fig=ax.figure, ax=ax, show_axes_labels=False, zorder=6, color="0.4", lw=1.0
    )
    _, _, im = datashade_edges_mpl(
        hp, fig=ax.figure, ax=ax, vmax=vmax, pixel_spread=PIXEL_SPREAD
    )
    return im


def probe_vmax(order: dict, sa: dict) -> float | None:
    """One global density ceiling so brightness is comparable across panels and frames."""
    bl = baselines(sa)
    fig, axes = plt.subplots(2, 5, dpi=DPI)
    vmaxes = []
    for k, ax in enumerate(axes.flat):
        im = render_panel(ax, order, selective_paths(sa, k, bl), None)
        if im is not None and im.norm.vmax is not None:
            vmaxes.append(float(im.norm.vmax))
    plt.close(fig)
    return max(vmaxes) if vmaxes else None


def render_frame(order: dict, sa: dict, step: int, out_path: Path, vmax) -> None:
    """Render the 2x5 grid of datashaded per-digit selectivity pathways for one checkpoint."""
    bl = baselines(sa)
    marker = build_plot(order)  # no edges; just a source for the frozen node placements
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5), dpi=DPI)
    for k, ax in enumerate(axes.flat):
        render_panel(ax, order, selective_paths(sa, k, bl), vmax)
        _mark_output_node(ax, marker, k)
        ax.set_title(str(k), fontsize=14)
    fig.suptitle(
        f"layer co-activation (h1 / h2 / output), polar  ·  step {step}", fontsize=18
    )
    fig.text(
        0.5,
        0.01,
        "axes: hidden1, hidden2, output  ·  edges = observed co-activation, not weights",
        ha="center",
        fontsize=11,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse plotting options."""
    p = argparse.ArgumentParser(
        description="Render mapping 2 (selectivity, datashaded)."
    )
    p.add_argument("--run-id", default=None, help="default: latest run")
    p.add_argument("--step", type=int, default=None, help="single checkpoint to render")
    p.add_argument("--animate", action="store_true", help="render all frames + movie")
    p.add_argument("--vmax", type=float, default=None, help="fixed density ceiling")
    p.add_argument("--fps", type=int, default=8)
    return p.parse_args()


def _sample_path(run_id: str, step: int) -> str:
    return mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=f"checkpoints/step_{step:06d}/sample_activations.npz",
    )


def main() -> None:
    """Render mapping 2 for one checkpoint (default final) or the whole run."""
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
    steps = sorted(
        int(a.path.split("_")[-1]) for a in client.list_artifacts(run_id, "checkpoints")
    )

    vmax = args.vmax
    if vmax is None:
        vmax = probe_vmax(order, dict(np.load(_sample_path(run_id, steps[-1]))))
        print(f"global vmax = {vmax:.1f}")

    targets = (
        steps if args.animate else [args.step if args.step is not None else steps[-1]]
    )
    out_paths = []
    for step in targets:
        sa = dict(np.load(_sample_path(run_id, step)))
        out = FRAMES_DIR / f"dense_step_{step:06d}.png"
        render_frame(order, sa, step, out, vmax)
        out_paths.append(out)
        print(f"wrote {out}")

    if args.animate:
        print(
            f"wrote {animate.stitch(out_paths, MOVIES_DIR / 'pathways_dense.mp4', args.fps)}"
        )


if __name__ == "__main__":
    main()
