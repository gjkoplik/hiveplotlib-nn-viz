# hiveplotlib-nn-viz — what we built, and why

Status: **prototype, exploratory** (last updated 2026-06-24). Throwaway repo, not part of
hiveplotlib. This file is the durable summary; read it before picking the work back up.

## The bet

Watch a small neural network learn MNIST, as a movie over training, using hiveplotlib's
polar parallel coordinates (P2CP) and radial layouts. Two reasons the tool fits:

1. **Deterministic layout.** Node positions are fixed by the data, so in a training movie
   every bit of motion is real signal, not the frame-to-frame wobble of t-SNE / UMAP.
2. **Generalizable and library-level.** These are a few lines on top of an existing
   library, the everyday way to get the view on any classifier, not a bespoke one-off.

## What we built (the active set)

One tiny MLP, `784 -> 64 -> 32 -> 10`, ReLU, CPU, fixed seed, trained on MNIST and logged
to mlflow over 51 log-spaced checkpoints (dense early, since MNIST converges by ~step 35).
Two figures, each animated, each a 2x5 grid of per-digit panels:

### Figure A — per-class output-probability P2CP  (`plot_p2cp.py` -> `movies/p2cp.mp4`)
One panel per true digit. That class's test images are datashaded loops over 10 axes (one
per class probability). Early every loop hugs a tight uncertain central ring; trained, each
class blooms into a single **petal** on its own axis (the home axis is drawn red), and the
confusions show as faint secondary lobes. Alpha-ramped colormap, log-density colorbar. This
is the polished, legible one.

### Figure B — cross-layer co-activation, polar vs straight  (mapping 2)
- `plot_dense.py` -> `movies/pathways_dense.mp4` (**polar**). Three axes
  (hidden1 / hidden2 / output). For each image we keep its class-*selective* neurons (see
  below) and draw observed co-activation between them across **all three** pairwise layer
  relationships (h1<->h2, h2<->output, and h1<->output), a closed triangle. Datashaded,
  animated. The edges are observed co-activation, **not** weights.
- `plot_dense_pcp.py` -> `movies/pathways_dense_pcp.mp4` (**straight**, the comparison).
  Same edges as straight parallel coordinates, axes `h1 -> h2 -> out -> h1` (a repeated h1
  axis to close the loop). The point of the pair: the straight version needs a repeated axis
  to do what polar closes for free, which is the argument for the radial layout.

### Pipeline and infra
- `train.py` trains and logs to mlflow. Per checkpoint it logs `model.pt`, `confusion.npy`,
  `class_activations.npz`, and `sample_activations.npz` (per-image hidden1/hidden2/output
  for a frozen, class-balanced test sample). Run-level it logs `neuron_order.npz` (frozen
  selectivity ordering) and `sample_indices.npy`. The active figures use only
  `sample_activations.npz` + `neuron_order.npz`; `confusion.npy` and `class_activations.npz`
  are vestigial (the retired figures used them), harmless to leave.
- `capture.py`, `order.py`, `data.py`, `model.py`, `tracking.py`, `animate.py` support it.
- **Helpers kept, not figures:** `plot_pathways.py` and `plot_compare.py` look dead but are
  retained because the live figures import small helpers from them (`build_base`,
  `_mark_output_node`; `DS_CMAP`, `layer_positions`). To fully delete them, lift those four
  into a shared module first (a refactor, not done).

### Run it
`make install` (then `make install-viz` for datashader), `make train`, `make ui`. Figures:
`uv run python -m nnviz.plot_p2cp --animate`, `... plot_dense --animate`,
`... plot_dense_pcp --animate`. Single frame: drop `--animate`, add `--step N`.

## What we tried and dropped (so we don't re-litigate)

- **Confusion slopegraph** (true-vs-predicted, was "mapping 1"): built, then dropped. The
  "correct" panel was just "accuracy went up" as filling bands, and the confusion it showed
  is already in Figure A's secondary lobes. Files deleted.
- **Stability long-exposure**: built, shelved. A union-of-a-window long exposure looked the
  same as the plain dense view, because count density conflates per-edge *recurrence* with
  spatial *overlap*. A true per-edge persistence weighting might differ but was not worth the
  extra figure to explain. Files deleted.
- **Raw-activation / top-K hidden pathways**: the hidden code is **distributed**, not sparse
  (measured: ~20-38 of 64 hidden1 neurons effectively active per image, and that barely
  changes over training). There is no clean per-digit "pathway" to reveal in hidden space;
  the per-digit panels stay similar however you threshold. The discriminative structure lives
  in **output** space, which is why Figure A reads cleanly and Figure B is subtler. We key
  Figure B's edges off **selectivity** (activation above the neuron's all-image baseline, the
  "average image"), not raw activation, with a data-driven cutoff (smallest neuron set
  covering ~80% of an image's selectivity, capped).

## Durable gotchas

- **mlflow 3.x blocks the file store** (it raises on `file:./mlruns`). Use the SQLite backend
  `sqlite:///mlflow.db`, centralized in `tracking.py`; `make ui` passes the same backend.
- **datashader: spread the AGGREGATE, then shade**, `tf.shade(tf.spread(agg))`, not
  `tf.spread(tf.shade(agg))`. Spreading the shaded image composites with "over", so a light
  (low-count) line paints over a dark crossing, which is impossible if you are truly counting.
  This is what hiveplotlib's `datashade_edges_mpl` does internally.
- **Standardize the density scale** (one global `vmax`) across panels and frames, or per-frame
  renormalization fakes the over-time change.
- API notes: build manual hive plots with `BaseHivePlot` (`HivePlot` requires
  `partition_variable`/`sorting_variables`). `datashade_edges_mpl` is edges-only, compose
  `axes_viz` for the axes. Per-class P2CPs come from `p2cp_n_axes(df, axes=..., split_on=...)`.

## Novelty framing (LOCKED, for any future writeup)

Prior-art scan (a deep-research workflow plus a focused follow-up) verdict:

- **Figure A (output-probability P2CP): incremental-toward-novel.** The **Grand Tour**
  (Distill 2020, distill.pub/2020/grand-tour) owns the *use case*, it animates MNIST softmax
  over training, per-class convergence to simplex corners, even naming per-class learning
  epochs (digit 1 at epoch 14, digit 7 at epoch 21). The difference is layout: the Grand Tour
  is a bespoke linear-projection interactive masterpiece; ours is `p2cp_n_axes()` on the
  softmax, the generalizable library-level way to get a Grand-Tour-flavored gestalt on any
  classifier, with per-class confusion surfaced as secondary lobes for free. **Frame it as a
  generalizable encoding that complements and explicitly cites the Grand Tour, not "first to
  visualize softmax training dynamics."** Pre-empt the "isn't this just the Grand Tour?"
  reviewer with a direct side-by-side (their simplex vs our ring-to-petal on the same
  checkpoints).
- **Figure B (cross-layer co-activation, polar): defensibly novel, the headline.** No located
  prior art draws neuron-to-neuron co-activation edges in any radial or parallel-coordinates
  form over training. The activation/co-activation canon (ActiVis, CNNVis, M-PHATE, Activation
  Atlas, Grand Tour) uses matrices, t-SNE / projections, or edge-bundled DAGs, never parallel
  coordinates, let alone polar. Standard PC of neural nets is essentially unused for this too
  (only a 2010 backprop-network straight-PC paper).
- **Caveat:** "novel" means none located after an adversarial search, not provably first;
  polar-PC-for-NNs is recent (~2021+), so an unindexed thesis or workshop could exist.
- **Closest prior art to cite:** Grand Tour (distill.pub/2020/grand-tour); ConfusionFlow
  (arXiv 1910.00969, confusion-over-training, Cartesian small multiples); ActiVis
  (arXiv 1704.01942) and CNNVis (arXiv 1604.07043) for the activation-viz canon.