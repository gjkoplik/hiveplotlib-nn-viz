# Watching an MNIST neural network learn with hive plots and P2CPs

We trained a tiny MLP (784-64-32-10, ReLU) on MNIST and saved a checkpoint every so often
through training. Both figures animate over those checkpoints.

The usual way to watch a net train is something like a t-SNE or UMAP movie, but those
redraw their layout every frame, so points drift around even when nothing real has changed
and you can't separate genuine learning from layout jitter. Here the layout is locked. We
decide where every neuron sits once (ordering each layer's neurons by which digit they end
up responding to most) and never move them again. So any motion you see is the network
actually changing, not the picture rearranging itself. The
[Grand Tour](https://distill.pub/2020/grand-tour/) makes the same move with a fixed linear
projection; we get there with no projection at all.

## `pathways_dense.mp4`: which neurons light up, per digit

A hive plot draws the network as three spokes coming out of a center, one per layer: first
hidden, second hidden, output. Each neuron is a point on its spoke. When an image passes
through, we draw a curve linking the neurons it activates, hopping first hidden to second
hidden to the predicted output digit.

We don't draw all of an image's neurons, only the ones that fire hardest in each hidden
layer. "Hardest" is measured against each neuron's own baseline: the level it typically sits
at across all images. Some neurons fire a lot no matter which digit they see, which tells
you nothing specific, so for a given image we keep only the neurons firing well above their
own usual level, the ones reacting to this digit in particular.

How many we keep is not fixed, and this matters for reading the figure. Per layer, we rank
an image's above-baseline neurons strongest first and keep adding them until they cover
about 80% of that image's above-baseline firing, or until we hit a per-layer cap, whichever
comes first. So an image whose response is concentrated in one or two neurons draws one or
two; an image whose response is smeared across many draws more, up to the cap of four in the
first hidden layer and two in the second. As the net learns and its representations sharpen,
each image needs fewer neurons to reach that 80%, so part of what you see tightening over
training is the representations themselves concentrating, not just edges moving around.

One hive plot per digit, drawing all of that digit's test images and shading by how much the
curves overlap.

What to watch: early in training every hive plot is a smear and they all look alike. As the
net learns, each digit collapses onto a few clean, well-worn routes, and the ten hive plots
become visibly different from one another.

### Why not use a PCP

In the hive plot we link the first hidden layer straight to the output, not just
first-to-second and second-to-output. The ring of axes gives that extra link for free; a
straight parallel coordinates plot only joins neighboring axes, so it would need the
first-hidden axis repeated as an extra column to show the same thing. Hive plots are also
compact, which lets us pack the per-digit views into side-by-side small multiples.

## `p2cp.mp4`: output confidence, per digit

A P2CP is the same spokes-from-a-center idea applied to plain data instead of a network.
Here there are ten spokes, one per output class. For a single image we draw a loop that
crosses each spoke at a height equal to the probability the net assigned to that class. A
confident "this is a 7" is a loop that spikes far out on the 7 spoke and stays low on the
others. One P2CP per true digit, with that digit's own spoke marked.

What to watch: early, every loop is bunched in the middle, because an untrained net spreads
about 10% across all ten classes. By the end each P2CP's loops bloom into a strong spike on
the correct spoke. The other spokes mostly drop low, but confusable digits keep a noticeable
secondary spike on each other's axes (e.g. 4 and 9), which you can read straight off the
plot.

## Prior art

Closest in spirit is the [Grand Tour](https://distill.pub/2020/grand-tour/) (Li, Zhao,
Scheidegger, Distill 2020): a fixed, deterministic projection, so motion in the animation
comes from the model and not the visualization. Hive plots and P2CPs get the same guarantee
by pinning the axes and the neuron order instead.
