# More Features Than Neurons

An interactive explainer on **superposition** in neural networks: why a single neuron
means several unrelated things at once, how sparsity makes that cramming affordable,
and how a sparse autoencoder runs the trick backwards.

**Read it: https://code259.github.io/more-features-than-neurons/**

Submitted to the [VISxAI](https://visxai.io) workshop at IEEE VIS 2026.

## Everything here is live

No precomputed animations. Every toy model (18 parameters) and every sparse
autoencoder on the page runs a full Adam training loop in plain JavaScript as you
read — no GPU, no server, no dependencies. The ↻ buttons re-roll the random seed, so
the phenomena can be checked rather than taken on faith.

The figures render a lightly time-averaged (Polyak-style) view of the weights so you
see the learning trajectory instead of per-batch sampling noise; training itself runs
on the raw weights.

The real-model activation records in §0 and §6 are verbatim, fetched once at build
time from OpenAI's public neuron dataset and the Neuronpedia API.

## Reproductions

`node test.mjs` reproduces the canonical results of Elhage et al., *Toy Models of
Superposition* (2022), which the page then renders live:

- a linear model collapses to the top-*m* principal components
- ReLU + negative bias produces superposition on identical data
- antipodal-pair onset, then the 5-in-2 pentagon, as sparsity rises
- feature dimensionality sticking at 1, ½, and 3/8 (square antiprism at 8-in-3)
- SAE recovery of ground-truth directions, with dead-atom and shattering regimes

## Build

`index.html` is fully self-contained; `build.mjs` inlines `css/style.css`,
`js/core.js`, `js/realdata.js`, and `js/widgets.js` into it.

```bash
node build.mjs   # rebuild index.html after editing any source file
node test.mjs    # verify the physics
```

## Credits

Toy model and phase-transition results from Elhage et al., [*Toy Models of
Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) (2022).
SAE recipe from Bricken et al., [*Towards
Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features)
(2023). Real neuron records from OpenAI's [neuron
explainer](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
dataset; real dictionary atoms from [Neuronpedia](https://www.neuronpedia.org)
(gpt2-small layer-6 residual SAE by Joseph Bloom).
