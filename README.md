# timeseries-compressibility

Interactive exploration of how compressible quantized time series are.

The generating model is: i.i.d. Gaussian noise (std σ, measured in quantization
steps) → FIR filter → round to integers. The app shows the filter (convolution
kernel and frequency response, with cutoffs in Hz against a chosen sample
rate), a window of the generated integer signal (stationary by default, with a
play toggle to let it stream endlessly), and the measured compression of a
block of the generated integers under ten methods — zlib, zstd, and an rANS
entropy coder, each raw, delta-coded, and LPC-residual-coded, plus the
conditional-Gaussian coder below — as bits per sample and as ratio against raw
int16 storage. The predictor order and the
block size are controls, so the measurement can be pushed from 10k to a million
samples and LPC from order 1 to 128. Under the chart, every coder that codes
against an explicit model is scored against it — ANS against the order-0
entropy of the stream it was handed, the arithmetic coder against its own
predictive distribution — which separates how well a coder does its job (1–2%
overhead for ANS, its symbol table plus its arithmetic loss) from how good the
model was in the first place.

The tenth method is the one that can pass those hollow bars: the same LPC
prediction kept at full precision, each sample arithmetic-coded under a
discretized Gaussian centred on the real-valued prediction. When the
prediction error is a fraction of a quantization step (narrowband filters,
moderate σ), whether the prediction falls near a bin centre or a bin edge is
worth ~0.3–0.4 bits/sample — information the integer residual has already
destroyed, which is why LPC+ANS plateaus far above R there. Its only limit is
R itself. Like every other bar, its size is real: encoded, decoded, verified,
side information included.

The entropy rate R of the process — the bits/sample limit no lossless method
can beat — is estimated in the browser by the method of the companion
[timeseries-entropy](https://github.com/concept-collection/timeseries-entropy)
package: an unbiased Monte-Carlo estimator of H(z_next | a long past), by Gibbs
sampling the latent Gaussian under the rounding constraints and applying
Rhee–Glynn randomized telescoping to the sampled chain. The package's analytic
approximation of R — Szegő's one-step prediction error with roundoff as a 1/12
dither floor, fed through the Gaussian⊕uniform entropy — is drawn as a dotted
reference line at all times, so the Monte-Carlo estimate lands beside its
prediction. A button starts a web
worker that averages one independent past at a time (live mean ± se, dashed
line on the chart) until stopped; the app also shows the exact command to run
the Python original at the same settings as an independent check. The
in-browser code in `src/entropy/` is a hand-synced TypeScript port of that
package — change one, change the other. A WebGPU Gibbs sweep may replace the
scalar one someday; the sweep is isolated so it can be swapped.

## Run it

```sh
npm install
npm run dev
```

## Layout

```
src/model/       the latent source (fixed seeded randomness indexed by sample
                 position, convolved zero-phase with the kernel on demand)
                 and the FIR presets
src/entropy/     the unbiased entropy-rate estimator: hand-synced TypeScript
                 port of the timeseries-entropy package (Gibbs conditional
                 sampler, Rhee–Glynn telescoping, Cody erfc / Acklam ndtri,
                 xoshiro128** RNG, the analytic rate prediction of theory.py)
src/compress/    lossless codecs run in the browser: zlib (fflate), zstd (wasm),
                 ans.ts (a bit-identical port of simple_ans), FLAC-style
                 integer LPC (borrowed from entropy-quantized-linear-transform),
                 and conditionalGaussian.ts — real-coefficient prediction with
                 each sample arithmetic-coded under a discretized Gaussian at
                 the real-valued prediction
src/worker/      the codecs and the estimator run off the main thread; the
                 estimator worker refines one past at a time until terminated
src/components/  controls, filter plots, signal canvas, compression chart,
                 and the entropy-rate method note
```

Every reported size round-trips through the decoder and includes whatever the
decoder needs (ANS symbol table, LPC coefficients). The signal view and the
compression block read the same fixed latent noise sequence — parameter changes
transform the same underlying data rather than resampling it, and the first
window shown is the start of the block that gets compressed.
