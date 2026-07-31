# timeseries-compressibility

Interactive exploration of how compressible quantized time series are.

The generating model is: i.i.d. Gaussian noise (std σ, measured in quantization
steps) → FIR filter → round to integers. The app shows the filter (convolution
kernel and frequency response, with cutoffs in Hz against a chosen sample
rate), a window of the generated integer signal (stationary by default, with a
play toggle to let it stream endlessly), and the measured compression of a
block of the generated integers under nine methods — zlib, zstd, and an rANS
entropy coder, each raw, delta-coded, and LPC-residual-coded — as bits per
sample and as ratio against raw int16 storage. The predictor order and the
block size are controls, so the measurement can be pushed from 10k to a million
samples and LPC from order 1 to 128. Each prefilter group also carries a hollow
bar: the order-0 entropy of the stream being coded, the limit a per-sample
entropy coder cannot beat, which ANS misses by 1–2% (its symbol table plus its
own arithmetic loss).

The reference rate R — the entropy rate of the process, the bits/sample limit
no lossless method can beat — comes from the companion
[timeseries-entropy](https://github.com/concept-collection/timeseries-entropy)
package: an unbiased Monte-Carlo estimator of H(z_next | a long past), by Gibbs
sampling the latent Gaussian under the rounding constraints and applying
Rhee–Glynn randomized telescoping to the sampled chain. The app shows the exact
command to run it at the current settings; estimating R in the browser is
planned, and until then the UI shows a placeholder for it.

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
src/compress/    lossless codecs run in the browser: zlib (fflate), zstd (wasm),
                 ans.ts (a bit-identical port of simple_ans), and FLAC-style
                 integer LPC; borrowed from entropy-quantized-linear-transform
src/worker/      the codecs run off the main thread on a debounced parameter set
src/components/  controls, filter plots, signal canvas, compression chart,
                 and the reference-rate method note
```

Every reported size round-trips through the decoder and includes whatever the
decoder needs (ANS symbol table, LPC coefficients). The signal view and the
compression block read the same fixed latent noise sequence — parameter changes
transform the same underlying data rather than resampling it, and the first
window shown is the start of the block that gets compressed.
