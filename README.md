# timeseries-compressibility

Interactive exploration of how compressible quantized time series are.

The generating model is: i.i.d. Gaussian noise (std σ, measured in quantization
steps) → FIR filter → optional additive uniform dither on [-½, ½) → round to
integers. The app shows the filter (convolution kernel and frequency response,
with cutoffs in Hz against a chosen sample rate), a window of the generated
integer signal (stationary by default, with a play toggle to let it stream
endlessly), and the measured compression of a block of the generated integers
under nine methods — zlib, zstd, and an rANS entropy coder, each raw,
delta-coded, and LPC-residual-coded — as bits per sample and as ratio against
raw int16 storage. The predictor order and the block size are controls, so the
measurement can be pushed from 10k to a million samples and LPC from order 1
to 128. Each prefilter group also carries a hollow bar: the order-0 entropy of
the stream being coded, the limit a per-sample entropy coder cannot beat, which
ANS misses by 1–2% (its symbol table plus its own arithmetic loss).

Alongside the measurements it plots a theoretical bits/sample R — the smaller
of a spectral estimate and a rigorous one-sample ceiling:

```
R     = min(Rspec, Rsamp)
Rspec = ∫₀¹ ½ log₂( 2πe (S(f) + ν) ) df    S(f) = σ²|H(f)|², ν = 1/12 (1/6 dithered)
Rsamp = H( round(N(0, v) [+ U(-½,½) with dither]) )    v = σ² Σ h²
```

Rspec is the Zamir–Feder rate of the dithered quantizer counted per Fourier
mode: the signal modes are independent Gaussians of variance S(f), and the
i.i.d. roundoff(+dither) noise is Gaussianized per mode by the CLT, so it
enters at its full variance ν — not at the entropy power 1/(2πe) an aligned
scalar quantizer would charge (the lattice lives in the sample basis; a dead
band inside a live process costs ≈0.25 bits/mode, not zero). At high SNR it
reduces to the Kolmogorov rate ½log₂(2πe σ²) + ∫log₂|H| df. Where the whole
process sits below the quantization step, Rspec bottoms out while the true
rate collapses; subadditivity H(z) ≤ Σ H(zₙ) makes Rsamp — the exact marginal
entropy of one stored sample — a true upper bound with the right collapse,
and the min selects it exactly there. Monte-Carlo puts R within ~0.01–0.02
bits/sample for v ≳ 0.25 (worst ~+0.03 at the branch crossover). LPC + ANS
should approach R; probing where the approximation holds is the point.

## Run it

```sh
npm install
npm run dev
```

## Layout

```
src/model/       the latent source (fixed seeded randomness indexed by sample
                 position, convolved zero-phase with the kernel on demand),
                 FIR presets, and the theoretical-rate formula
src/compress/    lossless codecs run in the browser: zlib (fflate), zstd (wasm),
                 ans.ts (a bit-identical port of simple_ans), and FLAC-style
                 integer LPC; borrowed from entropy-quantized-linear-transform
src/worker/      the codecs run off the main thread on a debounced parameter set
src/components/  controls, filter plots, signal canvas, compression chart
scripts/         true_rate.py — Monte-Carlo ground truth for R (Gibbs over the
                 latent Gaussian given the observed integers, Rao-Blackwellized
                 next-sample pmf); the app prints the exact command to run.
                 Requires numpy only.
```

Every reported size round-trips through the decoder and includes whatever the
decoder needs (ANS symbol table, LPC coefficients). The signal view and the
compression block read the same fixed latent noise sequence — parameter changes
transform the same underlying data rather than resampling it, and the first
window shown is the start of the block that gets compressed.
