# timeseries-compressibility

Interactive exploration of how compressible quantized time series are.

The generating model is: i.i.d. Gaussian noise (std σ, measured in quantization
steps) → FIR filter → optional additive uniform dither on [-½, ½) → round to
integers. The app shows the filter (convolution kernel and frequency response,
with cutoffs in Hz against a chosen sample rate), an endlessly scrolling view of
the generated integer signal, and the measured compression of a 120,000-sample
block under nine methods — zlib, zstd, and an rANS entropy coder, each raw,
delta-coded, and LPC-residual-coded — as bits per sample and as ratio against
raw int16 storage.

Alongside the measurements it plots the theoretical bits/sample: the entropy
rate of the stationary filtered Gaussian process quantized at unit step, in the
high-resolution limit,

```
R = ½ log₂(2πe) + ∫₀^½ log₂ S(f) df,    S(f) = σ² |H(f)|²
```

LPC + ANS should approach R — and does, where the formula is valid. Where the
filter's stopband pushes S(f) below one step² (see the dashed threshold on the
response plot), the formula under-predicts and can go negative; making that
breakdown visible is part of the point. The math section is a stub for the full
derivation.

## Run it

```sh
npm install
npm run dev
```

## Layout

```
src/model/       the pipeline (seeded Gaussian stream, FIR presets, dither,
                 rounding) and the entropy-rate integral
src/compress/    lossless codecs run in the browser: zlib (fflate), zstd (wasm),
                 ans.ts (a bit-identical port of simple_ans), and FLAC-style
                 integer LPC; borrowed from entropy-quantized-linear-transform
src/worker/      the codecs run off the main thread on a debounced parameter set
src/components/  controls, filter plots, scrolling canvas, compression chart
```

Every reported size round-trips through the decoder and includes whatever the
decoder needs (ANS symbol table, LPC coefficients). The compression block and
the scrolling display are fed by the same `Pipeline`, so what is compressed is
what is shown.
