# timeseries-compressibility

Interactive exploration of how compressible quantized time series are.

The generating model is: i.i.d. Gaussian noise (std σ, measured in quantization
steps) → FIR filter → optional additive uniform dither on [-½, ½) → round to
integers. The app shows the filter (convolution kernel and frequency response,
with cutoffs in Hz against a chosen sample rate), a window of the generated
integer signal (stationary by default, with a play toggle to let it stream
endlessly), and the measured compression of a 120,000-sample
block under nine methods — zlib, zstd, and an rANS entropy coder, each raw,
delta-coded, and LPC-residual-coded — as bits per sample and as ratio against
raw int16 storage.

Alongside the measurements it plots a theoretical bits/sample R: quantization
is modeled as an additive white noise floor on the spectrum, the one-step
Wiener prediction error of the resulting process comes from the
Szegő–Kolmogorov formula, and R is the exact entropy of that innovation
quantized at unit step:

```
S_z(f) = σ²|H(f)|² + σ_q²         σ_q² = 1/12 (1/6 with dither)
σ_e²   = exp( 2 ∫₀^½ ln S_z(f) df )
R      = H_Δ(σ_e)                 (exact quantized-Gaussian entropy)
```

Where the spectrum sits well above one step² this reduces to the classical
Gaussian entropy rate ½log₂(2πe) + ∫log₂S df; the noise floor keeps it finite
and positive where a deep stopband would send that integral to −∞. LPC + ANS
should approach R; probing where the approximation holds is the point. The
math section is a stub for the full derivation.

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
```

Every reported size round-trips through the decoder and includes whatever the
decoder needs (ANS symbol table, LPC coefficients). The signal view and the
compression block read the same fixed latent noise sequence — parameter changes
transform the same underlying data rather than resampling it, and the first
window shown is the start of the block that gets compressed.
