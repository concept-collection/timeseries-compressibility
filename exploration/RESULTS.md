# Why LPC+ANS gets 14x when the entropy rate says 23x

Setting: `--sigma 5 --filter bandpass --low 300 --high 2000 --taps 31 --rate 30000`.
All numbers are bits/sample on the quantized signal z (ratio = 16 / bits).

## Verdict

**The entropy estimate is right, and the gap is real.** LPC+ANS is not
suboptimal by accident — integer-residual prediction followed by a memoryless
entropy coder structurally cannot reach the conditional entropy of this
process. A codec that codes each sample against a discretized Gaussian
centred at the *real-valued* prediction reaches 22.3x with a verified
lossless round-trip; a Monte-Carlo model-based codec reaches ~23.7x.

## The mechanism

After optimal linear prediction from the quantized past, the prediction error
of y_next has std s ~= 0.28 quantization steps. The conditional distribution
of z_next = round(y_next) is therefore a Gaussian bump covering ~1-3 integer
bins, and its shape depends strongly on the *fractional part* of the
prediction mu: mu near a bin centre gives ~0.4 bits, mu near a bin edge gives
~1.1 bits.

- An ideal conditional coder uses mu per sample: mean entropy ~= **0.72 bits**.
- LPC+ANS codes the integer residual z - floor-prediction with one pooled
  histogram: that histogram is the *mixture* over all fractional parts,
  entropy ~= **1.07 bits** (semi-analytic; 1.10 measured with the app's
  15-bit integer coefficients).

The ~0.35-0.38 bit gap **is** the 14x-vs-22x discrepancy. Raising the LPC
order does nothing (14.6x plateau by order 64): the loss is in the residual
coding, not the prediction.

## Measurements (4M-sample realizations; codecs charged for all side info)

| method | bits/sample | ratio |
|---|---|---|
| raw int16 | 16 | 1x |
| order-0 entropy of z | 1.954 | 8.2x |
| LPC(32) + ANS (app's method, reproduced) | 1.101 (+~1.5% ANS) | 14.3x |
| LPC(128) + ANS ceiling | 1.098 | 14.6x |
| **conditional-Gaussian arithmetic codec (real bytes, exact round-trip)** | **0.7184** | **22.27x** |
| Gibbs posterior-predictive codec (achievable rate, 60 sweeps) | 0.674 +/- 0.065 | 23.7x |
| timeseries-entropy estimator, M=512 (default settings) | 0.615 +/- 0.065 | 26x |
| timeseries-entropy estimator, M=1024, thin=2 | 0.658 +/- 0.046 | 24.3x |
| semi-analytic conditional entropy (rounding-as-noise model) | 0.723 | 22.1x |

The true entropy rate is bracketed: <= 0.718 is *proven* by real compressed
bytes that round-trip; ~0.674 is achievable by the model-based codec; the
Monte-Carlo estimator's 0.62-0.68 is consistent with both. The estimator's
Rhee-Glynn pilot (`--pilot 6`) shows RMS Delta_m <= 0.016 decaying at
exponent ~2, so its r=1.5 truncation is sound at these settings.

## The codec that proves it (`codec_gaussian.py`)

1. Fit an order-64 real-coefficient predictor on the block (Levinson);
   send coefficients as float32 in the header (270 B total).
2. Fit a single residual scale s (grid search on the block); send it too.
3. For each sample: mu_t = dot(coeffs, previous 64 decoded samples);
   code z_t with an arithmetic coder under P(k) proportional to
   Phi((k+1/2-mu_t)/s) - Phi((k-1/2-mu_t)/s), tails folded into edge bins,
   16-bit frequencies. Decoder computes the identical mu_t from decoded
   samples, so tables agree bit for bit.
4. First 64 samples are coded under the marginal N(0, std_z) table.

1,048,576 samples -> 94,157 bytes = 0.7184 bits/sample = **22.27x**,
`np.array_equal(z, decoded) == True`. Encode/decode ~7 s each in pure
Python+numpy.

`gibbs_predictive_rate.py` pushes further: both sides could run the
timeseries-entropy Gibbs sampler on the decoded past (shared seed) and code
against the Rao-Blackwellized posterior predictive of the generative model —
the nonlinear optimum. Its measured cross-entropy is 0.674 +/- 0.065
bits/sample (23.7x). The -log2-of-a-mean is Jensen-biased upward, so more
sweeps would only lower it toward the true conditional entropy.

## Files

- `analysis_spectral.py` — semi-analytic model: Kolmogorov innovation
  variance of S_z = sigma^2|H|^2 + 1/12; conditional vs mixture entropy;
  empirical Wiener-predictor rates by order.
- `lpc_ans_repro.py` — exact port of the app's integer LPC (15-bit coeffs,
  shared shift, floor prediction, int16 wrap); reproduces 14.3x.
- `codec_gaussian.py` — the round-tripping conditional-Gaussian codec.
- `gibbs_predictive_rate.py` — achievable rate of the Monte-Carlo
  model-based codec, reusing timeseries_entropy.ConditionalChain.
- `entropy_cli_run.log`, `entropy_pilot.log`, `codec_gaussian_1M.log`,
  `gibbs_predictive.log` — raw outputs.

## Notes for the app / estimator

- The app's hollow "order-0 entropy of the residual" bar is the LPC+ANS
  *ceiling*, and the measured 14x sits right on it — the coder is fine; the
  prefilter family is the limit.
- The tenth method now exists in the app: `src/compress/conditionalGaussian.ts`,
  computed with the other nine in the compression worker and drawn as its own
  "conditional" chart group. In-browser at these settings, 100k block:
  LPC(32)+ANS 1.119 bits (14.30x) vs cond. Gaussian AC 0.738 bits (21.68x).
- With prediction-error std s, the fractional-phase loss of integer-residual
  coding is ~E[H(mix)] - E[H(cond)]; it vanishes for s >> 1 (white/large
  sigma) and grows as s drops below ~0.5 — worst exactly in the
  narrowband, moderate-sigma regime this parameter set sits in.
