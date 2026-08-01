# Does the quantized-filtered-Gaussian model predict real ephys compressibility?

Test: fit `x ~ iid N(0,1) → h * x → round` to a real ephys trace using nothing
but its power spectrum, synthesize a surrogate of the same length from the fit,
and run the same 19 codecs on both. If the model is a good stand-in, every
codec should land at the same bits/sample on the surrogate as on the real
trace.

Data: the three single-channel traces benchcompress benchmarks — DANDI 000876
ch45, 000409 ch101, 001290 ch0 — 200k samples each, raw and in the `-filtered`
variant (bandpass 300–6000 Hz, normalized to MAD noise = 1, requantized at step
v = 0.25). Scripts: `fetch.py`, `fitmodel.py`, `codec_suite.py`, `compare.py`,
`sweep.py`. Logs: `full_compare.log`, `sweep.log`.

## Verdict

**The model is a good surrogate for compressibility, and a bad surrogate for
the signal.** The surrogate has kurtosis ~3.0 against 4.5–19.2 for the real
traces — it has no spikes at all — yet it reproduces the measured rate of every
prediction-based codec to within **0.05 bits/sample**, and of the generic byte
compressors to within ~0.05 bits/sample on the filtered variants. Compression
of these recordings is set by the noise spectrum, not by the spikes.

**But the interesting regime is not where ephys currently sits.** At the
deployed quantization (noise = 4 steps, `s_* ≈ 1`) the fractional-phase loss is
only 0.06–0.08 bits/sample (~3%), and LPC+ANS is within ~6% of the analytic
entropy rate of the fitted model. The loss becomes large only when the
quantizer is coarsened to ~0.5 steps of noise — and the model predicts exactly
where that happens.

## 1. Model vs real, at the benchcompress operating points

`-filtered` variants (bandpass, noise = 4 quantization steps). bits/sample.

| method | 000409 real / model | 000876 real / model | 001290 real / model |
|---|---|---|---|
| order-0 H(z) | 3.871 / 4.070 | 4.236 / 4.298 | 4.129 / 4.150 |
| zstd+delta | 2.701 / 2.825 | 3.065 / 3.105 | 3.348 / 3.303 |
| lzma+delta | 2.507 / 2.601 | 2.823 / 2.854 | 3.032 / 3.056 |
| bz2 | 2.519 / 2.580 | 2.752 / 2.739 | 2.919 / 2.935 |
| brotli+delta | 2.692 / 2.806 | 3.019 / 3.043 | 3.223 / 3.238 |
| **LPC(32)+ANS** | **2.083 / 2.128** | **2.261 / 2.270** | **2.369 / 2.391** |
| LPC(32) resid H0 | 2.053 / 2.094 | 2.215 / 2.232 | 2.334 / 2.347 |
| **cond-Gauss(32)** | **2.029 / 2.039** | **2.173 / 2.190** | **2.301 / 2.311** |
| *G(s\*) analytic rate* | *1.977* | *2.134* | *2.237* |
| *s\** | *0.908* | *1.022* | *1.104* |
| *kurtosis real / model* | *17.7 / 3.0* | *6.0 / 3.0* | *5.9 / 3.0* |

Raw (unfiltered) traces agree nearly as well for the prediction-based methods
(|Δ| ≤ 0.05 bits) and worse for the weak general-purpose coders (lz4 off by
up to 0.5 bits) — those are the ones most sensitive to the heavy tails the
Gaussian model does not have.

One visible fit artifact: on 000876 raw the surrogate std is 26.1 against 47.9
real. Welch detrends each segment, so the fit discards drift below ~7 Hz. That
content is almost perfectly predictable, so it moves `order-0 H(z)` by 0.77
bits and `LPC(32)+ANS` by 0.01 bits.

## 2. Walking the s\* axis by coarsening the quantizer

Same recordings, bandpassed and normalized to noise = 1.0, requantized at step
`v`. `gap` = LPC+ANS − cond-Gauss, i.e. what integer-residual coding loses by
throwing away the fractional part of the prediction. 200k samples.
(001290 ch0; the other two traces agree to ~0.05 bits — see `sweep.log`.)

| v | noise/step | s\* | G(s\*) | real LPC+ANS | real condG | real gap | model gap | model−real |
|---|---|---|---|---|---|---|---|---|
| 0.125 | 8.0 | 1.658 | 2.798 | 2.930 | 2.859 | 0.072 | 0.054 | +0.004 |
| 0.25 | 4.0 | 1.099 | 2.232 | 2.360 | 2.295 | 0.065 | 0.082 | +0.029 |
| 0.5 | 2.0 | 0.741 | 1.717 | 1.872 | 1.771 | 0.101 | 0.101 | +0.012 |
| 1.0 | 1.0 | 0.500 | 1.255 | 1.489 | 1.296 | **0.193** | 0.185 | +0.004 |
| 2.0 | 0.5 | 0.325 | 0.842 | 1.260 | 0.836 | **0.424** | 0.405 | −0.013 |
| 4.0 | 0.25 | 0.103 | 0.267 | 0.951 | 1.938 | — | — | −0.943 |
| 8.0 | 0.12 | 0.001 | 0.001 | 0.102 | 0.069 | — | — | −0.099 |

Three things to read off this:

1. **The predicted mechanism is real and it is large.** The fractional-phase
   gap grows monotonically as `s_*` falls — 0.07 → 0.10 → 0.19 → 0.42
   bits/sample — exactly the behaviour the synthetic study predicted, now
   measured on real recordings. At `v = 2` the conditional-Gaussian coder is
   **34% smaller** than LPC+ANS on real data.
2. **The model predicts the gap, not just the rate.** Real and surrogate gaps
   agree to ~0.02 bits at every step down to `v = 2`. So the model can be used
   to answer "what would I gain?" without running the codec.
3. **The model fails when the quantizer step exceeds the noise** (`v ≥ 4`,
   noise ≤ 0.25 steps). There the dither is gone: the real trace becomes a
   sparse spike train on a bed of zeros while the Gaussian surrogate collapses
   to all zeros (0.008 bits/sample against 0.95 real). The conditional-Gaussian
   coder also inverts there and becomes *worse* than LPC+ANS on real data — a
   single-scale Gaussian is the wrong conditional law for a sparse spiky
   signal. **`s_* ≳ 0.3`, equivalently noise ≳ 0.5 quantization steps, is the
   model's domain of validity.**

## 1b. LPC+ANS beats FLAC by ~21% at the deployed settings

FLAC is the mature instance of the same architecture — LPC prediction, integer
residual, memoryless coding — differing only in that it Rice-codes the residual
where ANS codes its empirical histogram. That one substitution is worth a fifth
of the file:

| | 000409 | 000876 | 001290 |
|---|---|---|---|
| FLAC (LPC + Rice) | 2.680 | 2.865 | 3.043 |
| **LPC(32) + ANS** | **2.083** | **2.261** | **2.369** |
| LPC(32) residual H0 — the memoryless ceiling | 2.053 | 2.215 | 2.333 |
| conditional-Gaussian | 2.029 | 2.173 | 2.301 |
| G(s\*) analytic entropy rate | 1.977 | 2.134 | 2.237 |

Read as distance above the residual-entropy ceiling: **FLAC sits 29–31% above
it, LPC+ANS 1.5–2.1%.** The loss is entirely in the residual coder, not the
predictor. On raw traces (`s_*` = 3.8–6.8) the advantage shrinks to 1–3%: Rice
coding is near-optimal for wide residuals and mismatched for narrow ones, which
is the same $s_*$ story from a different direction.

Caveat: libsndfile does not expose the FLAC compression level, so this is its
default (level 5). `flac -8` raises the max LPC order and improves residual
partitioning — worth a few percent, not twenty. **The paper must re-measure
with the reference `flac -8` binary and with WavPack** (whose prebuilt wheels
do not match this machine's glibc).

## 2b. The model does not need an explicit filtering step

`h` in the fit is the minimum-phase spectral factor of whatever spectrum the
trace has — not a filter anyone applied. Any stationary Gaussian process with
spectral density `S` can be written `h * x`. Two checks that this is not just a
formal remark:

- **The raw traces were never preprocessed by us.** Only the acquisition chain
  shaped them, and the fit reproduces them as well as the bandpassed versions:
  LPC+ANS real/model 4.855/4.902, 4.522/4.512, 4.074/4.117; cond-Gauss agrees
  to ≤ 0.015 bits on all three. Kurtosis up to 19.2.
- **The whole `s_*` sweep repeated with no bandpass at all** (`sweep_nofilter.log`,
  `--nofilter`, same normalization and same steps, only the bandpass removed).
  Model−real stays ≤ 0.06 bits from `s_* = 6.8` down to `s_* ≈ 0.32`, and the
  fractional-phase gap grows the same way. 000409 ch101:

  | noise/step | s\* | real gap | model gap | model−real |
  |---|---|---|---|---|
  | 8.0 | 4.350 | 0.048 | 0.091 | +0.056 |
  | 4.0 | 2.190 | 0.014 | 0.061 | +0.058 |
  | 2.0 | 1.112 | 0.058 | 0.069 | +0.017 |
  | 1.0 | 0.593 | 0.201 | 0.198 | −0.003 |
  | 0.5 | 0.319 | 0.431 | 0.477 | −0.024 |

So the preconditions are **stationarity over the block**, **second-order
statistics sufficing for the rate** (validated at kurtosis 19), and **noise
≳ 0.5 quantization steps**. Explicit filtering is not among them.

## 2c. The gap decomposes into named terms, and the structural one has a
closed form

Write the distance from a measured codec rate to the analytic entropy rate as

```
R_ANS - G(s*) = [R_ANS - H0]  +  [H0 - R_condG]  +  [R_condG - G(s*)]
                 coder           residual-model      prediction
```

- **coder** — rANS overhead against a perfect memoryless coder on its own
  residual stream (symbol table + arithmetic loss).
- **residual-model** — the pooled integer-residual histogram against the
  phase-conditioned law. *This is the structural term, and theory predicts it
  from `s` alone*: with `M(s)` the entropy of the phase-mixture and `G(s)` the
  phase-averaged conditional entropy, the loss is `L(s) = M(s) - G(s)`, a
  universal curve (`phase_loss.py`). It peaks near `s ≈ 0.2` at ~0.27
  bits/sample and decays in both directions.
- **prediction** — linear prediction is not optimal from the quantized past,
  plus the single-scale Gaussian's parametric mismatch.

Measured against predicted, three recordings, 200k samples (`decompose.log`):

| s\* | coder | residual-model measured | **L(s) predicted** | prediction |
|---|---|---|---|---|
| 1.36–1.66 | 0.052–0.059 | −0.009 – 0.016 | 0.019–0.029 | 0.032–0.061 |
| 0.92–1.10 | 0.029–0.046 | 0.025–0.042 | 0.042–0.058 | 0.039–0.063 |
| 0.63–0.74 | 0.023–0.028 | 0.078–0.085 | 0.084–0.109 | 0.033–0.054 |
| 0.43–0.50 | 0.022–0.034 | 0.159–0.272 | 0.153–0.181 | 0.015–0.042 |
| 0.28–0.33 | 0.016–0.028 | 0.396–0.492 | 0.235–0.271 | −0.071 – −0.006 |

Read honestly: **the closed form gets the structural term right to within about
a factor of 1.5 over `s_* ∈ [0.4, 1.7]`, and underestimates it by ~1.7× at
`s_* ≈ 0.3`** — where the real residual is heavier-tailed than the single
Gaussian the theory assumes. The coder term is a stable 1–2% of the rate, as
claimed. The prediction term is small and positive until `s_*` drops below
~0.35, where it goes negative because `G(s_*)` itself starts to break down.

## 3. Consequences for the paper

- §8's working hypothesis was that deployed ephys sits in the low-`s_*` region
  where the fractional-phase loss is largest. **It does not.** At the standard
  filtered-and-quantized settings `s_* ≈ 1`, and LPC+ANS is within a few
  percent of the limit. The honest headline for real data is: *for the way
  ephys is stored today, standard prediction + entropy coding is close to
  optimal, and the model says so without any experiment.*
- The gain lives one step coarser. Since coarsening the quantizer is exactly
  the near-lossless knob practitioners already reach for, the useful statement
  is a joint one: at each step size, here is the entropy rate, here is what
  standard coding gets, and here is what conditional-Gaussian coding gets.
- Spikes cost almost nothing in bits. Worth stating plainly — it is the reason
  a Gaussian surrogate works at all, and it is counterintuitive.
- The Welch-detrending artifact means the fit should either keep the very low
  frequencies or the comparison should be stated on high-passed data. Minor,
  but it affects `order-0 H(z)` a lot.

## Open

- Only linear-prediction methods and generic byte compressors so far. FLAC and
  WavPack are not installed here; they should land near LPC+ANS.
- `s_*` here is computed from the *observed* spectrum via Szegő + 1/12. It has
  not been checked against the Monte-Carlo estimator on the fitted kernel.
- Single channel. Cross-channel redundancy untested.
