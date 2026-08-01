# Paper outline

## Title

The entropy rate of a noisy quantized signal and how far lossless codecs sit from it, with application to extracellular voltage recordings

## Thesis

The entropy rate of this signal class is computable. Here is where real codecs
sit relative to it.

**Scope decisions.** The unbiased Monte-Carlo estimator is the contribution;
the analytic formula is a fast approximation validated against it. The paper is
descriptive — *why* codecs fall short (the gap decomposition) is a separate
paper. Target ~10–12 pages: estimator construction and the unbiasedness
statement in the body, proof in an appendix.

**The estimator earns trust three ways:** it is provably unbiased; it agrees
with the theoretically justified analytic formula; and advanced codecs approach
it from above.

---

## 1. Introduction

- Instruments emit quantized samples of a noisy continuous process. Lossless
  compression is a real cost, and the practice is empirical — try codecs, keep
  the winner. Nobody knows how much is left.

### 1.1 Prior work

*Organized around what each body of work supplies and what it leaves open.*

- **Ephys compression benchmarks.** Buccino et al. benchmarked general-purpose
  and audio codecs on large-scale Neuropixels data, finding the audio codecs
  (WavPack, FLAC) outperform general-purpose compressors and that lossy modes
  buy substantial further ratio without measurably harming spike sorting. It is
  the reference point for this paper — **and it is codec-versus-codec.** With
  no entropy-rate estimate, "best" can only mean best of those tried, and the
  distance to what is achievable is unknown. Same for the surrounding
  literature on ephys and scientific-array compression (Blosc, Zarr/HDF5 filter
  pipelines).
- **Lossless audio coding.** FLAC, Shorten, WavPack, MPEG-4 ALS — the
  predict-then-code-the-residual architecture the ephys benchmarks inherit, and
  its Rice/Golomb residual coding.
- **Entropy-rate estimation.** Plug-in block entropy and its downward bias;
  Lempel–Ziv and compression-based estimators. The last are upper bounds of
  unknown tightness, so they cannot serve as the reference against which
  compressors are judged — using a compressor to estimate the limit that a
  compressor is being measured against begs the question.
- **Unbiased MCMC.** Rhee–Glynn randomized truncation; Jacob et al. — the
  machinery §3 builds on.
- **Quantization theory.** Bennett's dither model, Gray's quantization theory,
  high-resolution limits; the source of the 1/12 correction in §4.

### 1.2 This paper

- The missing quantity is the entropy rate $\bar H$. For this class it is not
  analytically available (rounding destroys Gaussianity) and not reliably
  estimable by plug-in methods (wide alphabet, long memory).
- We give an unbiased estimator of $\bar H$, an analytic approximation that
  agrees with it, and a benchmark of standard codecs against it — on synthetic
  sources with an exact limit, and on real extracellular recordings, both raw
  wideband and bandpass-filtered.
- Headline numbers to preview.
- Contributions.

## 2. Setup

- $z_t = \operatorname{round}(y_t)$, $y$ stationary Gaussian with PSD $S(f)$,
  unit quantization step. Amplitude in quantization steps; $S$ says everything.
- $y = h * x$ is the spectral-factorization form — $h$ is the minimum-phase
  factor of $S$, not a filter anyone applied.
- Estimand: $\bar H = \lim_{M\to\infty} H(z_{M+1} \mid z_1..z_M)$.
- Accounting: every reported size is the full encoded byte count, includes side
  information, and round-trips.

## 3. An unbiased estimator of the entropy rate

- **Why naive estimation fails**: plug-in bias, wide alphabet, long memory.
- **Stationary conditional sampling.** Gibbs on the latent $x$ under rounding
  constraints; started from the generating $x$, which is an exact draw from
  $p(x \mid z)$, so no burn-in bias. Each sweep emits one exact sample of
  $z_{M+1}$.
- **Unbiased entropy of the chain's marginal** by Rhee–Glynn randomized
  telescoping with antithetic half-block corrections. Statement of
  unbiasedness; proof in Appendix A.
- **Averaging over independent pasts** gives a valid standard error; the only
  remaining approximation is finite $M$.
- Diagnostics and cost. *(Table 1)*

## 4. An analytic approximation

- Szegő–Kolmogorov, its two failures (spectral nulls; only the quantized past
  is observed), and the 1/12 dither fix giving
  $s_*^2 = \exp\int \ln(S + 1/12) - 1/12$.
- $\bar H \approx G(s_*)$, the entropy of Gaussian ⊕ unit uniform.
- $s_*$ is computable from any signal's periodogram — no fit, no sampling.
- **Agreement with §3 across the regime map.** *(Figure 1)* Where it degrades
  and why.

## 5. Benchmark on synthetic sources

- Codec families, all round-trip verified, all charged for side information:
  general-purpose byte compressors (± delta, ± shuffle); lossless audio
  (FLAC, WavPack, …); array/neuro pipelines; transform coders.
- **Ratio-to-limit across the parameter space.** *(Figure 2)* Which families
  win where; how far the best codec is from $\bar H$.
- **5.x LPC+ANS.** Integer LPC with rANS over the empirical residual
  histogram. Absent from the ephys benchmarking literature; measures closest
  of the practical codecs. Method stated completely enough to reimplement.
- **Advanced codecs approach the limit.** A conditional-Gaussian arithmetic
  coder emits bytes that decode at 0.718 bits/sample where the estimator says
  0.62–0.68 — an upper bound proven independently of §3. Impractical, included
  as evidence. *(Table 2: the ladder.)*

## 6. Extracellular voltage recordings

**Both signals are in scope**: the raw wideband trace as acquired, and the
bandpass-filtered trace prepared for spike sorting. Both are stored and shared
in practice, and they sit in different regimes — measured $s_*$ is 3.8–6.8 raw
against 0.91–1.10 filtered — so the codec ranking and the size of the shortfall
differ between them. Report the two side by side throughout.

### 6.1 The limit transfers

- Fit the model to three DANDI traces, raw and filtered, from their power
  spectrum alone.
- The surrogate has no spikes (kurtosis 3.0 vs 4.5–19.2) yet reproduces every
  prediction-based codec to within 0.05 bits/sample, on both variants. Note
  that the raw traces were never preprocessed — the fitted $h$ is a spectral
  factor, not a filter anyone applied. Compressibility is set by the noise
  spectrum, not the events.
- Domain of validity: noise $\gtrsim$ half a quantization step.

### 6.2 Where deployed codecs sit

- Compute the limit for each recording and variant; measure what is actually
  used; report the shortfall. *(Figure 3, two panels: raw and filtered.)*
- The two variants give different answers, and the difference is worth stating
  plainly: on filtered traces LPC+ANS is ~21% below FLAC, while on raw traces
  the two are within 1–3%.
- Behaviour as the quantizer is coarsened — the near-lossless knob
  practitioners already use.

## 7. Discussion and limitations

Stationarity and the noise condition are the binding assumptions — Gaussianity
and filtering are not. Single channel. What the paper does not claim: no codec
recommendation, no explanation of the shortfall.

## 8. Artifacts

`timeseries-entropy` (estimator, theory, CLI); benchmark harness and codecs;
interactive browser app; DANDI provenance.

---

## Figures and tables

| # | Content |
|---|---|
| Fig 1 | Analytic $G(s_*)$ vs Monte-Carlo $\bar H$ across the regime map, with error bars |
| Fig 2 | Ratio-to-limit for each codec family across the synthetic parameter space |
| Fig 3 | Real recordings: measured codec rates against the fitted-model limit |
| Tab 1 | Estimator diagnostics: pilot decay, $M$-convergence, cost |
| Tab 2 | The ladder at a reference point: raw → order-0 → FLAC → LPC+ANS → cond-Gaussian → $\bar H$ |

## Appendices

- **A.** Rhee–Glynn telescoping: unbiasedness proof, variance conditions.
- **B.** Gibbs conditionals for the box-truncated latent.
- **C.** Derivation of $G(s)$.
- **D.** Conditional-Gaussian codec spec.
- **E.** Spectral fitting for §6.
- **F.** Codec configurations and versions.
