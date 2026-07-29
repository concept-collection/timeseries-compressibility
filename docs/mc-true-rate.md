# Monte Carlo estimation of the true entropy rate of a quantized filtered Gaussian process

**Purpose.** This document specifies, in enough detail for independent
implementation and testing (Python + numpy/scipy is the intended target), a
Monte Carlo method for estimating the *exact* entropy rate R of the integer
process produced by the pipeline below — the quantity that the closed-form
formula in the companion app ([timeseries-compressibility](../README.md))
only approximates. The estimator is consistent (converges to the true R as
particle count, window length, and sample count grow), all of its systematic
errors push in a known direction (upward), and it admits a two-sided
sandwich so convergence can be certified rather than assumed.

No code from the app needs to be reused; everything required is specified
here. Reference values measured with the app are given in §7 for
cross-checking.

---

## 1. The system under study

### 1.1 Generating model

$$x_n \sim \mathcal{N}(0,\sigma^2)\ \text{i.i.d.}, \qquad
y = h * x, \qquad
z_n = \operatorname{round}(y_n + d_n)$$

- $h_0,\dots,h_{L-1}$ is a known FIR kernel (designs in §1.2).
- $d_n \sim \mathcal{U}[-\tfrac12,\tfrac12)$ i.i.d. when **dither** is on;
  $d_n = 0$ otherwise. Dither is *non-subtractive*: the stored integers
  include it and the decoder never learns $d$.
- Quantization step = 1 (σ is measured in steps). Rounding ties are a
  measure-zero event, so the tie convention does not affect any law.
- $y$ is a stationary Gaussian process with autocovariance

  $$r_y(\ell) = \sigma^2 \sum_{j} h_j\, h_{j+\ell}
  \qquad (r_y(\ell)=0 \text{ for } |\ell| \ge L)$$

  and spectral density $S(f) = \sigma^2 |H(f)|^2$,
  $H(f) = \sum_n h_n e^{-2\pi i f n}$, $f \in [0,\tfrac12]$ cycles/sample.

The object of interest is the **entropy rate of the integer process** $z$:

$$R \;=\; \lim_{k\to\infty} H\!\left(z_0 \mid z_{-1},\dots,z_{-k}\right)
\qquad \text{[bits/sample]}$$

This is the true lossless compression limit for $z$, the number the app's
LPC + ANS bars try to approach from above.

### 1.2 Kernel designs (must match to compare numbers)

All comparisons in §7 use the **bandpass** kernel below; implement at least
`none` and `bandpass`.

- `none`: $h = [1]$ (then $z$ is i.i.d. and everything has closed form —
  the primary validation case).
- Hamming-windowed sinc lowpass, cutoff $f_c$ cycles/sample, odd length $n$,
  midpoint $m = (n-1)/2$:

  $$h_i = w_i \cdot \begin{cases} 2f_c & i = m \\
  \sin(2\pi f_c (i-m)) / (\pi (i-m)) & i \ne m \end{cases},
  \qquad w_i = 0.54 - 0.46\cos\!\left(\tfrac{2\pi i}{n-1}\right)$$

  then normalized to unit DC gain: $h \leftarrow h / \sum_i h_i$.
- `bandpass(f_1, f_2)` = lowpass($f_2$) − lowpass($f_1$), same length, no
  further normalization.
- The **default ephys case**: bandpass 300–6000 Hz at 30 kHz sample rate
  ($f_1 = 0.01$, $f_2 = 0.2$), 101 taps.
  Checkable constant: $\lVert h \rVert_2 = 0.60216$, so the filtered std is
  $\sigma_y = 3.011$ at $\sigma = 5$.

### 1.3 The approximate formula being tested (for comparison only)

The app's closed-form estimate, to be reimplemented for the comparison
plots:

$$S_z(f) = S(f) + \sigma_q^2, \qquad
\sigma_e = 2^{\int_0^{1/2} \log_2 S_z(f)\,df}, \qquad
R_{\text{approx}} = H_\Delta(\sigma_e)$$

with $\sigma_q^2 = 1/12$ (no dither) or $1/6$ (dither), the integral by
midpoint rule (8192 points suffices), and $H_\Delta(s)$ the exact entropy of
$\operatorname{round}(\mathcal N(0,s^2))$ on the unit lattice:
$H_\Delta(s) = -\sum_z p_z \log_2 p_z$,
$p_z = \Phi(\tfrac{z+1/2}{s}) - \Phi(\tfrac{z-1/2}{s})$.

Known inexactness (details in the analysis that motivated this work):

- **It is not a bound in either direction.**
- In the i.i.d. case (`none`) the truth is exactly $H_\Delta(\sigma)$, while
  the formula returns $H_\Delta(\sqrt{\sigma^2 + 1/12})$ — an overestimate
  (a double-count of the quantization variance): +0.0024 bits at σ = 5,
  +0.171 at σ = 0.5, +0.576 at σ = 0.1 (where the truth is ≈ 0).
- In the stopband-dominated regime (default bandpass, σ = 5) it lands
  remarkably close to achievable rates, plausibly partly by cancellation of
  errors; the MC estimator exists to resolve this.

---

## 2. Why Monte Carlo can be exact here

Three facts combine:

1. **The law of $z$ is fully known**, so conditional probabilities can be
   *computed*, not estimated from frequencies. For a window
   $z_{1:m}$, the event probability (no dither) is a Gaussian rectangle
   probability:

   $$P(z_{1:m}) = \Pr\!\left[\, y_t \in [z_t - \tfrac12,\, z_t + \tfrac12)
   \ \forall t \le m \,\right]$$

   — an $m$-dimensional box integral of $\mathcal N(0, \Sigma_y)$ with the
   banded covariance from $r_y$. (With dither the box edges are shifted by
   the $d_t$, which are handled by sampling; §4.4.)

2. **Predictive decomposition.**
   $-\log_2 P(z_{1:m}) = \sum_t -\log_2 P(z_t \mid z_{1:t-1})$, and by
   Shannon–McMillan–Breiman, $-\tfrac1m \log_2 P(z_{1:m}) \to R$ almost
   surely along a single simulated realization. Equivalently, averaging the
   conditional surprisal $-\log_2 P(z_0 \mid z_{-k:-1})$ over independent
   draws estimates $h_k = H(z_0 \mid z_{-k:-1})$, and $h_k \downarrow R$
   monotonically.

3. **Monotone error structure.** Every systematic error is upward:
   - finite window: $h_k \ge R$ for every $k$ (conditioning reduces
     entropy);
   - inner estimation: an unbiased $\hat P$ gives
     $\mathbb E[-\log_2 \hat P] \ge -\log_2 P$ by Jensen, with bias
     $\approx \operatorname{Var}(\hat P) / (2 P^2 \ln 2)$, vanishing as the
     particle count grows.

   So the estimate converges to R **from above**, and the only downward
   fluctuation is CLT noise, which gets an error bar.

For a two-sided certificate, use the hidden-Markov sandwich
(Cover & Thomas, *Elements of Information Theory*, §4.5): $z$ is a function
of the Markov state $s_t = (x_{t-L+1},\dots,x_t)$ (plus $d_t$), so

$$H\!\left(z_0 \mid z_{-k:-1},\, s_{-k-1}\right) \;\le\; R \;\le\;
H\!\left(z_0 \mid z_{-k:-1}\right)$$

with the lower bound *increasing* in $k$ and the gap
$I(s_{-k-1}; z_0 \mid z_{-k:-1}) \to 0$. In simulation the true state at
$-k-1$ is known (we generated it), so the lower bound is estimable by the
same machinery (§4.5).

---

## 3. The estimator in one paragraph

Simulate one long realization $z_{1:T}$ from the pipeline. Run a sequential
Monte Carlo over the *latent Gaussian path constrained to the observed
boxes*: each particle is a sampled path $y^i$ consistent with
$z_{1:t}$ so far; at each step the exact one-step predictive probability of
the observed box, given the particle's path, is a Gaussian
$\Phi$-difference (because $y_t \mid y_{1:t-1}$ is Gaussian with
linear-prediction mean and fixed innovation variance); the mean of those
$\Phi$-differences across particles is an unbiased estimate of
$P(z_t \mid z_{1:t-1})$; the particle is extended by sampling $y_t$ from
the corresponding truncated Gaussian (this "optimal one-step proposal" is
exactly Genz's separation-of-variables construction for box probabilities,
run sequentially with resampling). Average $-\log_2 \hat P(z_t \mid \cdot)$
over $t$ after a burn-in; repeat over independent replicates for error
bars. That average estimates R.

---

## 4. Algorithm specification

### 4.1 Precomputation

1. Build the kernel $h$ (§1.2) and the autocovariance
   $r_y(0),\dots,r_y(k_{\max})$ (zero beyond lag $L-1$).
2. Run **Levinson–Durbin** on $r_y$ up to order $k_{\max}$ (suggest
   $k_{\max} = 4L$, capped at ~1024). Store, for each order $p \le k_{\max}$
   — or just for the terminal order — the prediction coefficients
   $a^{(p)}_1,\dots,a^{(p)}_p$ and innovation variance $v_p$. After
   $p \gtrsim L$ these are effectively converged ($v_p \to$ the
   Szegő value for $S$ *without* any noise floor; e.g. for the default
   bandpass at σ = 5, $v_\infty \approx 0.0019$, i.e. innovation std
   ≈ 0.044). Guard against underflow of $v_p$ for extreme stopbands (floor
   at ~1e-30 and warn).

   For $t \le k_{\max}$ use the order-$(t-1)$ coefficients; past that, the
   terminal ones (steady state).

### 4.2 Per-step recursion (undithered)

State: particle paths $y^i_{1:t-1}$, $i = 1..N$ (only the last $k_{\max}$
values are needed), all with equal weight after resampling.

For $t = 1, 2, \dots, T$, with observed integer $z_t$:

1. Prediction per particle:
   $\mu^i = \sum_{j=1}^{p} a^{(p)}_j\, y^i_{t-j}$, innovation std
   $s = \sqrt{v_p}$, where $p = \min(t-1, k_{\max})$.
2. Standardized box edges:
   $\alpha^i = (z_t - \tfrac12 - \mu^i)/s$,
   $\beta^i = (z_t + \tfrac12 - \mu^i)/s$.
3. Incremental weight $w^i = \Phi(\beta^i) - \Phi(\alpha^i)$
   (log-space; see §5.1).
4. **Predictive estimate**
   $\hat p_t = \tfrac1N \sum_i w^i$. Record $-\log_2 \hat p_t$. If
   $\hat p_t = 0$ (all particles incompatible), the run has degenerated —
   restart the replicate with larger N; do not clamp.
5. Resample particles with probabilities $\propto w^i$ (systematic
   resampling; adaptive — only when $\mathrm{ESS} < N/2$ — is fine, but then
   $\hat p_t$ must use the standard normalized-weight form
   $\hat p_t = \sum_i W^i_{t-1} w^i_t$ with carried weights $W$).
6. Extend each surviving particle:
   $y^i_t = \mu^i + s\,\Phi^{-1}\!\big(\Phi(\alpha^i) + U^i\,(\Phi(\beta^i)-\Phi(\alpha^i))\big)$,
   $U^i \sim \mathcal U(0,1)$ — the truncated-Gaussian draw (§5.2).

### 4.3 Assembling the estimate

- $\hat R = -\dfrac{1}{T - B}\sum_{t=B+1}^{T} \log_2 \hat p_t$, with burn-in
  $B \ge 5L$ (the early steps estimate $h_{t-1}$ for small $t$, which is
  above R).
- Run $J$ independent replicates (fresh $z$, fresh particles). Report
  mean ± SE across replicates. Within-run block averaging is acceptable but
  replicates are simpler and honest about autocorrelation.
- **Bias control (essential):** repeat at $N$, $2N$, $4N$ particles. The
  estimate must decrease and plateau (Jensen bias shrinks like $1/N$).
  Treat the plateau as the answer; optionally Richardson-extrapolate in
  $1/N$. A per-step delta correction
  $-\log_2\hat p_t \to -\log_2 \hat p_t - \widehat{\operatorname{Var}}_i(w^i) / (2 N \hat p_t^2 \ln 2)$
  is a useful diagnostic but is not exact under resampling — the
  particle-doubling plateau is authoritative.

### 4.4 Dither

With dither, the box for $y_t$ is shifted by the (unknown to the decoder)
$d_t$: $y_t \in [z_t - \tfrac12 - d_t,\, z_t + \tfrac12 - d_t)$. Extend each
particle with its own $d^i_t \sim \mathcal U[-\tfrac12,\tfrac12)$ drawn
*before* step 2, and use the shifted edges. Everything else is unchanged;
$\hat p_t$ remains unbiased (the dither prior is part of the proposal).
Optionally, integrate $d_t$ analytically in the weight — the integral of a
$\Phi$-difference over a unit shift is a difference of
$G(t) = t\Phi(t) + \varphi(t)$ terms — and then sample $(d_t, y_t)$ jointly;
lower variance, more code. Start with the sampled version.

Note: when generating the *observed* $z_{1:T}$ for a dithered run, the
generator also draws $d_t$; those true dither values are **not** given to
the estimator (non-subtractive dither).

### 4.5 Sandwich lower bound (optional but recommended)

Estimate $H(z_0 \mid z_{-k:-1}, s_{-k-1})$: for each of many independent
windows, (a) simulate truth and keep the exact latent inputs up to time
$-k-1$; (b) condition the Gaussian law of $y_{-k:0}$ on that known state —
its conditional mean is the deterministic tail response
$\sum_{j > t+k} h_j x_{t-j}$ and its conditional covariance is that of the
*truncated* kernel (only the taps applied to inputs after $-k-1$); (c) run
the same SOV/SMC over the $k{+}1$ constrained steps and record the last
step's $-\log_2 \hat p$. Average over windows. Plot lower and upper curves
against $k$; the closing gap certifies memory-length convergence
independently of any modeling argument.

---

## 5. Numerical hazards (all known, all manageable)

### 5.1 Φ-differences in the tails

When both edges are far in one tail, $\Phi(\beta)-\Phi(\alpha)$ underflows
in the naive form. Compute in log space:
use `scipy.special.log_ndtr`; for $\alpha, \beta > 0$ switch to the upper
tail, $\log(\Phi(-\alpha) - \Phi(-\beta))$, and combine as
`log_ndtr(hi) + log1p(-exp(log_ndtr(lo) - log_ndtr(hi)))`. These
low-probability steps are precisely the ones that dominate the surprisal,
so they must not be clamped to zero.

### 5.2 Truncated-normal sampling

The inverse-CDF trick in §4.2(6) loses precision when the interval sits
beyond ~6σ. Use `scipy.stats.truncnorm` (which handles tails), or Robert's
exponential-rejection sampler for one-sided extreme tails. A particle
sitting exactly on an edge after sampling is harmless.

### 5.3 Weight degeneracy and path collapse

The innovation std $s$ can be much smaller than the box width (deep
stopbands, no dither). Then most particles get $w^i \approx 1$ and the rest
$\approx 0$: weights are Bernoulli-like with success probability
$\approx 2^{-R}$ per step. This is workable — the truncated draw in
step 6 spans the box, i.e. many innovation-σ's, so resampled duplicates
re-diversify quickly — but it costs particles. If ESS collapses persistently:

- raise N (first resort; the estimator is embarrassingly parallel across
  particles and replicates);
- use a lookahead / auxiliary proposal (weight by the next few
  observations before resampling);
- as a fallback formulation, run the SMC in **x-space** (particles over the
  i.i.d. inputs, observation $z_t$ constraining
  $\sigma \sum_j h_j x_{t-j}$). Beware: after causal reindexing, windowed-
  sinc kernels have tiny leading taps, which makes the naive one-step
  x-space proposal much worse than the y-space one specified here. x-space
  only becomes attractive with block/lookahead proposals.
- dithered runs are *easier* (the dither smooths the likelihood); debug
  there first.

### 5.4 Sanity invariants to assert in code

- $\hat p_t \le 1$ always; running $\hat R$ finite.
- Undithered, `none` kernel, any σ: the SMC reduces to i.i.d. draws and
  $\hat p_t$ must equal $\Phi$-difference of the marginal for every
  particle identically (zero variance across particles).
- Generated $z$ statistics: sample std of $z \approx \sqrt{\sigma^2\lVert h\rVert_2^2 + \sigma_q^2}$
  for $\sigma_y \gtrsim 1$.

---

## 6. Test plan

Ordered from cheap-and-exact to expensive-and-comparative.

1. **i.i.d. exactness (primary).** `none` kernel. True
   $R = H_\Delta(\sigma)$ in closed form. Require agreement within 3 SE at
   σ ∈ {0.5, 1, 5}, and that the reimplemented $R_{\text{approx}}$
   reproduces its known overestimates (§7 table).
2. **Marginal check.** $h_0$ (no conditioning) equals the marginal quantized
   entropy for any kernel: compare a short-window run against the
   1-D formula with $s = \sqrt{r_y(0)}$ (undithered).
3. **Brute force, small windows.** For MA(1)-like kernels
   ($h = [1, a]$, σ ≤ 2), compute $h_k$ for $k \le 3$ exactly:
   enumerate integer tuples within ±8σ_y and evaluate box probabilities
   with `scipy.stats.multivariate_normal.cdf` (Genz), then
   $h_k = H_{k+1} - H_k$. The SMC at the same $k$ (fresh independent
   windows, fixed $k$) must match within error bars.
4. **Monotonicity.** $\hat h_k$ non-increasing in $k$ (within noise);
   estimate decreasing and plateauing in particle count N; sandwich gap
   (if implemented) shrinking in $k$.
5. **Dead zone.** `none`, σ = 0.1: $\hat R \approx 0$ (truth ~1e-5), vs
   $R_{\text{approx}} = 0.576$. With dither, σ → 0: truth → 0 as well
   (uniform ±½ dither alone never flips the integer), vs
   $R_{\text{approx}} \approx 0.56$.
6. **Achievability cross-check.** For the default bandpass, σ = 5:
   $\hat R$ must be ≤ the achievable rates in §7 (LPC residual entropy
   ≈ 1.947) plus noise, and plausibly close to $R_{\text{approx}} = 1.940$.
   This is the headline number this whole exercise exists to pin down.
7. **Sweep deliverable.** σ ∈ {0.5, 1, 2, 5, 10, 20, 50, 100} × {dither
   on, off} for the default bandpass: plot $\hat R$ with error bars against
   $R_{\text{approx}}$, plus the i.i.d. sweep where truth is closed-form.
   Deliver the numbers as CSV alongside the plot.

Suggested starting parameters: N = 4096 particles, T = 4000 steps,
burn-in B = 1000, J = 16 replicates, $k_{\max} = 512$ for the 101-tap
kernel. Rough cost: O(N·k_max) per step ⇒ ~10⁹–10¹⁰ flops per replicate —
seconds to a minute in vectorized numpy (vectorize across particles; the
prediction is a matrix–vector product against the shared coefficient
vector).

## 7. Reference values (default bandpass = 300–6000 Hz @ 30 kHz, 101 taps)

Closed-form / app-measured values for validating reimplementations. "Exact"
rows are analytic; others were measured with the app's codecs on a
100k-sample block (seed-dependent in the third decimal).

| Quantity | Value | Status |
|---|---|---|
| $H_\Delta(s)$ at s = 0.1 / 0.3 / 1 / 5 / 50 | 0.00001 / 0.55042 / 2.10483 / 4.37142 / 7.69098 | exact |
| True R, `none`, σ = 5 / 0.5 / 0.1 | 4.37142 / 1.24174 / ≈1e-5 | exact ($=H_\Delta(\sigma)$) |
| $R_{\text{approx}}$, `none`, σ = 0.1 / 1 / 5 / 100 | 0.57611 / 2.15829 / 4.37382 / 8.69096 | exact given §1.3 |
| $R_{\text{approx}}$, bandpass, σ = 0.5 / 5 / 20 / 100 | 0.8620 / 1.9397 / 2.7282 / 3.7166 | exact given §1.3 |
| $R_{\text{approx}}$, bandpass + dither, σ = 5 | 2.2111 | exact given §1.3 |
| $\lVert h \rVert_2$, default bandpass | 0.60216 | exact given §1.2 |
| Order-0 entropy of LPC(32) / LPC(128) residual, bandpass σ = 5 | 1.965 / 1.947 | measured (achievable ⇒ upper bounds on R) |
| LPC(32)+ANS achieved, bandpass σ = 5 | 2.002–2.011 | measured |
| Pure high-res rate (no floor), bandpass σ = 5 | −2.4604 | exact; demonstrates why the floor exists |

## 8. References

- A. Genz, "Numerical computation of multivariate normal probabilities,"
  *J. Comput. Graph. Statist.*, 1992 — the separation-of-variables
  construction that §4.2 runs sequentially.
- A. Genz, F. Bretz, *Computation of Multivariate Normal and t
  Probabilities*, Springer, 2009.
- Z. I. Botev, "The normal law under linear restrictions: simulation and
  estimation via minimax tilting," *JRSS-B*, 2017 — variance reduction if
  §5.3 becomes limiting.
- T. M. Cover, J. A. Thomas, *Elements of Information Theory*, 2nd ed.:
  §4.5 (HMM entropy-rate sandwich), §16.8 / AEP (Shannon–McMillan–Breiman).
- Companion theory note in the sibling repository
  `entropy-quantized-linear-gaussian` (`entropy_quantized_linear_gaussian.md`):
  the block-entropy version of the same box-probability machinery, including
  the inner-bias caveat this spec's §4.3 addresses.
