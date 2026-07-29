#!/usr/bin/env python3
"""Monte Carlo estimate of the true entropy rate of the quantized filtered
Gaussian pipeline, alongside the closed-form approximation the UI plots.

Implements docs/mc-true-rate.md: sequential Monte Carlo (Genz separation of
variables with resampling) over the latent Gaussian path constrained to the
observed integer boxes. All systematic errors are upward, so the estimate
converges to the true R from above; double --particles and compare to check
convergence.

The UI shows the exact command for the current parameter set. Requires
numpy and scipy.

Examples:
  python scripts/true_rate.py --sigma 5 --filter bandpass --low-hz 300 \
      --high-hz 6000 --taps 101 --sample-rate 30000
  python scripts/true_rate.py --sigma 0.5 --filter none
  python scripts/true_rate.py --selftest
"""

import argparse
import math
import sys

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri

# ---------------------------------------------------------------- kernels

def windowed_sinc_lowpass(fc: float, taps: int) -> np.ndarray:
    n = taps if taps % 2 == 1 else taps + 1
    i = np.arange(n)
    t = i - (n - 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(t == 0, 2 * fc, np.sin(2 * np.pi * fc * t) / (np.pi * t))
    w = 0.54 - 0.46 * np.cos(2 * np.pi * i / (n - 1))
    h = sinc * w
    return h / h.sum()


def design_kernel(args) -> np.ndarray:
    f = args.filter
    if f == "none":
        return np.array([1.0])
    if f == "first-difference":
        return np.array([1.0, -1.0])
    if f == "moving-average":
        return np.full(args.width, 1.0 / args.width)
    if f == "lowpass":
        return windowed_sinc_lowpass(args.cutoff_hz / args.sample_rate, args.taps)
    if f == "bandpass":
        lo = windowed_sinc_lowpass(args.low_hz / args.sample_rate, args.taps)
        hi = windowed_sinc_lowpass(args.high_hz / args.sample_rate, args.taps)
        return hi - lo
    raise ValueError(f)

# ------------------------------------------------- closed-form quantities

def h_delta(s: float) -> float:
    """Exact entropy (bits) of round(N(0, s^2)) on the unit lattice."""
    if s <= 0:
        return 0.0
    zmax = int(np.ceil(8 * s + 4))
    z = np.arange(-zmax, zmax + 1)
    p = ndtr((z + 0.5) / s) - ndtr((z - 0.5) / s)
    p = p[p > 1e-300]
    return float(-(p @ np.log2(p)))


def magnitude_sq(h: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.arange(len(h))
    e = np.exp(-2j * np.pi * np.outer(f, n))
    H = e @ h
    return np.abs(H) ** 2


def r_approx(h: np.ndarray, sigma: float, dither: bool, points: int = 8192) -> float:
    """The UI's formula: noise-floored Szego prediction error through H_delta."""
    floor = 1 / 6 if dither else 1 / 12
    f = (np.arange(points) + 0.5) * 0.5 / points
    s_z = sigma**2 * magnitude_sq(h, f) + floor
    integral = float(np.log2(s_z).mean() * 0.5)
    return h_delta(2.0**integral)

# ----------------------------------------------------------- MC estimator

def autocovariance(h: np.ndarray, sigma: float, kmax: int) -> np.ndarray:
    full = sigma**2 * np.correlate(h, h, "full")
    r = np.zeros(kmax + 1)
    take = min(kmax + 1, len(h))
    r[:take] = full[len(h) - 1 : len(h) - 1 + take]
    return r


def levinson_all(r: np.ndarray, kmax: int):
    """Prediction coefficients for every order 0..kmax and innovation variances.

    A[p] holds a_1..a_p with yhat_t = sum_j a_j y_{t-j}; v[p] is the order-p
    prediction error variance.
    """
    A = [np.zeros(0)]
    v = np.zeros(kmax + 1)
    v[0] = r[0]
    a = np.zeros(kmax)
    for p in range(1, kmax + 1):
        acc = r[p] - (a[: p - 1] @ r[p - 1 : 0 : -1] if p > 1 else 0.0)
        k = acc / v[p - 1]
        if p > 1:
            a[: p - 1] = a[: p - 1] - k * a[p - 2 :: -1]
        a[p - 1] = k
        v[p] = max(v[p - 1] * (1 - k * k), 1e-30)
        A.append(a[:p].copy())
    return A, v


def log_phi_diff(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """log(Phi(beta) - Phi(alpha)) elementwise, safe in both tails."""
    out = np.empty_like(alpha)
    hi_tail = alpha >= 0  # reflect to the lower tail
    a = np.where(hi_tail, -beta, alpha)
    b = np.where(hi_tail, -alpha, beta)
    straddle = b >= 0  # a < 0 <= b: safe in linear space
    with np.errstate(divide="ignore"):
        out[straddle] = np.log(ndtr(b[straddle]) - ndtr(a[straddle]))
    lo = ~straddle  # both below 0: log-space difference
    la, lb = log_ndtr(a[lo]), log_ndtr(b[lo])
    out[lo] = lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-12)))
    return out


def sample_truncated(mu, s, lo, hi, rng):
    """Sample N(mu, s^2) truncated to [lo, hi], vectorized, tail-safe by
    reflection. Callers guarantee the interval has nonzero mass."""
    alpha = (lo - mu) / s
    beta = (hi - mu) / s
    flip = alpha > 0
    a = np.where(flip, -beta, alpha)
    b = np.where(flip, -alpha, beta)
    pa, pb = ndtr(a), ndtr(b)
    u = pa + rng.random(len(mu)) * (pb - pa)
    x = ndtri(np.clip(u, 1e-320, 1 - 1e-16))
    x = np.where(flip, -x, x)
    return mu + s * np.clip(x, alpha, beta)


def generate_z(h, sigma, dither, T, rng):
    L = len(h)
    x = rng.standard_normal(T + L - 1) * sigma
    y = np.convolve(x, h, "valid")
    if dither:
        y = y + rng.uniform(-0.5, 0.5, T)
    return np.round(y).astype(np.int64)


def smc_replicate(h, sigma, dither, T, burn, N, kmax, rng):
    """One replicate: -(1/(T-burn)) sum log2 phat(z_t | z_<t) after burn-in."""
    A, v = levinson_all(autocovariance(h, sigma, kmax), kmax)
    s_by_order = np.sqrt(v)
    z = generate_z(h, sigma, dither, T, rng)

    W = np.zeros((N, kmax), dtype=np.float32)  # each particle's last kmax values
    log_p = np.zeros(T)
    bad_steps = 0
    for t in range(T):
        p = min(t, kmax)
        a = A[p]
        s = s_by_order[p]
        mu = (W[:, kmax - p :] @ a[::-1].astype(np.float32)).astype(np.float64) if p else np.zeros(N)
        d = rng.uniform(-0.5, 0.5, N) if dither else 0.0
        lo = z[t] - 0.5 - d
        hi = z[t] + 0.5 - d
        logw = log_phi_diff((lo - mu) / s, (hi - mu) / s)
        lse = logsumexp(logw)
        if not np.isfinite(lse):
            raise RuntimeError(
                f"particle collapse at step {t}: no particle is consistent with "
                f"the observation — rerun with more --particles"
            )
        log_p[t] = lse - math.log(N)
        # A per-step surprisal beyond ~40 bits means the particle cloud has
        # drifted away from every path consistent with the data — the
        # genealogical-collapse failure mode of docs/mc-true-rate.md §5.3,
        # not a property of the data. Fail loudly rather than average it in.
        if -log_p[t] / math.log(2) > 40:
            bad_steps += 1
            if bad_steps > 25:
                raise RuntimeError(
                    "the sequential filter degenerated (near-deterministic dynamics; "
                    "see docs/mc-true-rate.md §5.3) — this kernel needs the "
                    "lookahead/twisted-proposal extension. Reduce --taps to study "
                    "the trend with a shallower stopband."
                )
        # Systematic resampling, then extend the chosen ancestors.
        probs = np.exp(logw - logw.max())
        cdf = np.cumsum(probs)
        cdf /= cdf[-1]
        ancestors = np.searchsorted(cdf, (rng.random() + np.arange(N)) / N)
        mu_a = mu[ancestors]
        lo_a = lo[ancestors] if dither else np.full(N, lo)
        hi_a = hi[ancestors] if dither else np.full(N, hi)
        y_new = sample_truncated(mu_a, s, lo_a, hi_a, rng)
        W = W[ancestors]
        W[:, :-1] = W[:, 1:]
        W[:, -1] = y_new.astype(np.float32)
    return float(-log_p[burn:].mean() / math.log(2))


def estimate_true_rate(h, sigma, dither, args, seed):
    rates = []
    for j in range(args.replicates):
        rng = np.random.default_rng(seed + j)
        r = smc_replicate(h, sigma, dither, args.steps, args.burn, args.particles, args.kmax, rng)
        rates.append(r)
        print(f"  replicate {j + 1}/{args.replicates}: {r:.4f}", flush=True)
    rates = np.array(rates)
    se = rates.std(ddof=1) / math.sqrt(len(rates)) if len(rates) > 1 else float("nan")
    return rates.mean(), se

# ------------------------------------------------------------- self-test

def selftest() -> int:
    failures = 0

    def check(name, got, want, tol):
        nonlocal failures
        ok = abs(got - want) <= tol
        failures += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'} {name}: got {got:.5f}, want {want:.5f} (tol {tol})")

    print("H_delta against reference values:")
    for s, want in [(0.1, 0.00001), (0.3, 0.55042), (1, 2.10483), (5, 4.37142), (50, 7.69098)]:
        check(f"H_delta({s})", h_delta(s), want, 2e-4)

    print("Formula against reference values (iid):")
    none = np.array([1.0])
    for s, want in [(0.1, 0.57611), (1, 2.15829), (5, 4.37382), (100, 8.69096)]:
        check(f"R_approx none sigma={s}", r_approx(none, s, False), want, 1e-3)

    print("Formula against reference values (default bandpass, end-to-end kernel check):")
    lo = windowed_sinc_lowpass(300 / 30000, 101)
    hi = windowed_sinc_lowpass(6000 / 30000, 101)
    bp = hi - lo
    check("||h||_2", float(np.sqrt((bp**2).sum())), 0.60216, 1e-4)
    for s, want in [(0.5, 0.8620), (5, 1.9397), (20, 2.7282), (100, 3.7166)]:
        check(f"R_approx bandpass sigma={s}", r_approx(bp, s, False), want, 1e-3)
    check("R_approx bandpass sigma=5 dither", r_approx(bp, 5, True), 2.2111, 1e-3)

    print("Quick MC on the exact iid case (truth = H_delta(sigma)):")
    rng = np.random.default_rng(1)
    est = smc_replicate(none, 5.0, False, T=1500, burn=300, N=512, kmax=8, rng=rng)
    check("MC none sigma=5", est, h_delta(5.0), 0.05)
    est = smc_replicate(none, 0.5, False, T=1500, burn=300, N=512, kmax=8, rng=rng)
    check("MC none sigma=0.5", est, h_delta(0.5), 0.05)

    print("FAILED" if failures else "all tests passed")
    return 1 if failures else 0

# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sigma", type=float, default=5.0, help="input std in quantization steps")
    ap.add_argument(
        "--filter",
        choices=["none", "moving-average", "lowpass", "bandpass", "first-difference"],
        default="none",
    )
    ap.add_argument("--low-hz", type=float, default=300, help="bandpass low edge")
    ap.add_argument("--high-hz", type=float, default=6000, help="bandpass high edge")
    ap.add_argument("--cutoff-hz", type=float, default=6000, help="lowpass cutoff")
    ap.add_argument("--taps", type=int, default=101, help="windowed-sinc kernel length")
    ap.add_argument("--width", type=int, default=8, help="moving-average width")
    ap.add_argument("--sample-rate", type=float, default=30000)
    ap.add_argument("--dither", action="store_true")
    ap.add_argument("--particles", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--burn", type=int, default=800)
    ap.add_argument("--replicates", type=int, default=8)
    ap.add_argument("--kmax", type=int, default=0, help="predictor memory; 0 = auto (4L, capped 512)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    h = design_kernel(args)
    L = len(h)
    if args.kmax == 0:
        args.kmax = min(max(4 * L, 64), 512)
    args.burn = max(args.burn, 2 * args.kmax)
    if args.steps <= args.burn + 500:
        args.steps = args.burn + 2000

    norm = float(np.sqrt((h**2).sum()))
    formula = r_approx(h, args.sigma, args.dither)
    _, v = levinson_all(autocovariance(h, args.sigma, args.kmax), args.kmax)
    s_inn = math.sqrt(v[args.kmax])
    print(f"kernel: {args.filter}, L={L}, ||h||2={norm:.5f}, sigma_y={args.sigma * norm:.3f}, "
          f"innovation std={s_inn:.4f}")
    if s_inn < 0.12:
        print("warning: innovation std << quantization step — near-deterministic dynamics. "
              "The plain sequential filter (docs/mc-true-rate.md §5.3) will likely degenerate "
              "here; reduce --taps for a shallower stopband, or implement the "
              "lookahead/twisted-proposal extension.")
    print(f"formula (as in the UI): R = {formula:.4f} bits/sample"
          + (f", ideal ratio {16 / formula:.3f}x vs int16" if formula > 0 else ""))
    if args.filter == "none":
        exact = h_delta(args.sigma) if not args.dither else None
        if exact is not None:
            print(f"exact truth (iid closed form): R = {exact:.4f} bits/sample"
                  + (f", ideal ratio {16 / exact:.3f}x vs int16" if exact > 0 else ""))

    print(f"MC (N={args.particles} particles, T={args.steps} steps, burn={args.burn}, "
          f"kmax={args.kmax}, {args.replicates} replicates):")
    try:
        mean, se = estimate_true_rate(h, args.sigma, args.dither, args, args.seed)
    except RuntimeError as e:
        print(f"aborted: {e}")
        return 2
    line = f"MC true rate: R = {mean:.4f} +/- {se:.4f} bits/sample"
    if mean > 0:
        # First-order error propagation: d(16/R) = 16 dR / R^2.
        line += f", ideal ratio {16 / mean:.3f}x +/- {16 * se / mean**2:.3f} vs int16"
    print(line)
    print(f"difference (formula - MC): {formula - mean:+.4f} bits/sample")
    print("note: finite window and inner MC both bias the estimate upward; "
          "double --particles and compare to confirm convergence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
