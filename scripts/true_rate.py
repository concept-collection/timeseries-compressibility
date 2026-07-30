#!/usr/bin/env python3
"""Monte-Carlo ground truth for the theoretical rate R shown in the app.

Model (matching the app): x iid N(0, sigma^2) -> y = h * x -> optional
additive uniform dither on [-1/2, 1/2) -> z = round(.). The entropy rate
H = E[-log2 P(z_next | past)] is the true lossless limit in bits/sample.
(The app applies the kernel zero-phase; a time shift does not change the
law of the process, so causal convolution is used here.)

Method
  1. Draw a past: sample x (and dither) from the prior and push it through
     the pipeline to get z_1..z_M.
  2. The generating x is itself an exact draw from p(x | z_1..z_M), so a
     Gibbs chain started there is already in stationarity - no burn-in
     bias, only autocorrelation.
  3. Gibbs-sample x | z: this posterior is a box-truncated multivariate
     normal, and each conditional x_i | rest is N(0, sigma^2) truncated to
     an interval read off the <= L constraint boxes x_i appears in. With
     dither on, the dither values are extra latents with uniform
     conditionals. Coordinates a multiple of L apart share no constraint,
     so each of the L "colors" is updated as one vectorized block.
  4. Rao-Blackwellization: given a chain state, the next sample is
     z_next = round(c + h_0 * x_free (+ d)) with x_free ~ N(0, sigma^2)
     still unconstrained, so P(z_next = j | state) has a closed form.
     Averaging these pmfs over the chain gives P(z_next | past) exactly in
     the limit; its entropy is the conditional entropy for that past.
  5. Average over independent pasts; report mean +/- standard error.

The estimate targets H(z_{M+1} | z_1..z_M), which is an upper bound on the
rate and decreases toward it as --past grows beyond the memory of the
process. More --sweeps reduces the (downward) plug-in bias from noise in
the averaged pmf; narrowband filters and large sigma mix more slowly and
deserve more sweeps.

Requires numpy only.
"""

import argparse
import math

import numpy as np

SQRT2PI = math.sqrt(2 * math.pi)


# ---------------------------------------------------------------- kernels
# Ported from src/model/filters.ts; must stay in step with it.

def windowed_sinc_lowpass(fc, taps):
    n = taps | 1
    mid = (n - 1) / 2
    i = np.arange(n)
    t = i - mid
    sinc = np.where(t == 0, 2 * fc, np.sin(2 * np.pi * fc * t) / (np.pi * np.where(t == 0, 1, t)))
    w = 0.54 - 0.46 * np.cos(2 * np.pi * i / (n - 1))
    h = sinc * w
    return h / h.sum()


def design_kernel(args):
    if args.filter == 'none':
        return np.array([1.0])
    if args.filter == 'moving-average':
        return np.full(args.width, 1.0 / args.width)
    if args.filter == 'lowpass':
        return windowed_sinc_lowpass(args.high / args.rate, args.taps)
    if args.filter == 'bandpass':
        lo = windowed_sinc_lowpass(args.low / args.rate, args.taps)
        hi = windowed_sinc_lowpass(args.high / args.rate, args.taps)
        return hi - lo
    if args.filter == 'first-difference':
        return np.array([1.0, -1.0])
    raise ValueError(args.filter)


# ------------------------------------------------- the app's formula for R
# Ported from src/model/theory.ts. Phi via math.erf (machine precision).

TWO_PI_E = 2 * math.pi * math.e


def quantized_gaussian_entropy(s):
    if s <= 0.02:
        return 0.0
    # The discrete entropy approaches 1/2 log2(2 pi e s^2) from above like
    # log2(e)/(24 s^2) (the delta^2/24 Fisher-information correction); at
    # s >= 6 the corrected asymptote is within 2e-6 bits.
    if s >= 6:
        return 0.5 * math.log2(TWO_PI_E * s * s) + math.log2(math.e) / (24 * s * s)
    zmax = int(math.ceil(8 * s + 4))
    H = 0.0
    prev = 0.5 * (1 + math.erf((-zmax - 0.5) / (s * math.sqrt(2))))
    for z in range(-zmax, zmax + 1):
        cur = 0.5 * (1 + math.erf((z + 0.5) / (s * math.sqrt(2))))
        p = cur - prev
        prev = cur
        if p > 0:
            H -= p * math.log2(p)
    return H


def dithered_quantized_gaussian_entropy(s):
    """Exact entropy of round(N(0, s^2) + U[-1/2, 1/2)) — the marginal of a
    stored sample with dither on. The pmf has the closed form
    p_j = s * (G((j+1)/s) - 2 G(j/s) + G((j-1)/s)) with G(t) = t Phi(t) + phi(t)
    the antiderivative of Phi; machine-exact at every s via math.erf."""
    if s <= 0:
        return 0.0

    def G(t):
        return t * 0.5 * (1 + math.erf(t / math.sqrt(2))) + math.exp(-0.5 * t * t) / SQRT2PI

    jmax = int(math.ceil(8 * s + 2))
    H = 0.0
    for j in range(-jmax, jmax + 1):
        p = s * (G((j + 1) / s) - 2 * G(j / s) + G((j - 1) / s))
        if p > 0:
            H -= p * math.log2(p)
    return H


def formula_rates(kernel, sigma, dither, points=8192):
    """The app's R = min(Rspec, Rsamp). Rspec charges the rounding(+dither)
    noise at its full variance nu per Fourier mode; Rsamp is the exact
    marginal entropy of one stored sample, a subadditivity upper bound that
    takes over when the whole process is sub-threshold. Midpoint grid on
    [0, 1/2]; |H| is symmetric, so the grid mean equals the unit-circle
    integral. Returns (rspec, rsamp)."""
    nu = 1 / 6 if dither else 1 / 12
    f = (0.5 * (np.arange(points) + 0.5)) / points
    w = -2j * np.pi * np.outer(f, np.arange(len(kernel)))
    S = sigma * sigma * np.abs(np.exp(w) @ kernel) ** 2
    rspec = float(np.mean(0.5 * np.log2(TWO_PI_E * (S + nu))))
    v = sigma * sigma * float(np.sum(np.asarray(kernel) ** 2))
    rsamp = (dithered_quantized_gaussian_entropy(math.sqrt(v)) if dither
             else quantized_gaussian_entropy(math.sqrt(v)))
    return rspec, rsamp


# ------------------------------------------------ vectorized normal helpers

def norm_pdf(t):
    return np.exp(-0.5 * t * t) / SQRT2PI


def norm_cdf(t):
    """Abramowitz-Stegun 26.2.17; |error| < 7.5e-8, plenty for sampling and
    for pmf weights whose entropy is wanted to ~1e-4 bits."""
    t = np.asarray(t, dtype=float)
    z = np.abs(t)
    k = 1.0 / (1.0 + 0.2316419 * z)
    poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))))
    tail = norm_pdf(z) * poly
    return np.where(t >= 0, 1.0 - tail, tail)


def norm_ppf(p):
    """Acklam's rational approximation to the standard normal quantile."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p = np.asarray(p, dtype=float)
    x = np.empty_like(p)
    plow, phigh = 0.02425, 1 - 0.02425

    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)

    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        x[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        x[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        x[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return x


def trunc_std_normal(lo, hi, rng):
    """Standard normal truncated to [lo, hi], by inverse CDF. Mirrored into
    the lower tail so the CDF differences keep precision."""
    flip = (lo + hi) > 0
    a = np.where(flip, -hi, lo)
    b = np.where(flip, -lo, hi)
    Fa = norm_cdf(a)
    Fb = norm_cdf(b)
    u = Fa + (Fb - Fa) * rng.random(a.shape)
    x = norm_ppf(np.clip(u, 1e-300, 1 - 1e-16))
    x = np.where(flip, -x, x)
    return np.clip(x, lo, hi)


# --------------------------------------------------------- the RB next-pmf

def big_g(t):
    """G(t) = t Phi(t) + phi(t), the antiderivative of Phi."""
    return t * norm_cdf(t) + norm_pdf(t)


def next_pmf(c, s0, dither):
    """P(z_next = j | chain state): round(c + N(0, s0^2) (+ U(-1/2,1/2)))."""
    s = max(s0, 1e-12)
    half = 8 * s + (1.0 if dither else 0.0) + 1.0
    js = np.arange(math.floor(c - half), math.ceil(c + half) + 1)
    if dither:
        # Integrating the Gaussian bin probability over the dither gives a
        # second difference of G; as s -> 0 it degrades gracefully to the
        # uniform-overlap width.
        p = s * (big_g((js + 1 - c) / s) - 2 * big_g((js - c) / s) + big_g((js - 1 - c) / s))
    else:
        edges = norm_cdf((np.append(js, js[-1] + 1) - 0.5 - c) / s)
        p = np.diff(edges)
    return js, np.maximum(p, 0.0)


# ---------------------------------------------------------------- one past

def conditional_entropy_of_one_past(kernel, sigma, dither, M, sweeps, rng):
    h = np.asarray(kernel, dtype=float)
    L = len(h)
    hr = h[::-1]
    N = M + L - 1  # latents covering the windows of z_1..z_M

    # The past, with its true latents as the (stationary) chain start.
    x = sigma * rng.standard_normal(N)
    y = np.convolve(x, h, mode='valid')
    d = (rng.random(M) - 0.5) if dither else None
    z = np.floor(y + (d if dither else 0.0) + 0.5)

    # Boxes and y live in padded arrays so that every coordinate x_i sees
    # exactly L constraint rows (rows outside the data are unconstrained).
    P = L - 1
    ypad = np.zeros(M + 2 * P)
    lo = np.full(M + 2 * P, -np.inf)
    hi = np.full(M + 2 * P, np.inf)

    def set_boxes():
        dd = d if dither else 0.0
        lo[P:P + M] = z - 0.5 - dd
        hi[P:P + M] = z + 0.5 - dd

    set_boxes()

    # Color classes: coordinates L apart share no constraint row, so a class
    # updates as one vectorized block. Row i+j (padded) carries coefficient
    # h[j] for coordinate i.
    classes = [np.arange(c0, N, L) for c0 in range(L)]
    rowmats = [idx[:, None] + np.arange(L)[None, :] for idx in classes]
    nonzero = h != 0

    s0 = sigma * abs(h[0])
    pmf = {}

    for _ in range(sweeps):
        ypad[P:P + M] = np.convolve(x, h, mode='valid')  # kill fp drift
        if dither:
            ycur = ypad[P:P + M]
            dlo = np.maximum(-0.5, z - 0.5 - ycur)
            dhi = np.minimum(0.5, z + 0.5 - ycur)
            d = dlo + np.maximum(dhi - dlo, 0.0) * rng.random(M)
            set_boxes()
        for idx, rows in zip(classes, rowmats):
            r = ypad[rows] - np.outer(x[idx], h)
            with np.errstate(divide='ignore', invalid='ignore'):
                b1 = (lo[rows] - r) / h[None, :]
                b2 = (hi[rows] - r) / h[None, :]
            xlo = np.where(h[None, :] > 0, b1, b2)
            xhi = np.where(h[None, :] > 0, b2, b1)
            xlo[:, ~nonzero] = -np.inf
            xhi[:, ~nonzero] = np.inf
            xlo = xlo.max(axis=1)
            xhi = xhi.min(axis=1)
            xnew = trunc_std_normal(xlo / sigma, xhi / sigma, rng) * sigma
            delta = xnew - x[idx]
            x[idx] = xnew
            ypad[rows] += delta[:, None] * h[None, :]

        c = float(hr[:-1] @ x[M:]) if L > 1 else 0.0
        js, p = next_pmf(c, s0, dither)
        for j, pj in zip(js, p):
            if pj > 1e-15:
                pmf[int(j)] = pmf.get(int(j), 0.0) + pj

    total = sum(pmf.values())
    return -sum((p / total) * math.log2(p / total) for p in pmf.values() if p > 0)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description='Monte-Carlo estimate of the true entropy rate, for '
                    'checking the analytic rate R shown in the app.')
    ap.add_argument('--sigma', type=float, required=True, help='input std, in quantization steps')
    ap.add_argument('--filter', required=True,
                    choices=['none', 'moving-average', 'lowpass', 'bandpass', 'first-difference'])
    ap.add_argument('--low', type=float, help='bandpass low edge, Hz')
    ap.add_argument('--high', type=float, help='lowpass cutoff / bandpass high edge, Hz')
    ap.add_argument('--taps', type=int, default=101, help='windowed-sinc kernel length')
    ap.add_argument('--width', type=int, default=8, help='moving-average width')
    ap.add_argument('--rate', type=float, default=30000, help='sample rate, Hz')
    ap.add_argument('--dither', action='store_true')
    ap.add_argument('--past', type=int, help='conditioning window M (default max(512, 4·taps))')
    ap.add_argument('--pasts', type=int, default=24, help='independent pasts to average')
    ap.add_argument('--sweeps', type=int, default=600, help='Gibbs sweeps per past')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.filter == 'bandpass' and (args.low is None or args.high is None):
        ap.error('bandpass needs --low and --high')
    if args.filter == 'lowpass' and args.high is None:
        ap.error('lowpass needs --high')

    kernel = design_kernel(args)
    L = len(kernel)
    M = args.past if args.past is not None else max(512, 4 * L)

    R_spec, R_samp = formula_rates(kernel, args.sigma, args.dither)
    R = min(R_spec, R_samp)
    print(f'model: sigma={args.sigma} filter={args.filter} L={L} dither={args.dither}')
    ratio = f'  (ratio vs int16: {16 / R:.3f}x)' if R > 0 else ''
    print(f'formula R = min(spec {R_spec:.4f}, samp {R_samp:.4f}) = {R:.4f} bits/sample{ratio}')
    print(f'MC: {args.pasts} pasts x {args.sweeps} sweeps, conditioning on M={M} samples')

    rng = np.random.default_rng(args.seed)
    Hs = []
    for i in range(args.pasts):
        Hs.append(conditional_entropy_of_one_past(
            kernel, args.sigma, args.dither, M, args.sweeps, rng))
        mean = float(np.mean(Hs))
        se = float(np.std(Hs, ddof=1) / math.sqrt(len(Hs))) if len(Hs) > 1 else float('nan')
        print(f'  past {i + 1:3d}/{args.pasts}: H = {Hs[-1]:.4f}   running mean {mean:.4f} +/- {se:.4f}')

    mean = float(np.mean(Hs))
    se = float(np.std(Hs, ddof=1) / math.sqrt(len(Hs)))
    print(f'\nMC entropy rate: {mean:.4f} +/- {se:.4f} bits/sample'
          f'  (ratio vs int16: {16 / mean:.3f}x)')
    print(f'formula R:       {R:.4f} bits/sample   (formula - MC = {R - mean:+.4f}; '
          f'spec {R_spec:.4f}, samp {R_samp:.4f})')
    print('note: the MC value estimates H(z_next | M past samples), an upper bound '
          'on the rate that tightens as --past grows.')


if __name__ == '__main__':
    main()
