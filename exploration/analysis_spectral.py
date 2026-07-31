"""Semi-analytic cross-check of the entropy rate for
x iid N(0, sigma^2) -> 31-tap bandpass 300-2000 Hz @ 30 kHz -> round.

Model the rounding as additive iid U(-1/2, 1/2) noise (variance 1/12) that is
independent of y. Then z = y + q is a stationary process with spectrum
S_z(f) = sigma^2 |H(f)|^2 + 1/12, and

  - one-step innovation variance of z:  P_z = exp( mean_f ln S_z(f) )   (Kolmogorov)
  - Var(y_next | past z)  =  s^2  =  P_z - 1/12   (q_next independent of past)
  - H(z_next | past z) ~= E_frac [ entropy of round-binned N(mu, s^2) ]
        averaged over the fractional part of mu (approximately uniform)
  - LPC + memoryless coder ~= entropy of the MIXTURE distribution of
        round(y_next) - floor(mu): the integer-residual histogram pools all
        fractional parts, so it cannot use frac(mu).

Also validated empirically on a long simulated realization with a
real-coefficient Wiener predictor.
"""
import numpy as np
from scipy.special import ndtr

SIGMA = 5.0
RATE = 30000.0
LOW, HIGH, TAPS = 300.0, 2000.0, 31


def windowed_sinc_lowpass(fc, taps):
    n = taps | 1
    mid = (n - 1) / 2
    i = np.arange(n)
    t = i - mid
    sinc = np.where(t == 0, 2 * fc,
                    np.sin(2 * np.pi * fc * t) / (np.pi * np.where(t == 0, 1, t)))
    w = 0.54 - 0.46 * np.cos(2 * np.pi * i / (n - 1))
    h = sinc * w
    return h / h.sum()


def windowed_sinc_bandpass(f_lo, f_hi, taps):
    return windowed_sinc_lowpass(f_hi, taps) - windowed_sinc_lowpass(f_lo, taps)


def binned_gaussian_entropy(mu_frac, s, kmax=60):
    """Entropy (bits) of round(N(mu_frac, s^2)) for scalar/array mu_frac."""
    mu_frac = np.atleast_1d(mu_frac)[:, None]
    k = np.arange(-kmax, kmax + 1)[None, :]
    p = ndtr((k + 0.5 - mu_frac) / s) - ndtr((k - 0.5 - mu_frac) / s)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    return terms.sum(axis=1)


def mixture_residual_entropy(s, kmax=60, nfrac=4001):
    """Entropy (bits) of round(y) - floor(mu), y ~ N(mu, s^2), frac(mu) uniform."""
    u = (np.arange(nfrac) + 0.5) / nfrac   # frac part of mu in (0,1)
    k = np.arange(-kmax, kmax + 1)[None, :]
    p = ndtr((k + 0.5 - u[:, None]) / s) - ndtr((k - 0.5 - u[:, None]) / s)
    pmix = p.mean(axis=0)
    pmix = pmix[pmix > 0]
    return float(-(pmix * np.log2(pmix)).sum())


def main():
    h = windowed_sinc_bandpass(LOW / RATE, HIGH / RATE, TAPS)
    var_y = SIGMA**2 * (h**2).sum()
    print(f'kernel taps={len(h)}  sum h^2 = {(h**2).sum():.6f}')
    print(f'Var(y) = {var_y:.4f}   std(y) = {np.sqrt(var_y):.4f} steps')

    nfft = 1 << 18
    Hf2 = np.abs(np.fft.fft(h, nfft))**2
    S_y = SIGMA**2 * Hf2
    S_z = S_y + 1.0 / 12.0
    P_z = np.exp(np.mean(np.log(S_z)))
    s2 = P_z - 1.0 / 12.0
    s = np.sqrt(s2)
    print(f'\nadditive-noise model:')
    print(f'  innovation var of z:  P_z = {P_z:.5f}')
    print(f'  Var(y_next | past z): s^2 = {s2:.5f}   s = {s:.4f}')

    # conditional entropy: average over fractional part of the predictive mean
    fracs = (np.arange(4001) + 0.5) / 4001 - 0.5
    Hc = binned_gaussian_entropy(fracs, s).mean()
    print(f'\n  H(z_next | past z)  ~= {Hc:.4f} bits/sample '
          f'(ratio {16/Hc:.1f}x)  <- what an ideal conditional coder gets')

    Hmix = mixture_residual_entropy(s)
    print(f'  order-0 entropy of integer LPC residual ~= {Hmix:.4f} bits/sample '
          f'(ratio {16/Hmix:.1f}x)  <- LPC + ANS ceiling')

    # what pooling the fractional part costs
    print(f'  gap (frac-part information thrown away) = {Hmix - Hc:.4f} bits')

    # ---- empirical check on a long realization -------------------------------
    rng = np.random.default_rng(1)
    N = 1 << 22
    x = SIGMA * rng.standard_normal(N + len(h) - 1)
    y = np.convolve(x, h, mode='valid')
    z = np.floor(y + 0.5)
    counts = np.unique(z, return_counts=True)[1]
    p = counts / counts.sum()
    H0 = -(p * np.log2(p)).sum()
    print(f'\nempirical ({N} samples): Var(z) = {z.var():.4f}   '
          f'order-0 H(z) = {H0:.4f} bits')

    # real-coefficient Wiener predictor of z_next from past z, various orders
    from scipy.linalg import solve_toeplitz
    from scipy.signal import lfilter
    zc = z - z.mean()
    maxlag = 256
    r = np.array([zc @ zc if lag == 0 else zc[lag:] @ zc[:-lag]
                  for lag in range(maxlag + 1)]) / len(zc)
    for order in (8, 16, 32, 64, 128, 256):
        a = solve_toeplitz(r[:order], r[1:order + 1])
        pred = lfilter(np.concatenate(([0.0], a)), [1.0], zc)  # strictly causal
        e = zc - pred
        ev = e[order:].var()
        s2e = ev - 1.0 / 12.0
        # ideal conditional-Gaussian coding rate at this order
        mu = pred[order:] + z.mean()
        zt = z[order:]
        se = np.sqrt(max(s2e, 1e-6))
        pz = (ndtr((zt + 0.5 - mu) / se) - ndtr((zt - 0.5 - mu) / se))
        pz = np.maximum(pz, 1e-30)
        rate = float(-np.log2(pz).mean())
        print(f'  order {order:4d}: pred-err var {ev:.5f}  '
              f'(s^2={s2e:.5f})  conditional-Gaussian rate {rate:.4f} bits')


if __name__ == '__main__':
    main()
