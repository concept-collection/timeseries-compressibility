"""How long must the conditioning window M be?

The latent y = h*x is exactly (L-1)-dependent, so a naive reading says the
past only needs to be ~L samples. That is wrong for prediction: y is an
MA(L-1) process, and the optimal *predictor* of an MA process from its own
past is AR(infinity) — the whitening filter is 1/H(z), whose impulse response
decays at a rate set by how close H's zeros sit to the unit circle. For a
narrowband filter those zeros are very close, so the memory that matters runs
many multiples of L.

What keeps it finite is the rounding: z = round(y) has spectrum
S_z = sigma^2 |H|^2 + 1/12, which never reaches zero, so its whitening filter
always converges. Deep nulls (large sigma, narrow band, many taps) mean slow
decay; the 1/12 floor sets where the decay finally bites.

This computes, from that spectral model, the prediction order (= the past
length that matters) needed to get within 0.01 bits/sample of the infinite-
past limit — the M the estimator needs.
"""
import numpy as np
from scipy.special import ndtr

RATE = 30000.0
LOW, HIGH = 300.0, 2000.0


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


def bandpass(taps):
    return windowed_sinc_lowpass(HIGH / RATE, taps) - windowed_sinc_lowpass(LOW / RATE, taps)


def levinson_all(r, P):
    """Prediction-error variance for every order 0..P, one O(P^2) pass.
    Also returns the order-P coefficients (the whitening filter's AR part)."""
    a = np.zeros(P)
    err = np.empty(P + 1)
    err[0] = r[0]
    e = r[0]
    for i in range(P):
        acc = r[i + 1] - (a[:i] @ r[i:0:-1] if i else 0.0)
        k = acc / e
        if i:
            a[:i] = a[:i] - k * a[i - 1::-1]
        a[i] = k
        e *= 1 - k * k
        err[i + 1] = e
        if e <= 0:
            err[i + 1:] = e
            break
    return err, a


def entropy_at(var_pred, nfrac=1001, kmax=60):
    """Ideal conditional-coding rate (bits) given prediction-error variance."""
    s = np.sqrt(max(var_pred - 1.0 / 12.0, 1e-9))
    d = ((np.arange(nfrac) + 0.5) / nfrac - 0.5)[:, None]
    k = np.arange(-kmax, kmax + 1)[None, :]
    p = ndtr((k + 0.5 - d) / s) - ndtr((k - 0.5 - d) / s)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.where(p > 0, -p * np.log2(p), 0.0).sum(axis=1)
    return float(h.mean())


def report(sigma, taps, P=6000, tol_bits=0.01):
    h = bandpass(taps)
    L = len(h)
    # r_z[k] = sigma^2 (h corr h)[k] + (1/12) delta[k]; zero past lag L-1.
    rr = np.correlate(h, h, 'full')[L - 1:]
    r = np.zeros(P + 2)
    r[:L] = sigma**2 * rr
    r[0] += 1.0 / 12.0

    err, a = levinson_all(r, P)
    nfft = 1 << 16
    S = sigma**2 * np.abs(np.fft.fft(h, nfft))**2 + 1.0 / 12.0
    P_inf = float(np.exp(np.mean(np.log(S))))     # Kolmogorov limit

    H_inf = entropy_at(P_inf)
    # The rate is monotone in the prediction-error variance, and err[] is
    # monotone in the order — so bisect for the variance that costs tol bits,
    # then read off the first order that reaches it.
    vlo, vhi = P_inf, P_inf * 16
    for _ in range(60):
        vmid = 0.5 * (vlo + vhi)
        if entropy_at(vmid) - H_inf < tol_bits:
            vlo = vmid
        else:
            vhi = vmid
    reached = np.nonzero(err[1:] <= vlo)[0]
    need = int(reached[0]) + 1 if reached.size else None

    # where the whitening filter's tail falls below 1e-3 of its peak
    tail = np.abs(a) / np.abs(a).max()
    decay = int(np.max(np.nonzero(tail > 1e-3)[0])) + 1 if (tail > 1e-3).any() else P

    cli_default = max(512, 4 * L)
    flag = 'OK' if cli_default >= (need or P) else '** TOO SHORT **'
    print(f'  sigma={sigma:<4g} taps={taps:<4d} L={L:<4d}  '
          f'H_inf={H_inf:.3f} bits ({16/H_inf:5.1f}x)   '
          f'need M>~{need if need else ">"+str(P):<5} '
          f'(={(need or P)/L:4.1f} L)   AR tail {decay:<5d}  '
          f'CLI default M={cli_default} -> {flag}')


if __name__ == '__main__':
    print('bandpass 300-2000 Hz @ 30 kHz; "need M" = order within 0.01 bits '
          'of the infinite-past limit\n')
    for sigma in (5, 20):
        for taps in (15, 31, 101, 301):
            report(sigma, taps)
        print()
