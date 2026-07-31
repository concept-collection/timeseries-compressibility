"""Reproduce the app's LPC(order) + ANS rate in Python.

Exact port of src/compress/lpc.ts semantics: Levinson fit on the block's
autocorrelation, coefficients quantized to 15-bit signed ints with a shared
right shift, integer prediction with floor division, residual wrapped to
int16. The coded size is the order-0 entropy of the residual (the ANS bars in
the app sit 1-2% above this, for the symbol table plus arithmetic loss).
"""
import numpy as np
from scipy.linalg import solve_toeplitz

SIGMA = 5.0
RATE = 30000.0
LOW, HIGH, TAPS = 300.0, 2000.0, 31
COEFF_PRECISION = 15


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


def make_signal(n, seed=1):
    h = windowed_sinc_lowpass(HIGH / RATE, TAPS) - windowed_sinc_lowpass(LOW / RATE, TAPS)
    rng = np.random.default_rng(seed)
    x = SIGMA * rng.standard_normal(n + len(h) - 1)
    y = np.convolve(x, h, mode='valid')
    return np.floor(y + 0.5).astype(np.int64)


def fit_lpc_quantized(z, order):
    zf = z.astype(np.float64)
    r = np.array([zf @ zf if lag == 0 else zf[lag:] @ zf[:-lag]
                  for lag in range(order + 1)])
    a = solve_toeplitz(r[:order], r[1:order + 1])
    max_abs = np.abs(a).max()
    shift = COEFF_PRECISION - 1 - int(np.floor(np.log2(max_abs))) - 1
    shift = max(0, min(15, shift))
    limit = 2 ** (COEFF_PRECISION - 1)
    q = np.clip(np.round(a * 2.0**shift), -limit, limit - 1).astype(np.int64)
    return q, shift


def lpc_residual(z, q, shift):
    order = len(q)
    # pred[n] = floor( sum_k q[k] z[n-1-k] / 2^shift )  for n >= order
    acc = np.convolve(z, q, mode='full')[:len(z)]     # acc[n] = sum_k q[k] z[n-k]
    pred = np.empty_like(z)
    pred[:] = 0
    pred[1:] = acc[:-1] // (1 << shift)               # floor division, exact int64
    e = z.copy()
    e[order:] = z[order:] - pred[order:]
    e = ((e + (1 << 15)) & 0xFFFF) - (1 << 15)        # int16 wraparound
    return e


def entropy_bits(v):
    counts = np.unique(v, return_counts=True)[1]
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    n = 1 << 22
    z = make_signal(n)
    print(f'{n} samples   var(z) = {z.var():.4f}   order-0 H(z) = {entropy_bits(z):.4f} bits '
          f'({16/entropy_bits(z):.1f}x)')
    for order in (8, 16, 32, 64, 128):
        q, shift = fit_lpc_quantized(z, order)
        e = lpc_residual(z, q, shift)
        H = entropy_bits(e[order:])
        coeff_bits = (2 + 2 * order) * 8 / n
        print(f'  LPC({order:3d}): shift={shift:2d}  residual H0 = {H:.4f} bits '
              f'-> ratio {16/H:.2f}x  (with ~1.5% ANS overhead: {16/(H*1.015):.2f}x)')


if __name__ == '__main__':
    main()
