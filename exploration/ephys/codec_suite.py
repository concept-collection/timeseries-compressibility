"""Lossless codecs measured in bits/sample on an int16 trace.

Every entry returns the full encoded size including whatever the decoder needs.
The generic byte compressors are run on the raw int16 buffer, on the
int16-wrapped first difference, and on the byte-split ("shuffled") buffer that
Blosc-style pipelines use.
"""
import bz2
import lzma
import math
import zlib

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
from scipy.special import ndtr

COEFF_PRECISION = 15


# ------------------------------------------------------------- preprocessing
def as_int16(z):
    return np.asarray(z, dtype=np.int16)


def delta(z):
    d = np.empty_like(z)
    d[0] = z[0]
    d[1:] = (z[1:].astype(np.int32) - z[:-1].astype(np.int32)).astype(np.int16)
    return d


def byteshuffle(z):
    b = z.tobytes()
    a = np.frombuffer(b, dtype=np.uint8).reshape(-1, 2)
    return np.concatenate([a[:, 0], a[:, 1]]).tobytes()


# ------------------------------------------------------- generic compressors
def _zlib(b):
    return len(zlib.compress(b, 9))


def _lzma(b):
    return len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))


def _bz2(b):
    return len(bz2.compress(b, 9))


def _zstd(b):
    import zstandard
    return len(zstandard.ZstdCompressor(level=19).compress(b))


def _brotli(b):
    import brotli
    return len(brotli.compress(b, quality=11))


def _lz4(b):
    import lz4.frame
    return len(lz4.frame.compress(b, compression_level=12))


GENERIC = {
    "zlib": _zlib, "zstd": _zstd, "lzma": _lzma,
    "bz2": _bz2, "brotli": _brotli, "lz4": _lz4,
}


# ------------------------------------------------------- lossless audio codecs
def flac_bytes(z, rate=30000):
    """FLAC via libsndfile: LPC + Rice-coded integer residual, the mature
    instance of the architecture LPC+ANS also belongs to. libsndfile does not
    expose the compression level; this is its default (level 5)."""
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(z, dtype=np.int16), int(rate),
             format="FLAC", subtype="PCM_16")
    return buf.getbuffer().nbytes


# --------------------------------------------------------------------- rates
def entropy0(v):
    counts = np.unique(np.asarray(v), return_counts=True)[1]
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ------------------------------------------------ integer LPC (the app's own)
def autocorr(z, order, ridge=1e-8):
    """Autocorrelation lags 0..order, with a small ridge on lag 0 so Levinson
    stays non-singular on degenerate (near-constant, near-empty) blocks."""
    zf = np.asarray(z, dtype=np.float64)
    r = np.array([zf @ zf if lag == 0 else zf[lag:] @ zf[:-lag]
                  for lag in range(order + 1)])
    r[0] = max(r[0], 1e-12 * zf.size) * (1.0 + ridge)
    return r


def fit_lpc_quantized(z, order):
    r = autocorr(z, order)
    a = solve_toeplitz(r[:order], r[1:order + 1])
    peak = float(np.abs(a).max())
    if peak <= 0:
        return np.zeros(order, dtype=np.int64), 0
    shift = COEFF_PRECISION - 1 - int(np.floor(np.log2(peak))) - 1
    shift = max(0, min(15, shift))
    limit = 2 ** (COEFF_PRECISION - 1)
    q = np.clip(np.round(a * 2.0 ** shift), -limit, limit - 1).astype(np.int64)
    return q, shift


def lpc_residual(z, q, shift):
    z = np.asarray(z, dtype=np.int64)
    order = len(q)
    acc = np.convolve(z, q, mode="full")[:len(z)]
    pred = np.zeros_like(z)
    pred[1:] = acc[:-1] >> shift                       # floor division
    e = z.copy()
    e[order:] = z[order:] - pred[order:]
    return (((e + (1 << 15)) & 0xFFFF) - (1 << 15)).astype(np.int16)


def lpc_ans_bytes(z, order):
    """Real encoded size of integer-LPC + rANS, side information included."""
    import simple_ans
    q, shift = fit_lpc_quantized(z, order)
    e = lpc_residual(z, q, shift)
    enc = simple_ans.ans_encode(e)
    payload = 4 * enc.words.size
    table = 4 * enc.symbol_counts.size + 2 * enc.symbol_values.size
    header = 2 * order + 4 + 8          # coefficients, shift, n
    return payload + table + header, entropy0(e)


# ------------------------------- conditional-Gaussian coding (achievable rate)
def conditional_gaussian_rate(z, order):
    """Cross-entropy of the real-valued-prediction conditional-Gaussian model,
    in bits/sample, plus header cost. This is what the arithmetic coder of
    exploration/codec_gaussian.py achieves to within ~0.1%."""
    zf = np.asarray(z, dtype=np.float64)
    n = zf.size
    r = autocorr(zf, order) / n
    a = solve_toeplitz(r[:order], r[1:order + 1])
    pred = lfilter(np.concatenate(([0.0], a)), [1.0], zf)
    zt, mu = zf[order:], pred[order:]
    s0 = math.sqrt(max((zt - mu).var() - 1.0 / 12.0, 1e-6))
    best = np.inf
    for s in s0 * np.linspace(0.7, 1.3, 25):
        p = ndtr((zt + 0.5 - mu) / s) - ndtr((zt - 0.5 - mu) / s)
        best = min(best, float(-np.log2(np.maximum(p, 1e-12)).mean()))
    header_bits = 8 * (4 * order + 14) / n
    return best + header_bits


# ------------------------------------------------------------------ the suite
def measure(z, lpc_order=32, generic=True):
    """bits/sample for every method, as an ordered dict."""
    z = as_int16(z)
    n = z.size
    out = {}
    out["raw int16"] = 16.0
    out["order-0 H(z)"] = entropy0(z)
    if generic:
        buffers = {"": z.tobytes(), "+delta": delta(z).tobytes(),
                   "+shuffle": byteshuffle(z)}
        for name, fn in GENERIC.items():
            for suffix, buf in buffers.items():
                out[f"{name}{suffix}"] = 8.0 * fn(buf) / n
    try:
        out["FLAC"] = 8.0 * flac_bytes(z) / n
    except Exception as exc:                                  # pragma: no cover
        out["FLAC"] = float("nan")
        print(f"    (FLAC unavailable: {exc})")
    nbytes, resid_h0 = lpc_ans_bytes(z, lpc_order)
    out[f"LPC({lpc_order})+ANS"] = 8.0 * nbytes / n
    out[f"LPC({lpc_order}) resid H0"] = resid_h0
    out[f"cond-Gauss({lpc_order})"] = conditional_gaussian_rate(z, lpc_order)
    return out
