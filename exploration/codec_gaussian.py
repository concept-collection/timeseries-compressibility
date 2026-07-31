"""A codec that beats LPC+ANS on the quantized filtered-Gaussian process.

The insight: with prediction-error std ~0.28 quantization steps, the
conditional distribution of z_t is a Gaussian bump over 1-3 integer bins whose
placement depends on the *fractional part* of the real-valued prediction.
Integer-residual LPC + a memoryless entropy coder pools all fractional parts
into one histogram and pays the mixture entropy (~1.10 bits). Coding z_t
against a discretized Gaussian centred at the real-valued prediction pays the
conditional entropy (~0.72 bits) instead.

Codec = order-p linear predictor (coefficients fitted on the block, sent as
float32 in the header) + adaptive binary arithmetic coding of z_t under
N(mu_t, s^2) discretized to integer bins, where mu_t is the float prediction
from already-decoded samples and s is a single fitted scale sent in the
header. Encoder and decoder compute mu_t with the identical np.dot on
identical float64 data, so the frequency tables agree bit for bit.

Everything the decoder needs is charged: order, coefficients, s, marginal
std for the warm-up samples, and the sample count.
"""
import math
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
from scipy.special import ndtr

SIGMA = 5.0
RATE = 30000.0
LOW, HIGH, TAPS = 300.0, 2000.0, 31

TOTAL_BITS = 16
TOTAL = 1 << TOTAL_BITS
KWIN = 32              # symbols are residuals in [-KWIN, KWIN-1] around round(mu)
SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------- the process
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


# ------------------------------------------------- arithmetic coder (WNC-style)
class BitWriter:
    def __init__(self):
        self.bytes = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, bit):
        self.acc = (self.acc << 1) | bit
        self.nbits += 1
        if self.nbits == 8:
            self.bytes.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def flush(self):
        while self.nbits:
            self.write(0)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self):
        byte = self.data[self.pos >> 3] if (self.pos >> 3) < len(self.data) else 0
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


class ArithEncoder:
    FULL = (1 << 32) - 1
    HALF = 1 << 31
    QUARTER = 1 << 30

    def __init__(self):
        self.low = 0
        self.high = self.FULL
        self.pending = 0
        self.out = BitWriter()

    def _emit(self, bit):
        self.out.write(bit)
        while self.pending:
            self.out.write(1 - bit)
            self.pending -= 1

    def encode(self, cum, freq, tot):
        span = self.high - self.low + 1
        self.high = self.low + span * (cum + freq) // tot - 1
        self.low = self.low + span * cum // tot
        while True:
            if self.high < self.HALF:
                self._emit(0)
            elif self.low >= self.HALF:
                self._emit(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.HALF + self.QUARTER:
                self.pending += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < self.QUARTER else 1)
        self.out.flush()
        return bytes(self.out.bytes)


class ArithDecoder:
    FULL = (1 << 32) - 1
    HALF = 1 << 31
    QUARTER = 1 << 30

    def __init__(self, data):
        self.low = 0
        self.high = self.FULL
        self.inp = BitReader(data)
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | self.inp.read()

    def target(self, tot):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * tot - 1) // span

    def consume(self, cum, freq, tot):
        span = self.high - self.low + 1
        self.high = self.low + span * (cum + freq) // tot - 1
        self.low = self.low + span * cum // tot
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.low -= self.HALF
                self.high -= self.HALF
                self.value -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.HALF + self.QUARTER:
                self.low -= self.QUARTER
                self.high -= self.QUARTER
                self.value -= self.QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.inp.read()


# --------------------------------------------- discretized-Gaussian frequencies
def freq_table(d, s):
    """Integer frequencies (sum TOTAL) for residual symbols -KWIN..KWIN-1:
    bin k has probability P(round(N(d, s^2)) = k), tails folded into the edge
    bins, every bin floored at 1. Deterministic in (d, s)."""
    inv = 1.0 / (s * SQRT2)
    m = min(KWIN - 1, int(8.0 * s) + 2)
    freqs = [1] * (2 * KWIN)
    spare = TOTAL - 2 * KWIN
    lo_cdf = 0.0
    probs = []
    for k in range(-m, m + 1):
        hi_cdf = 1.0 if k == m else 0.5 * math.erfc(-(k + 0.5 - d) * inv)
        probs.append(hi_cdf - lo_cdf)
        lo_cdf = hi_cdf
    scaled = [int(p * spare) for p in probs]
    deficit = spare - sum(scaled)
    scaled[max(range(len(scaled)), key=scaled.__getitem__)] += deficit
    for i, k in enumerate(range(-m, m + 1)):
        freqs[k + KWIN] += scaled[i]
    return freqs


def cumulative(freqs):
    cum = [0] * (len(freqs) + 1)
    for i, f in enumerate(freqs):
        cum[i + 1] = cum[i] + f
    return cum


# ------------------------------------------------------------------- the codec
def fit_model(z, order):
    zf = z.astype(np.float64)
    r = np.array([zf @ zf if lag == 0 else zf[lag:] @ zf[:-lag]
                  for lag in range(order + 1)]) / len(zf)
    a = solve_toeplitz(r[:order], r[1:order + 1]).astype(np.float32)
    # residual scale: fit s by minimizing the ideal code length on the block
    pred = lfilter(np.concatenate(([0.0], a.astype(np.float64))), [1.0], zf)
    e = zf[order:] - pred[order:]
    s0 = math.sqrt(max(e.var() - 1.0 / 12.0, 1e-6))
    best = (None, np.inf)
    zt, mu = zf[order:], pred[order:]
    for s in s0 * np.linspace(0.85, 1.15, 13):
        p = ndtr((zt + 0.5 - mu) / s) - ndtr((zt - 0.5 - mu) / s)
        bits = float(-np.log2(np.maximum(p, 1e-12)).mean())
        if bits < best[1]:
            best = (s, bits)
    s = np.float32(best[0])
    std_z = np.float32(max(zf.std(), 1e-3))
    return a, s, std_z, best[1]


HEADER_BYTES = 4 + 2 + 4 + 4  # n, order, s, std_z  (+ coefficients, counted below)


def encode(z, order):
    a, s, std_z, ideal_bits = fit_model(z, order)
    arev = a[::-1].astype(np.float64)
    zf = z.astype(np.float64)
    enc = ArithEncoder()

    warm = freq_table(0.0, float(std_z))
    warm_cum = cumulative(warm)
    s_f = float(s)
    for t in range(len(z)):
        if t < order:
            table, cum = warm, warm_cum
            c = 0
        else:
            mu = float(np.dot(arev, zf[t - order:t]))
            c = math.floor(mu + 0.5)
            table = freq_table(mu - c, s_f)
            cum = cumulative(table)
        sym = int(z[t]) - c + KWIN
        if not 0 <= sym < 2 * KWIN:
            raise ValueError(f'residual out of range at {t}')  # production: escape code
        enc.encode(cum[sym], table[sym], TOTAL)
    payload = enc.finish()
    header = HEADER_BYTES + 4 * order
    return payload, header, a, s, std_z, ideal_bits


def decode(payload, n, order, a, s, std_z):
    arev = a[::-1].astype(np.float64)
    zf = np.zeros(n, dtype=np.float64)
    z = np.zeros(n, dtype=np.int64)
    dec = ArithDecoder(payload)
    warm = freq_table(0.0, float(std_z))
    warm_cum = cumulative(warm)
    s_f = float(s)
    for t in range(n):
        if t < order:
            table, cum = warm, warm_cum
            c = 0
        else:
            mu = float(np.dot(arev, zf[t - order:t]))
            c = math.floor(mu + 0.5)
            table = freq_table(mu - c, s_f)
            cum = cumulative(table)
        tgt = dec.target(TOTAL)
        # find symbol: cum[sym] <= tgt < cum[sym+1]
        lo, hi = 0, 2 * KWIN
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if cum[mid] <= tgt:
                lo = mid
            else:
                hi = mid
        sym = lo
        dec.consume(cum[sym], table[sym], TOTAL)
        z[t] = sym - KWIN + c
        zf[t] = float(z[t])
    return z


def main():
    import time
    n = 1 << 20
    order = 64
    z = make_signal(n)

    t0 = time.time()
    payload, header, a, s, std_z, ideal_bits = encode(z, order)
    t1 = time.time()
    total_bytes = len(payload) + header
    bps = 8.0 * total_bytes / n
    print(f'encoded {n} samples in {t1 - t0:.1f}s')
    print(f'  ideal model cross-entropy (block fit): {ideal_bits:.4f} bits/sample')
    print(f'  payload {len(payload)} B + header {header} B = {total_bytes} B')
    print(f'  -> {bps:.4f} bits/sample   ratio vs int16: {16 / bps:.2f}x')

    t0 = time.time()
    zdec = decode(payload, n, order, a, s, std_z)
    t1 = time.time()
    ok = bool(np.array_equal(z, zdec))
    print(f'decoded in {t1 - t0:.1f}s   round-trip exact: {ok}')
    if not ok:
        bad = np.nonzero(z != zdec)[0][:5]
        print(f'  first mismatches at {bad}')


if __name__ == '__main__':
    main()
