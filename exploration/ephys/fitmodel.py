"""Fit the paper's generative model  x ~ iid N(0, 1) -> h * x -> round  to a
real integer trace, and synthesize a surrogate from the fit.

Only the spectrum is identifiable (sigma and the kernel scale are the same
knob), so sigma is fixed at 1 and h carries everything.

Fit:
  1. Welch PSD of z, in the convention  mean_{f in [0,1)} S(f) = var(z).
  2. Dither model: S_z = S_y + 1/12, so S_y = max(S_z - 1/12, floor).
  3. h = the minimum-phase spectral factor of S_y (real-cepstrum method).

Prediction (no sampling required):
  s_*^2 = exp( mean_f ln(S_y + 1/12) ) - 1/12      [Szego on the observed
  spectrum], and the predicted entropy rate is G(s_*), the entropy of
  N(0, s_*^2) + U(-1/2, 1/2).
"""
import numpy as np
from scipy.signal import welch, fftconvolve

FLOOR_FRAC = 1e-4   # S_y is floored at this fraction of its own mean


def psd(z, nfft=4096):
    """Two-sided PSD on the rfft grid f = k/nfft, k = 0..nfft/2, normalized so
    that the mean over f in [0, 1) equals var(z)."""
    f, pxx = welch(np.asarray(z, dtype=np.float64), fs=1.0, nperseg=nfft,
                   noverlap=nfft // 2, window="hann", detrend="constant",
                   return_onesided=True, scaling="density")
    return f, pxx / 2.0


def _grid_mean(v):
    """Mean over f in [0, 1) of a quantity given on the rfft half-grid
    (trapezoid on [0, 1/2]; the spectrum is symmetric)."""
    return float((v.sum() - 0.5 * (v[0] + v[-1])) / (v.size - 1))


def minimum_phase(s_half, n_taps=None):
    """Minimum-phase impulse response whose |H(f)|^2 matches s_half on the
    rfft grid. Real-cepstrum construction."""
    s_full = np.concatenate([s_half, s_half[-2:0:-1]])
    n = s_full.size
    c = np.fft.ifft(0.5 * np.log(s_full)).real       # real cepstrum of |H|
    cm = np.zeros(n)
    cm[0] = c[0]
    cm[1:n // 2] = 2.0 * c[1:n // 2]
    cm[n // 2] = c[n // 2]
    h = np.fft.ifft(np.exp(np.fft.fft(cm))).real     # causal, min phase
    return h if n_taps is None else h[:n_taps]


class Fit:
    def __init__(self, z, nfft=4096, n_taps=None):
        z = np.asarray(z, dtype=np.float64)
        self.n = z.size
        self.nfft = nfft
        self.freq, self.s_z = psd(z, nfft)
        floor = FLOOR_FRAC * self.s_z.mean()
        self.s_y = np.maximum(self.s_z - 1.0 / 12.0, floor)
        self.kernel = minimum_phase(self.s_y, n_taps)
        # Szego on the observed spectrum, with roundoff as a 1/12 dither floor
        gm = np.exp(_grid_mean(np.log(self.s_y + 1.0 / 12.0)))
        self.s_star = float(np.sqrt(max(gm - 1.0 / 12.0, 0.0)))
        self.sigma_inf = float(np.sqrt(np.exp(_grid_mean(np.log(self.s_y)))))

    @property
    def predicted_rate(self):
        """G(s_*) in bits/sample — the analytic entropy-rate prediction."""
        from timeseries_entropy.theory import gauss_uniform_entropy
        return float(gauss_uniform_entropy(self.s_star))

    def kernel_error_db(self):
        """How well the (possibly truncated) kernel reproduces the fitted
        spectrum: RMS error in dB over the grid."""
        h = np.abs(np.fft.rfft(self.kernel, self.nfft)) ** 2
        return float(np.sqrt(np.mean((10 * np.log10(h / self.s_y)) ** 2)))

    def synthesize(self, n=None, seed=0):
        """A surrogate integer trace from the fitted model."""
        n = self.n if n is None else n
        rng = np.random.default_rng(seed)
        k = self.kernel
        x = rng.standard_normal(n + k.size - 1)
        y = fftconvolve(x, k, mode="valid")
        return np.round(y).astype(np.int16)
