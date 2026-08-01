"""Decompose the measured gap between LPC+ANS and the entropy rate into named
terms, and check each against what the theory predicts.

    R_ANS  -  Hbar
      = [R_ANS - H0(resid)]        coder overhead: rANS vs a perfect memoryless
                                   coder on its own residual stream
      + [H0(resid) - R_condG]      residual-model loss: the pooled integer
                                   histogram vs the phase-conditioned law.
                                   Theory says this is L(s) = M(s) - G(s).
      + [R_condG - G(s_*)]         prediction suboptimality + parametric
                                   mismatch of the single-scale Gaussian
      + [G(s_*) - Hbar]            error of the analytic rate itself

`s_fit` is the residual scale the conditional-Gaussian model actually fits,
which is the honest argument to L(.) — `s_*` is its prediction from the
spectrum, and the two differing is itself informative.
"""
import os
import sys

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
from scipy.special import ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec_suite as cc                                     # noqa: E402
from fitmodel import Fit                                     # noqa: E402
from phase_loss import phase_entropies                       # noqa: E402
from sweep import noise_units, STEPS                         # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def cond_gauss_detail(z, order):
    """Conditional-Gaussian rate plus the residual scale it fits."""
    zf = np.asarray(z, dtype=np.float64)
    n = zf.size
    r = cc.autocorr(zf, order) / n
    a = solve_toeplitz(r[:order], r[1:order + 1])
    pred = lfilter(np.concatenate(([0.0], a)), [1.0], zf)
    zt, mu = zf[order:], pred[order:]
    s0 = np.sqrt(max((zt - mu).var() - 1.0 / 12.0, 1e-6))
    best_bits, best_s = np.inf, s0
    for s in s0 * np.linspace(0.7, 1.3, 25):
        p = ndtr((zt + 0.5 - mu) / s) - ndtr((zt - 0.5 - mu) / s)
        bits = float(-np.log2(np.maximum(p, 1e-12)).mean())
        if bits < best_bits:
            best_bits, best_s = bits, float(s)
    return best_bits + 8 * (4 * order + 14) / n, best_s


def main(n=200_000, order=32, trace="001290"):
    path = [p for p in sorted(os.listdir(CACHE))
            if trace in p and p.endswith(".raw.npy")][0]
    y = noise_units(np.load(os.path.join(CACHE, path)))[:n]
    print(f"{path}   n = {n}   order = {order}\n")
    print(f"  {'v':>5}{'s_*':>7}{'s_fit':>7} | "
          f"{'R_ANS':>7}{'H0':>7}{'R_cG':>7}{'G(s*)':>7} | "
          f"{'coder':>7}{'resid':>7}{'pred':>7} | {'L(s_fit)':>9}{'L(s_*)':>8}")
    print("  " + "-" * 92)
    for v in STEPS:
        z = np.round(y / v).astype(np.int16)
        if np.unique(z).size < 3:
            continue
        fit = Fit(z, nfft=4096)
        nb, h0 = cc.lpc_ans_bytes(z, order)
        r_ans = 8.0 * nb / z.size
        r_cg, s_fit = cond_gauss_detail(z, order)
        g = fit.predicted_rate
        coder, resid, pred = r_ans - h0, h0 - r_cg, r_cg - g
        print(f"  {v:5.2f}{fit.s_star:7.3f}{s_fit:7.3f} | "
              f"{r_ans:7.3f}{h0:7.3f}{r_cg:7.3f}{g:7.3f} | "
              f"{coder:7.3f}{resid:7.3f}{pred:7.3f} | "
              f"{phase_entropies(s_fit)[2]:9.3f}{phase_entropies(fit.s_star)[2]:8.3f}")


if __name__ == "__main__":
    main(trace=sys.argv[1] if len(sys.argv) > 1 else "001290")
