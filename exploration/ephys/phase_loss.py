"""The fractional-phase loss as a universal function of s alone.

Given a real-valued prediction mu and prediction-error std s (both in
quantization steps), write d = mu - round(mu) for the fractional phase.
The conditional law of the integer residual r = round(y) - round(mu) is

    P(r = k | d) = Phi((k + 1/2 - d)/s) - Phi((k - 1/2 - d)/s).

A coder that knows mu pays the phase-averaged conditional entropy
    G(s) = E_d[ H(P(. | d)) ]                      (= theory.gauss_uniform_entropy)
A coder that codes the integer residual with one pooled histogram pays the
entropy of the phase-mixture
    M(s) = H( E_d[ P(. | d) ] ).
The difference L(s) = M(s) - G(s) >= 0 is a mutual information: what the
integer residual throws away about the phase. It depends on nothing but s.
"""
import numpy as np
from scipy.special import ndtr


def phase_entropies(s, kmax=None, nphase=2001):
    s = float(s)
    if s <= 0:
        return 0.0, 0.0, 0.0
    kmax = kmax or max(4, int(np.ceil(8 * s)) + 2)
    k = np.arange(-kmax, kmax + 1)[:, None]
    d = np.linspace(-0.5, 0.5, nphase)[None, :]
    p = ndtr((k + 0.5 - d) / s) - ndtr((k - 0.5 - d) / s)
    p = np.clip(p, 1e-300, None)
    p /= p.sum(axis=0, keepdims=True)

    def H(q):
        q = np.clip(q, 1e-300, None)
        return -(q * np.log2(q)).sum(axis=0)

    w = np.ones(nphase); w[0] = w[-1] = 0.5; w /= w.sum()
    cond = float(H(p) @ w)                    # G(s): knows the phase
    mix = float(H((p * w).sum(axis=1)))       # M(s): pools over phases
    return mix, cond, mix - cond


def curve(s_values):
    return np.array([phase_entropies(s) for s in s_values])


if __name__ == "__main__":
    print(f"{'s':>7}{'M(s) mixture':>14}{'G(s) cond':>12}{'L(s) loss':>12}")
    for s in [0.1, 0.2, 0.28, 0.33, 0.4, 0.5, 0.6, 0.74, 0.9, 1.0, 1.1,
              1.5, 1.66, 2.0, 3.0, 5.0]:
        m, g, l = phase_entropies(s)
        print(f"{s:7.2f}{m:14.4f}{g:12.4f}{l:12.4f}")

    print("\nAgainst the measured real-data gaps (001290 ch0, bandpassed):")
    print(f"  {'s_*':>6}{'predicted L':>13}{'measured gap':>14}")
    for s_star, measured in [(1.66, 0.072), (1.10, 0.065), (0.74, 0.101),
                             (0.50, 0.193), (0.33, 0.424)]:
        print(f"  {s_star:6.2f}{phase_entropies(s_star)[2]:13.4f}{measured:14.4f}")
