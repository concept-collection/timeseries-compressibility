"""Achievable rate of a model-based Monte-Carlo codec (the nonlinear optimum).

The Gaussian-conditional codec (codec_gaussian.py) is linear: it conditions on
the past only through a linear prediction and charges a fixed Gaussian. The
true conditional law P(z_t | z_{t-M..t-1}) is the posterior predictive of the
generative model — computable by Gibbs-sampling the latent x under the box
constraints round(h*x) = z (exactly the machinery of timeseries-entropy) and
Rao-Blackwellizing:

    P_hat(k) = mean over sweeps of  Phi((k+1/2-c)/(sigma h0)) - Phi((k-1/2-c)/(sigma h0)),
    c = hr_head . x[M:]

An encoder/decoder pair sharing the RNG seed can both compute P_hat from
already-decoded samples, so   rate = E[-log2 P_tilde(z_t)]   is achievable
(P_tilde mixes in a wide floor distribution so no symbol gets probability 0).
The -log2 of a finite-sweep average is on average >= the -log2 of the true
predictive (Jensen), so this measures an upper bound that tightens with more
sweeps: whatever number comes out IS achievable by this codec family.

Chain start: the generating latents of the window, an exact draw from
p(x | all z) — near the target p(x | window z); BURN sweeps then wash the
difference. Reuses ConditionalChain's sweep by overwriting its state.
"""
import numpy as np
from scipy.special import ndtr

import sys
from pathlib import Path

# The companion package, cloned alongside this repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'timeseries-entropy' / 'src'))
from timeseries_entropy.model import ConditionalChain

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


def make_kernel():
    return windowed_sinc_lowpass(HIGH / RATE, TAPS) - windowed_sinc_lowpass(LOW / RATE, TAPS)


def posterior_predictive_rate(M=512, burn=30, sweeps=60, positions=300,
                              spacing=64, seed=7, eps=1e-3):
    h = make_kernel()
    L = len(h)
    rng = np.random.default_rng(seed)

    n = positions * spacing + M + 4 * L
    x = SIGMA * rng.standard_normal(n + L - 1)
    y = np.convolve(x, h, mode='valid')
    z = np.floor(y + 0.5)

    h0 = abs(h[0])
    print(f'sigma*|h0| (innovation scale given full latent past) = {SIGMA * h0:.5f}')

    # wide fallback so P_tilde is never 0 (what a real codec would also do)
    std_z = z.std()
    kmax = 32
    ks = np.arange(-kmax, kmax + 1)
    fallback = ndtr((ks + 0.5) / std_z) - ndtr((ks - 0.5) / std_z)
    fallback /= fallback.sum()

    proto = ConditionalChain(h, SIGMA, past=M, rng=rng)  # template; state overwritten
    rates = []
    hit_floor = 0
    for i in range(positions):
        t = M + i * spacing                      # predict z[t] from z[t-M:t]
        i0 = t - M
        proto.z = z[i0:t].copy()
        proto.x = x[i0:t + L - 1].copy()         # exact posterior draw (given all z)
        P = L - 1
        proto.lo[P:P + M] = proto.z - 0.5
        proto.hi[P:P + M] = proto.z + 0.5

        for _ in range(burn):
            proto._sweep()
        acc = 0.0
        zt = int(z[t])
        for _ in range(sweeps):
            proto._sweep()
            c = float(proto.hr_head @ proto.x[M:])
            acc += (ndtr((zt + 0.5 - c) / (SIGMA * h0))
                    - ndtr((zt - 0.5 - c) / (SIGMA * h0)))
        p_hat = acc / sweeps
        p_tilde = (1 - eps) * p_hat + eps * float(fallback[np.clip(zt, -kmax, kmax) + kmax])
        if p_hat < eps:
            hit_floor += 1
        rates.append(-np.log2(p_tilde))
        if (i + 1) % 50 == 0:
            r = np.array(rates)
            print(f'  {i + 1:4d}/{positions}: rate = {r.mean():.4f} '
                  f'+/- {r.std(ddof=1) / np.sqrt(len(r)):.4f} bits/sample '
                  f'(floor hits: {hit_floor})')
    r = np.array(rates)
    print(f'\nMC-codec achievable rate (M={M}, {sweeps} sweeps): '
          f'{r.mean():.4f} +/- {r.std(ddof=1) / np.sqrt(len(r)):.4f} bits/sample '
          f'-> ratio {16 / r.mean():.1f}x')
    return r


if __name__ == '__main__':
    posterior_predictive_rate()
