"""Walk real ephys along the s_* axis by changing the quantization step.

The filtered ephys variant that benchcompress benchmarks fixes the step at
v = 0.25 noise units, which puts the noise at 4 quantization steps — the
high-resolution corner where integer-residual coding is already near optimal.
Coarser steps move the same recording down the s_* axis into the regime where
the fractional-phase loss is supposed to bite.

For each step size: requantize, fit the model, synthesize a surrogate, and
measure the prediction-based methods on both.

Usage:  python sweep.py [--n N] [--order O] [--zstd]
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.signal import butter, lfilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec_suite as cc                                     # noqa: E402
from fitmodel import Fit                                     # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
RATE = 30000.0
STEPS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def bandpass(x, lowcut=300.0, highcut=6000.0, rate=RATE):
    nyq = 0.5 * rate
    b, a = butter(5, [lowcut / nyq, highcut / nyq], btype="band")
    return lfilter(b, a, x)


def highpass(x, lowcut=300.0, rate=RATE):
    nyq = 0.5 * rate
    b, a = butter(5, lowcut / nyq, btype="high")
    return lfilter(b, a, x)


def noise_units(raw, rate=RATE, do_bandpass=True):
    """Trace scaled so that the MAD noise level is 1.0, optionally bandpassed
    first. With do_bandpass=False nothing shapes the spectrum but the
    acquisition hardware — the test of whether the model needs an explicit
    filtering step or only a spectrum."""
    x = np.asarray(raw, dtype=np.float64) - np.median(raw)
    xf = bandpass(x, rate=rate) if do_bandpass else x
    ref = xf if do_bandpass else highpass(x, rate=rate)
    nl = float(np.median(np.abs(ref - np.median(ref))) / 0.6745)
    return xf / nl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--order", type=int, default=32)
    p.add_argument("--zstd", action="store_true",
                   help="also measure zstd+delta as the practical baseline")
    p.add_argument("--nofilter", action="store_true",
                   help="skip the bandpass: sweep the unfiltered trace")
    args = p.parse_args()

    for path in sorted(glob.glob(os.path.join(CACHE, "*.raw.npy"))):
        name = os.path.basename(path)[: -len(".raw.npy")]
        y = noise_units(np.load(path), do_bandpass=not args.nofilter)[: args.n]
        shaping = ("unfiltered (acquisition spectrum only)" if args.nofilter
                   else "bandpass 300-6000 Hz")
        print(f"\n{'=' * 96}\n{name}   n = {y.size}   "
              f"({shaping}, noise std = 1.0 before quantization)\n{'=' * 96}")
        print(f"  {'step':>6}{'noise/step':>11}{'s_*':>8}{'G(s_*)':>9}"
              f"{'| LPC+ANS':>12}{'condG':>8}{'gap':>7}"
              f"{'| LPC+ANS':>12}{'condG':>8}{'gap':>7}{'| model-real':>13}")
        print(f"  {' ' * 34}{'real':>27}{'model surrogate':>27}")
        print(f"  {'-' * 94}")
        for v in STEPS:
            z = np.round(y / v).astype(np.int16)
            if np.unique(z).size < 3:
                continue
            fit = Fit(z, nfft=4096)
            zs = fit.synthesize(seed=0)

            def block(zz):
                nb, h0 = cc.lpc_ans_bytes(zz, args.order)
                return (8.0 * nb / zz.size,
                        cc.conditional_gaussian_rate(zz, args.order))

            ra, rc = block(z)
            ma, mc = block(zs)
            extra = ""
            if args.zstd:
                nz = cc.GENERIC["zstd"](cc.delta(z).tobytes())
                ns = cc.GENERIC["zstd"](cc.delta(zs).tobytes())
                extra = (f"   zstd+delta real {8.0 * nz / z.size:6.3f} "
                         f"model {8.0 * ns / zs.size:6.3f}")
            print(f"  {v:6.3f}{1.0 / v:11.2f}{fit.s_star:8.3f}"
                  f"{fit.predicted_rate:9.4f}"
                  f"{ra:12.4f}{rc:8.4f}{ra - rc:7.4f}"
                  f"{ma:12.4f}{mc:8.4f}{ma - mc:7.4f}"
                  f"{ma - ra:+13.4f}{extra}")


if __name__ == "__main__":
    main()
