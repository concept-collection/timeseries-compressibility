"""Does the paper's model explain real ephys compressibility?

For each cached trace: fit  x ~ N(0,1) -> h * x -> round  to its spectrum,
synthesize a surrogate of the same length from the fit, and run the same codec
suite on both. If the model is a good stand-in, every codec should land at the
same bits/sample on the surrogate as on the real trace.

Usage:  python compare.py [--n N] [--nfft NFFT] [--taps T] [--order O] [--fast]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec_suite as cc                                  # noqa: E402
from fitmodel import Fit                                     # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def report(name, z_real, args):
    z_real = np.asarray(z_real, dtype=np.int16)[: args.n]
    z_real = (z_real - int(np.median(z_real))).astype(np.int16)

    fit = Fit(z_real, nfft=args.nfft, n_taps=args.taps)
    z_syn = fit.synthesize(seed=0)

    print(f"\n{'=' * 78}\n{name}   n = {z_real.size}\n{'=' * 78}")
    print(f"  real:      std {z_real.std():8.3f}   "
          f"kurtosis {float(((z_real - z_real.mean()) ** 4).mean() / z_real.var() ** 2):6.2f}")
    print(f"  surrogate: std {z_syn.std():8.3f}   "
          f"kurtosis {float(((z_syn - z_syn.mean()) ** 4).mean() / z_syn.var() ** 2):6.2f}")
    print(f"  fit: taps {fit.kernel.size}  spectrum RMS error "
          f"{fit.kernel_error_db():.2f} dB")
    print(f"  s_* = {fit.s_star:.4f} quantization steps  ->  "
          f"predicted entropy rate G(s_*) = {fit.predicted_rate:.4f} bits/sample "
          f"({16 / max(fit.predicted_rate, 1e-9):.1f}x)")

    t0 = time.time()
    a = cc.measure(z_real, lpc_order=args.order, generic=not args.fast)
    b = cc.measure(z_syn, lpc_order=args.order, generic=not args.fast)
    print(f"  [{time.time() - t0:.1f}s]")

    print(f"\n  {'method':<22}{'real':>9}{'model':>9}{'diff':>8}"
          f"{'real x':>9}{'model x':>9}")
    print(f"  {'-' * 66}")
    for k in a:
        d = b[k] - a[k]
        print(f"  {k:<22}{a[k]:9.4f}{b[k]:9.4f}{d:+8.4f}"
              f"{16 / a[k]:9.2f}{16 / b[k]:9.2f}")
    return {"name": name, "fit": fit, "real": a, "model": b}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--nfft", type=int, default=4096)
    p.add_argument("--taps", type=int, default=None)
    p.add_argument("--order", type=int, default=32)
    p.add_argument("--fast", action="store_true",
                   help="skip the generic byte compressors")
    p.add_argument("--only", type=str, default=None)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(CACHE, "*.npy")))
    if args.only:
        paths = [q for q in paths if args.only in os.path.basename(q)]
    if not paths:
        sys.exit("no cached traces; run fetch.py first")

    for path in paths:
        name = os.path.basename(path)[: -len(".npy")]
        report(name, np.load(path), args)


if __name__ == "__main__":
    main()
