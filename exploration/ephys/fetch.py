"""Download single-channel ephys traces from the benchcompress datasets and
cache them as .npy, together with the bandpass+requantized "filtered" variant
that benchcompress benchmarks.

Usage:  python fetch.py [outdir]
"""
import os
import sys

import numpy as np

CACHE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache")

NUM_SAMPLES = 500_000
RATE = 30000.0

# (name, dandi asset url, dataset path in the nwb, channel)
SOURCES = [
    ("ecephys-000876-ch45",
     "https://api.dandiarchive.org/api/assets/7e1de06d-d478-40e2-9b64-9dd04eafaa4c/download/",
     "/acquisition/ElectricalSeriesAP/data", 45),
    ("ecephys-000409-ch101",
     "https://api.dandiarchive.org/api/assets/c04f6b30-82bf-40e1-9210-34f0bcd8be24/download/",
     "/acquisition/ElectricalSeriesAp/data", 101),
    ("ecephys-001290-ch0",
     "https://api.dandiarchive.org/api/assets/78c99d23-da88-4ecd-9086-c488a126eac5/download/",
     "/acquisition/ElectricalSeriesAPImec/data", 0),
]


def bandpass(x, lowcut, highcut, rate):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * rate
    b, a = butter(5, [lowcut / nyq, highcut / nyq], btype="band")
    return lfilter(b, a, x)


def highpass(x, lowcut, rate):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * rate
    b, a = butter(5, lowcut / nyq, btype="high")
    return lfilter(b, a, x)


def noise_level(x, rate):
    xf = highpass(x, 300.0, rate)
    return float(np.median(np.abs(xf - np.median(xf))) / 0.6745)


def filtered_variant(x, rate=RATE, v=0.25, lowcut=300.0, highcut=6000.0):
    """benchcompress's `-filtered` transform: bandpass, normalize by the MAD
    noise level, requantize at step v (so the noise std is ~1/v = 4 steps)."""
    xf = bandpass(x - np.median(x), lowcut, highcut, rate)
    nl = noise_level(xf, rate)
    return np.round(xf / nl / v).astype(np.int16)


def main():
    os.makedirs(CACHE, exist_ok=True)
    for name, url, path, ch in SOURCES:
        raw_path = os.path.join(CACHE, f"{name}.raw.npy")
        filt_path = os.path.join(CACHE, f"{name}.filtered.npy")
        if os.path.exists(raw_path) and os.path.exists(filt_path):
            print(f"{name}: cached")
            continue
        print(f"{name}: downloading {NUM_SAMPLES} samples ...", flush=True)
        import lindi
        h5f = lindi.LindiH5pyFile.from_hdf5_file(url)
        ds = h5f[path]
        raw = np.asarray(ds[:NUM_SAMPLES, ch : ch + 1]).flatten().astype(np.int16)
        np.save(raw_path, raw)
        np.save(filt_path, filtered_variant(raw.astype(np.float64)))
        print(f"{name}: raw std={raw.std():.1f}  "
              f"filtered std={np.load(filt_path).std():.2f}", flush=True)


if __name__ == "__main__":
    main()
