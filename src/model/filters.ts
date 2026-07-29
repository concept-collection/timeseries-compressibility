/**
 * FIR filter presets and their frequency response.
 *
 * The filter is always realized as an explicit convolution kernel, so the
 * pipeline is exactly x → h*x → (+dither) → round, and the theory can work
 * from |H(f)| of the same taps the data actually went through. Cutoffs are
 * given in Hz against a user-set sample rate; internally everything is in
 * normalized frequency (cycles/sample, Nyquist = 0.5).
 */

export type FilterFamily = 'none' | 'movingAverage' | 'lowpass' | 'bandpass' | 'firstDifference'

export interface FilterSpec {
  family: FilterFamily
  /** Low band edge, Hz (bandpass only). */
  lowHz: number
  /** Cutoff / high band edge, Hz (lowpass, bandpass). */
  highHz: number
  /** Kernel length for the windowed-sinc designs; forced odd. */
  taps: number
  /** Moving-average width. */
  width: number
}

export const FAMILY_LABELS: Record<FilterFamily, string> = {
  none: 'none (white noise)',
  movingAverage: 'moving average',
  lowpass: 'lowpass',
  bandpass: 'bandpass',
  firstDifference: 'first difference',
}

export const DEFAULT_SPEC: FilterSpec = {
  family: 'bandpass',
  lowHz: 300,
  highHz: 6000,
  taps: 101,
  width: 8,
}

/** Hamming-windowed sinc lowpass with unit DC gain; fc in cycles/sample. */
function windowedSincLowpass(fc: number, taps: number): Float64Array {
  const n = taps | 1
  const mid = (n - 1) / 2
  const h = new Float64Array(n)
  let sum = 0
  for (let i = 0; i < n; i++) {
    const t = i - mid
    const sinc = t === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * t) / (Math.PI * t)
    const w = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1))
    h[i] = sinc * w
    sum += h[i]
  }
  for (let i = 0; i < n; i++) h[i] /= sum
  return h
}

/** Clamp the spec's band edges into (0, Nyquist) for the given sample rate. */
export function clampSpec(spec: FilterSpec, sampleRateHz: number): FilterSpec {
  const nyq = sampleRateHz / 2
  const highHz = Math.min(Math.max(spec.highHz, 2), nyq * 0.98)
  const lowHz = Math.min(Math.max(spec.lowHz, 1), highHz * 0.9)
  return { ...spec, highHz, lowHz }
}

export function designKernel(spec: FilterSpec, sampleRateHz: number): Float64Array {
  const s = clampSpec(spec, sampleRateHz)
  switch (s.family) {
    case 'none':
      return new Float64Array([1])
    case 'movingAverage': {
      const w = Math.max(2, Math.round(s.width))
      return new Float64Array(w).fill(1 / w)
    }
    case 'lowpass':
      return windowedSincLowpass(s.highHz / sampleRateHz, s.taps)
    case 'bandpass': {
      const lo = windowedSincLowpass(s.lowHz / sampleRateHz, s.taps)
      const hi = windowedSincLowpass(s.highHz / sampleRateHz, s.taps)
      const h = new Float64Array(hi.length)
      for (let i = 0; i < h.length; i++) h[i] = hi[i] - lo[i]
      return h
    }
    case 'firstDifference':
      return new Float64Array([1, -1])
  }
}

/** ‖h‖₂ — the gain from input σ to the filtered signal's σ_y. */
export function kernelNorm(h: Float64Array): number {
  let sum = 0
  for (const v of h) sum += v * v
  return Math.sqrt(sum)
}

/** |H(f)| at `points` frequencies uniform on [0, 0.5] cycles/sample. */
export function magnitudeResponse(h: Float64Array, points: number): Float64Array {
  const out = new Float64Array(points)
  for (let k = 0; k < points; k++) {
    const f = (0.5 * k) / (points - 1)
    let re = 0
    let im = 0
    for (let i = 0; i < h.length; i++) {
      re += h[i] * Math.cos(2 * Math.PI * f * i)
      im -= h[i] * Math.sin(2 * Math.PI * f * i)
    }
    out[k] = Math.hypot(re, im)
  }
  return out
}
