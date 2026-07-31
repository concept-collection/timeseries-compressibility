/**
 * Analytic prediction of the entropy rate of z = round(h * x), x iid
 * N(0, σ²) — the quantization-corrected formula from theory.py of the
 * sibling timeseries-entropy package (see its docstring for the
 * derivation); keep the two in step:
 *
 *     R ≈ G( √( exp( ∫₀¹ ln(σ² |H(f)|² + 1/12) df ) − 1/12 ) ),
 *
 * where G(s) is the differential entropy of N(0, s²) + U(−½, ½). Szegő's
 * one-step prediction error with roundoff as a 1/12 dither floor, fed
 * through the dithered-quantization entropy — finite at spectral zeros,
 * saturating to 0 at coarse quantization. Only this corrected prediction
 * is ported; the high-resolution Szegő form needs polynomial roots and is
 * not shown in the app.
 */
import { ndtr } from './normal'

const FLOOR = 1 / 12

/**
 * Mean over f in [0, 1) of ln(σ² |H(f)|² + 1/12): trapezoid on [0, 1/2]
 * (the spectrum is symmetric), with |H(f)|² = r₀ + 2 Σ_k r_k cos(2πfk)
 * from the kernel autocorrelation r, the cosines by Chebyshev recurrence.
 * n = 2¹⁴ intervals matches the Python 2¹⁸-point FFT integral to ~1e-8
 * over the app's whole kernel/σ range.
 */
export function logSpectrumMean(kernel: Float64Array, sigma: number, n = 1 << 14): number {
  const L = kernel.length
  const r = new Float64Array(L)
  for (let k = 0; k < L; k++) {
    let s = 0
    for (let j = 0; j + k < L; j++) s += kernel[j] * kernel[j + k]
    r[k] = s
  }
  const S = new Float64Array(n + 1).fill(r[0])
  for (let k = 1; k < L; k++) {
    const w = 2 * r[k]
    if (w === 0) continue
    const t = 2 * Math.cos((Math.PI * k) / n) // c_{i+1} = t·c_i − c_{i−1}
    let cPrev = 1
    let c = t / 2
    S[0] += w
    for (let i = 1; i <= n; i++) {
      S[i] += w * c
      const cNext = t * c - cPrev
      cPrev = c
      c = cNext
    }
  }
  const s2 = sigma * sigma
  let sum = 0
  for (let i = 0; i <= n; i++) {
    const v = Math.log(s2 * Math.max(S[i], 0) + FLOOR)
    sum += i === 0 || i === n ? v / 2 : v
  }
  return sum / n
}

/** G(s) = h(N(0, s²) + U(−½, ½)) in bits — the exact average entropy of
 * round(c + N(0, s²)) over a uniform grid offset c. */
export function gaussUniformEntropy(s: number): number {
  if (s <= 0) return 0
  if (s < 1e-3) return edgeConstant() * s
  const dv = Math.min(s / 8, 0.01)
  const vMax = 0.5 + 8 * s + 1
  let sum = 0
  for (let i = 0; i * dv < vMax; i++) {
    const v = i * dv
    const g = ndtr((v + 0.5) / s) - ndtr((v - 0.5) / s)
    const term = g > 0 ? -g * Math.log2(g) : 0
    sum += i === 0 ? term / 2 : term
  }
  return 2 * dv * sum
}

let EDGE_C: number | null = null

/** ∫ h₂(Φ(t)) dt: the small-s slope of G(s). */
function edgeConstant(): number {
  if (EDGE_C === null) {
    const n = 20001
    let sum = 0
    for (let i = 0; i < n; i++) {
      const t = -12 + (24 * i) / (n - 1)
      const p = Math.min(Math.max(ndtr(t), 1e-300), 1 - 1e-16)
      const h2 = -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p))
      sum += i === 0 || i === n - 1 ? h2 / 2 : h2
    }
    EDGE_C = (sum * 24) / (n - 1)
  }
  return EDGE_C
}

/** The predicted entropy rate G(s*) of z = round(h * x), in bits/sample. */
export function predictEntropyRate(kernel: Float64Array, sigma: number): number {
  const gmW = Math.exp(logSpectrumMean(kernel, sigma))
  return gaussUniformEntropy(Math.sqrt(Math.max(gmW - FLOOR, 0)))
}
