/**
 * The theoretical bits/sample for the quantized filtered-Gaussian process.
 *
 * The pure high-resolution entropy rate ½log₂(2πe) + ∫log₂ S(f) df diverges
 * to -∞ wherever the spectrum falls far below the quantization step, so it is
 * useless for filters with deep stopbands. Instead, model the roundoff as an
 * additive white noise floor σ_q² (1/12 without dither; 1/6 with, since the
 * dither itself is carried into the stored integers):
 *
 *     S_z(f) = S(f) + σ_q²,        S(f) = σ² |H(f)|²
 *
 * The one-step Wiener prediction error of that process (Szegő/Kolmogorov)
 *
 *     σ_e² = exp( 2 ∫₀^{1/2} ln S_z(f) df )
 *
 * is what an ideal predictor leaves behind; the rate is the exact entropy of
 * that innovation quantized at unit step, R = H_Δ(σ_e). Where S ≫ 1 this
 * reduces to the classical ½log₂(2πe σ_e²); in the coarse regime it stays
 * positive and finite. It remains an approximation — roundoff is not truly
 * white or independent — and testing it against LPC+ANS is the app's point.
 */

const INTEGRATION_POINTS = 8192

/**
 * Exact entropy (bits) of round(N(0, s²)) on the unit lattice. Per-bin
 * probabilities by Simpson integration of the density, so no erf is needed
 * and the tail keeps relative accuracy.
 */
export function quantizedGaussianEntropy(s: number): number {
  if (s <= 0.02) return 0
  const zMax = Math.ceil(8 * s + 4)
  // Enough points that a bin spans a few per standard deviation even when the
  // bin is wide compared to the distribution.
  const m = Math.min(401, Math.max(9, 2 * Math.ceil(3 / s) + 9)) | 1
  const h = 1 / (m - 1)
  const density = (u: number) => Math.exp((-u * u) / (2 * s * s)) / (Math.sqrt(2 * Math.PI) * s)
  let sumH = 0
  let total = 0
  for (let z = -zMax; z <= zMax; z++) {
    let acc = density(z - 0.5) + density(z + 0.5)
    for (let i = 1; i < m - 1; i++) acc += (i % 2 === 1 ? 4 : 2) * density(z - 0.5 + i * h)
    const p = (acc * h) / 3
    if (p > 0) {
      sumH -= p * Math.log2(p)
      total += p
    }
  }
  // Renormalize away the residual quadrature/truncation mass.
  return sumH / total + Math.log2(total)
}

export function theoreticalRateBits(kernel: Float64Array, sigma: number, dither: boolean): number {
  const noiseVar = dither ? 1 / 6 : 1 / 12
  const L = kernel.length
  let integral = 0
  for (let k = 0; k < INTEGRATION_POINTS; k++) {
    const f = (0.5 * (k + 0.5)) / INTEGRATION_POINTS
    let re = 0
    let im = 0
    for (let i = 0; i < L; i++) {
      re += kernel[i] * Math.cos(2 * Math.PI * f * i)
      im -= kernel[i] * Math.sin(2 * Math.PI * f * i)
    }
    integral += Math.log2(sigma * sigma * (re * re + im * im) + noiseVar)
  }
  integral *= 0.5 / INTEGRATION_POINTS
  // σ_e² = 2^(2·integral), so σ_e = 2^integral.
  return quantizedGaussianEntropy(2 ** integral)
}
