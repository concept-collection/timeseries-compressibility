/**
 * The theoretical bits/sample for the quantized filtered-Gaussian process:
 *
 *     R̂ = min(R_spec, R_samp)
 *
 *     R_spec = ∫₀¹ ½ log₂(2πe (S(f) + ν)) df,   S(f) = σ²|H(f)|²
 *     ν      = 1/12 (1/6 with dither)
 *     R_samp = exact entropy of one stored sample, N(0, v) rounded
 *              (+ uniform dither first when it is on),  v = σ² Σ h²
 *
 * R_spec is the Zamir–Feder rate of the dithered quantizer, counted per
 * Fourier mode: the signal modes are independent Gaussians of variance S(f),
 * and each mode of the i.i.d. roundoff-plus-dither noise mixes all N samples'
 * contributions, so it is Gaussianized by the CLT and enters at its full
 * variance ν — not at the entropy power 1/(2πe) an aligned scalar quantizer
 * would charge (the quantization lattice lives in the sample basis, not the
 * Fourier basis; charging entropy power is what made earlier versions of this
 * estimate underestimate the rate). A consequence worth naming: a dead band
 * inside a live process contributes ½log₂(2πe ν) ≈ 0.25 bits per mode, not
 * zero. What R_spec ignores is the cross-mode dependence of the cube noise —
 * at most ½log₂(2πe/12) ≈ 0.2546 bits/sample, in practice ≲ 0.02 unless
 * nearly the whole spectrum is noise-dominated.
 *
 * That failure mode is exactly the globally sub-threshold process, and there
 * subadditivity gives a rigorous ceiling with the right collapse: H(z) ≤
 * Σ H(zₙ) = N·R_samp, the marginal entropy of a single output sample. The
 * min selects it precisely where the spectral branch fails. Monte Carlo puts
 * R̂ within ~0.01–0.02 bits/sample for v ≳ 0.25, worst ~+0.03 near the
 * crossover; testing that against LPC+ANS is the app's point.
 */

const TWO_PI_E = 2 * Math.PI * Math.E
const INTEGRATION_POINTS = 8192

/**
 * Exact entropy (bits) of round(N(0, s²)) on the unit lattice. Per-bin
 * probabilities by Simpson integration of the density, so no erf is needed
 * and the tail keeps relative accuracy.
 */
export function quantizedGaussianEntropy(s: number): number {
  if (s <= 0.02) return 0
  // The discrete entropy approaches the differential entropy ½log₂(2πe s²)
  // from above like log₂e/(24 s²) — the Δ²/24 Fisher-information correction,
  // with the next term O(1/s⁴). At s ≥ 6 the corrected asymptote is within
  // 2·10⁻⁶ bits, so the sum is only ever taken over a handful of bins.
  if (s >= 6) return 0.5 * Math.log2(TWO_PI_E * s * s) + Math.LOG2E / (24 * s * s)
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

/**
 * Exact entropy (bits) of round(N(0, s²) + U[-½,½)) — the marginal of a
 * stored sample when dither is on. Conditioned on the Gaussian landing at t,
 * bin j is hit with probability equal to the overlap of the dither interval
 * with the bin, the triangular hat Λ(j−t) = max(0, 1−|j−t|); so p_j is the
 * density integrated against Λ, done by Simpson on each side of the kink.
 */
export function ditheredQuantizedGaussianEntropy(s: number): number {
  if (s <= 0) return 0
  // Approaches ½log₂(2πe s²) from above like log₂e/(12 s²) — the dither's
  // 1/12 of variance plus the Δ²/24 quantization correction, each worth
  // log₂e/(24 s²). Within 10⁻⁵ bits at s ≥ 6.
  if (s >= 6) return 0.5 * Math.log2(TWO_PI_E * s * s) + Math.LOG2E / (12 * s * s)
  if (s <= 0.1) {
    // Only the neighbors of zero are reachable, through the tip of the hat:
    // p±1 = ∫₀^∞ t φ_s(t) dt = s/√(2π), machine-exact in this range.
    const p1 = s / Math.sqrt(2 * Math.PI)
    const p0 = 1 - 2 * p1
    return -p0 * Math.log2(p0) - 2 * p1 * Math.log2(p1)
  }
  const zMax = Math.ceil(8 * s + 2)
  const m = Math.min(401, Math.max(9, 2 * Math.ceil(3 / s) + 9)) | 1
  const h = 1 / (m - 1)
  const density = (u: number) => Math.exp((-u * u) / (2 * s * s)) / (Math.sqrt(2 * Math.PI) * s)
  // Simpson of φ_s(t)·w(t) over [a, a+1] with w linear from w0 to w1.
  const half = (a: number, w0: number, w1: number) => {
    let acc = density(a) * w0 + density(a + 1) * w1
    for (let i = 1; i < m - 1; i++) {
      const t = i * h
      acc += (i % 2 === 1 ? 4 : 2) * density(a + t) * (w0 + (w1 - w0) * t)
    }
    return (acc * h) / 3
  }
  let sumH = 0
  let total = 0
  for (let j = -zMax; j <= zMax; j++) {
    const p = half(j - 1, 0, 1) + half(j, 1, 0)
    if (p > 0) {
      sumH -= p * Math.log2(p)
      total += p
    }
  }
  return sumH / total + Math.log2(total)
}

export function theoreticalRateBits(kernel: Float64Array, sigma: number, dither: boolean): number {
  const nu = dither ? 1 / 6 : 1 / 12
  const L = kernel.length
  let rspec = 0
  // Midpoints on [0, ½]; |H| is symmetric about ½ for a real kernel, so the
  // grid average equals the integral over the full frequency circle.
  for (let k = 0; k < INTEGRATION_POINTS; k++) {
    const f = (0.5 * (k + 0.5)) / INTEGRATION_POINTS
    let re = 0
    let im = 0
    for (let i = 0; i < L; i++) {
      re += kernel[i] * Math.cos(2 * Math.PI * f * i)
      im -= kernel[i] * Math.sin(2 * Math.PI * f * i)
    }
    rspec += 0.5 * Math.log2(TWO_PI_E * (sigma * sigma * (re * re + im * im) + nu))
  }
  rspec /= INTEGRATION_POINTS
  let v = 0
  for (let i = 0; i < L; i++) v += kernel[i] * kernel[i]
  v *= sigma * sigma
  const rsamp = dither
    ? ditheredQuantizedGaussianEntropy(Math.sqrt(v))
    : quantizedGaussianEntropy(Math.sqrt(v))
  return Math.min(rspec, rsamp)
}
