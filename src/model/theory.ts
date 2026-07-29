/**
 * The theoretical bits/sample: the entropy rate of the stationary Gaussian
 * process y = h * x with x ~ N(0, σ²) i.i.d., quantized at unit step. In the
 * fine-quantization (high-resolution) regime the discrete entropy rate
 * approaches the differential entropy rate (Kolmogorov):
 *
 *     R = ½ log₂(2πe) + ∫₀^{1/2} log₂ S(f) df,   S(f) = σ² |H(f)|²
 *
 * with f in cycles/sample. With no filter this reduces to ½ log₂(2πe σ²).
 * The formula ignores dither and degrades where S(f) falls to the order of
 * the quantization step or below — exactly the regime the app probes.
 */

const INTEGRATION_POINTS = 8192

export function entropyRateBits(kernel: Float64Array, sigma: number): number {
  const L = kernel.length
  let integral = 0
  for (let k = 0; k < INTEGRATION_POINTS; k++) {
    // Midpoint rule keeps f = 0 (where a bandpass H vanishes) off the grid;
    // the log singularity at isolated zeros is integrable.
    const f = (0.5 * (k + 0.5)) / INTEGRATION_POINTS
    let re = 0
    let im = 0
    for (let i = 0; i < L; i++) {
      re += kernel[i] * Math.cos(2 * Math.PI * f * i)
      im -= kernel[i] * Math.sin(2 * Math.PI * f * i)
    }
    const S = sigma * sigma * (re * re + im * im)
    integral += Math.log2(Math.max(S, 1e-300))
  }
  integral *= 0.5 / INTEGRATION_POINTS
  return 0.5 * Math.log2(2 * Math.PI * Math.E) + integral
}
