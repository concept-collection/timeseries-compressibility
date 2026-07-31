/**
 * The process and its conditional Gibbs sampler: x iid N(0, σ²) → y = h*x
 * (causal FIR) → z = round(y), and a stationary chain of exact draws of
 * z_{M+1} given a fixed past z_1..z_M.
 *
 * TypeScript port of model.py from the sibling timeseries-entropy package —
 * keep the two in step. The numpy version updates same-color coordinate
 * blocks vectorized; here a sweep is a plain sequential scan over the
 * coordinates, an equally valid systematic-scan Gibbs sweep. This CPU sweep
 * is the piece a WebGPU backend would replace.
 */
import { ndtr, ndtri } from './normal'
import type { Rng } from './rng'

/** Standard normal truncated to [lo, hi], by inverse CDF. Mirrored into the
 * lower tail so the CDF differences keep precision. */
export function truncatedStdNormal(lo: number, hi: number, rng: Rng): number {
  const flip = lo > -hi // midpoint above 0 (robust to (-inf, inf) intervals)
  const a = flip ? -hi : lo
  const b = flip ? -lo : hi
  const fa = ndtr(a)
  const fb = ndtr(b)
  const u = Math.min(Math.max(fa + (fb - fa) * rng.uniform(), 1e-300), 1 - 1e-16)
  let x = ndtri(u)
  if (flip) x = -x
  return Math.min(Math.max(x, lo), hi)
}

/**
 * Constructing the chain draws the past from the prior; the generating
 * latents are themselves an exact draw from p(x | z), so the Gibbs chain
 * starts in stationarity — no burn-in bias, only autocorrelation. Each Gibbs
 * conditional x_i | rest is N(0, σ²) truncated to the interval read off the
 * ≤ L constraint boxes x_i appears in. draw(k) advances the chain k steps
 * (thin sweeps each) and returns k sampled z_{M+1} values, each marginally
 * distributed exactly as z_{M+1} | z_1..z_M.
 */
export class ConditionalChain {
  private readonly h: Float64Array
  private readonly sigma: number
  private readonly thin: number
  private readonly rng: Rng
  private readonly L: number
  private readonly M: number
  /** Latents x_0..x_{M+L-2}; boxes live in padded rows so that coordinate i
   * sees exactly L constraint rows i..i+L-1 (rows outside the data are
   * unconstrained), with coefficient h[j] in row i+j. */
  private readonly x: Float64Array
  private readonly ypad: Float64Array
  private readonly lo: Float64Array
  private readonly hi: Float64Array

  constructor(kernel: Float64Array, sigma: number, past: number, rng: Rng, thin = 1) {
    if (kernel.length === 0) throw new Error('kernel must be nonempty')
    if (!(sigma > 0)) throw new Error('sigma must be positive')
    if (past < 1) throw new Error('past must be >= 1')
    this.h = kernel
    this.sigma = sigma
    this.thin = thin
    this.rng = rng
    const L = (this.L = kernel.length)
    const M = (this.M = past)

    const x = (this.x = new Float64Array(M + L - 1))
    for (let i = 0; i < x.length; i++) x[i] = sigma * rng.normal()

    const P = L - 1
    this.ypad = new Float64Array(M + 2 * P)
    this.lo = new Float64Array(M + 2 * P).fill(-Infinity)
    this.hi = new Float64Array(M + 2 * P).fill(Infinity)
    this.refreshY()
    for (let m = 0; m < M; m++) {
      const z = Math.floor(this.ypad[P + m] + 0.5)
      this.lo[P + m] = z - 0.5
      this.hi[P + m] = z + 0.5
    }
  }

  /** ypad[P + m] = y_m = Σ_j h[j] x_{m+L-1-j}, recomputed to kill fp drift. */
  private refreshY(): void {
    const { h, x, ypad, L, M } = this
    const P = L - 1
    for (let m = 0; m < M; m++) {
      let y = 0
      for (let j = 0; j < L; j++) y += h[j] * x[m + L - 1 - j]
      ypad[P + m] = y
    }
  }

  private sweep(): void {
    const { h, x, ypad, lo, hi, sigma, rng, L } = this
    this.refreshY()
    for (let i = 0; i < x.length; i++) {
      const xi = x[i]
      let xlo = -Infinity
      let xhi = Infinity
      for (let j = 0; j < L; j++) {
        const hj = h[j]
        if (hj === 0) continue
        const row = i + j
        const res = ypad[row] - hj * xi
        const b1 = (lo[row] - res) / hj
        const b2 = (hi[row] - res) / hj
        if (hj > 0) {
          if (b1 > xlo) xlo = b1
          if (b2 < xhi) xhi = b2
        } else {
          if (b2 > xlo) xlo = b2
          if (b1 < xhi) xhi = b1
        }
      }
      const xn = truncatedStdNormal(xlo / sigma, xhi / sigma, rng) * sigma
      const d = xn - xi
      if (d !== 0) {
        for (let j = 0; j < L; j++) ypad[i + j] += d * h[j]
        x[i] = xn
      }
    }
  }

  /** The next k samples of z_{M+1}, continuing the chain. */
  draw = (k: number): Int32Array => {
    const { h, x, sigma, rng, L, M } = this
    const out = new Int32Array(k)
    for (let s = 0; s < k; s++) {
      for (let t = 0; t < this.thin; t++) this.sweep()
      // z_{M+1} = round(c + h[0] · x_free) with x_free ~ N(0, σ²) fresh.
      let c = 0
      for (let i = 0; i < L - 1; i++) c += h[L - 1 - i] * x[M + i]
      out[s] = Math.floor(c + sigma * h[0] * rng.normal() + 0.5)
    }
    return out
  }
}
