/**
 * Rhee–Glynn (randomized telescoping) unbiased entropy estimation.
 *
 * TypeScript port of estimator.py from the sibling timeseries-entropy
 * package — keep the two in step. See that file for the full derivation;
 * in short: plug-in entropies of blocks whose sizes double per level form a
 * telescoping sum via the antithetic correction
 * Δ_m = h(B_m) − [h(B_m¹) + h(B_m²)]/2; truncating at a random level N with
 * P(N ≥ m) = 2^(−r m) and reweighting by the survival probabilities gives
 * an estimator whose expectation is exactly the entropy of the stationary
 * marginal, despite the bias of every finite-block plug-in estimate and any
 * autocorrelation of the draws. All entropies are in bits.
 */
import type { Rng } from './rng'

/** draw(k) returns the next k consecutive samples of a stationary chain. */
export type Draw = (k: number) => Int32Array

function entropyFromCounts(counts: Map<number, number>, n: number): number {
  let s = 0
  for (const c of counts.values()) s += c * Math.log2(c)
  return Math.log2(n) - s / n
}

function countInto(counts: Map<number, number>, seg: Int32Array): void {
  for (let i = 0; i < seg.length; i++) {
    counts.set(seg[i], (counts.get(seg[i]) ?? 0) + 1)
  }
}

/** h(B_0) and [Δ_1 … Δ_levels] over one growing block; counts merge upward
 * so the cost is linear in the n0 * 2**levels samples drawn. */
function telescope(draw: Draw, n0: number, levels: number): { h0: number; deltas: number[] } {
  const counts = new Map<number, number>()
  countInto(counts, draw(n0))
  let size = n0
  const h0 = entropyFromCounts(counts, size)
  let hPrev = h0
  const deltas: number[] = []
  for (let m = 0; m < levels; m++) {
    const half = new Map<number, number>()
    countInto(half, draw(size))
    const h2 = entropyFromCounts(half, size)
    for (const [v, c] of half) counts.set(v, (counts.get(v) ?? 0) + c)
    size *= 2
    const hFull = entropyFromCounts(counts, size)
    deltas.push(hFull - 0.5 * (hPrev + h2))
    hPrev = hFull
  }
  return { h0, deltas }
}

/**
 * One randomized-telescoping realization of the marginal entropy (bits).
 * Consumes n0 * 2**N samples with P(N ≥ m) = 2^(−r m); average many
 * realizations (they may continue one chain back-to-back) — each has
 * expectation exactly H. r = 1.5 suits the typical Δ second-moment decay
 * of 2^(−2m); finite work needs r > 1, finite variance needs decay > r.
 */
export function unbiasedEntropy(draw: Draw, n0: number, r: number, rng: Rng): number {
  const rho = 2 ** -r
  let N = 0
  while (rng.uniform() < rho) N++
  const { h0, deltas } = telescope(draw, n0, N)
  let est = h0
  for (let m = 1; m <= N; m++) est += deltas[m - 1] * 2 ** (r * m)
  return est
}
