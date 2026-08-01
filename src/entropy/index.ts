/**
 * In-browser unbiased estimation of H(z_{M+1} | z_1..z_M) — the entropy
 * rate R — for the app's model. Hand-synced TypeScript port of the sibling
 * timeseries-entropy package (see the headers of estimator.ts / model.ts);
 * defaults match its CLI so browser runs and `timeseries-entropy` runs
 * target the same estimand with the same variance behavior.
 */
import { integratedAutocorrTime, unbiasedEntropy } from './estimator'
import { ConditionalChain } from './model'
import { Rng } from './rng'

export { unbiasedEntropy, integratedAutocorrTime } from './estimator'
export { ConditionalChain, truncatedStdNormal } from './model'
export { Rng } from './rng'
export { ndtr, ndtri, erfc } from './normal'
export { predictEntropyRate, gaussUniformEntropy, logSpectrumMean } from './theory'

export const N0 = 128
export const R_EXPONENT = 1.5
/** Per-past realization budget at thin = 1; the resolved thin divides it. */
export const REPS_PER_PAST = 8
/** Draws taken at thin = 1 to measure each chain's autocorrelation time. */
export const PROBE = 512
/** Cap on the auto-resolved thin (bounds cost; also the probe cannot
 * resolve times much beyond PROBE / 10). */
export const THIN_CAP = 64

/** The conditioning window M at a given kernel length, as in the CLI. */
export function defaultPast(kernelLength: number): number {
  return Math.max(512, 4 * kernelLength)
}

/** The seed for one independent past, derived so past i is reproducible
 * whether or not the run was stopped and resumed in between. */
export function pastSeed(baseSeed: number, pastIndex: number): number {
  return (baseSeed + Math.imul(0x9e3779b9, pastIndex + 1)) >>> 0
}

/**
 * One independent past's unbiased estimate: a fresh stationary chain,
 * auto-thinned to its measured mixing, averaged over randomized-telescoping
 * realizations run back-to-back on it. Averaging these over pasts estimates
 * H(z_{M+1} | z_1..z_M).
 *
 * Auto-thinning mirrors thin='auto' in the Python package: a PROBE-draw
 * pilot at thin = 1 estimates the chain's integrated autocorrelation time
 * tau, the chain then takes ceil(tau) sweeps per draw (capped at THIN_CAP),
 * and REPS_PER_PAST acts as a budget — realizations = max(1, budget/thin) —
 * so per-past cost stays roughly flat. Without this, slowly mixing chains
 * (narrowband kernels x large sigma) make the level corrections decay too
 * slowly for R_EXPONENT and the estimator's variance is infinite: still
 * unbiased, but rare realizations of hundreds of bits.
 */
export function estimateOnePast(
  kernel: Float64Array,
  sigma: number,
  past: number,
  seed: number,
  onRep?: (repsDone: number, reps: number) => void,
): number {
  const rng = new Rng(seed)
  const chain = new ConditionalChain(kernel, sigma, past, rng, 1)
  const tau = integratedAutocorrTime(chain.draw(PROBE))
  chain.thin = Math.min(THIN_CAP, Math.max(1, Math.ceil(tau)))
  const reps = Math.max(1, Math.round(REPS_PER_PAST / chain.thin))
  let sum = 0
  for (let rep = 0; rep < reps; rep++) {
    sum += unbiasedEntropy(chain.draw, N0, R_EXPONENT, rng)
    onRep?.(rep + 1, reps)
  }
  return sum / reps
}
