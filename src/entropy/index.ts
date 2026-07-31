/**
 * In-browser unbiased estimation of H(z_{M+1} | z_1..z_M) — the reference
 * rate R — for the app's model. Hand-synced TypeScript port of the sibling
 * timeseries-entropy package (see the headers of estimator.ts / model.ts);
 * defaults match its CLI so browser runs and `timeseries-entropy` runs
 * target the same estimand with the same variance behavior.
 */
import { unbiasedEntropy } from './estimator'
import { ConditionalChain } from './model'
import { Rng } from './rng'

export { unbiasedEntropy } from './estimator'
export { ConditionalChain, truncatedStdNormal } from './model'
export { Rng } from './rng'
export { ndtr, ndtri, erfc } from './normal'

export const N0 = 128
export const R_EXPONENT = 1.5
export const REPS_PER_PAST = 8
export const THIN = 1

/** The conditioning window M at a given kernel length, as in the CLI. */
export function defaultPast(kernelLength: number): number {
  return Math.max(512, 4 * kernelLength)
}

/** The seed for one independent past, derived so past i is reproducible
 * whether or not the run was stopped and resumed in between. */
export function pastSeed(baseSeed: number, pastIndex: number): number {
  return (baseSeed + Math.imul(0x9e3779b9, pastIndex + 1)) >>> 0
}

/** One independent past's unbiased estimate: a fresh stationary chain,
 * averaged over reps randomized-telescoping realizations run back-to-back
 * on it. Averaging these over pasts estimates H(z_{M+1} | z_1..z_M). */
export function estimateOnePast(
  kernel: Float64Array,
  sigma: number,
  past: number,
  seed: number,
  onRep?: (repsDone: number) => void,
): number {
  const rng = new Rng(seed)
  const chain = new ConditionalChain(kernel, sigma, past, rng, THIN)
  let sum = 0
  for (let rep = 0; rep < REPS_PER_PAST; rep++) {
    sum += unbiasedEntropy(chain.draw, N0, R_EXPONENT, rng)
    onRep?.(rep + 1)
  }
  return sum / REPS_PER_PAST
}
