/**
 * The entropy-rate estimator off the main thread: one message starts an
 * endless refinement loop that posts one unbiased value per completed past
 * (with per-rep progress in between). Stopping is worker termination — the
 * loop never yields, but outgoing messages still flow, and per-past values
 * already delivered survive on the main thread, so a later run resumes at
 * startPast with the same derived seeds as an uninterrupted one.
 */
import { estimateOnePast, pastSeed, REPS_PER_PAST } from '../entropy'

export interface EntropyRequest {
  kernel: Float64Array
  sigma: number
  /** Conditioning window M. */
  past: number
  seed: number
  /** Index of the first past to compute (count already done). */
  startPast: number
}

export type EntropyUpdate =
  | { type: 'progress'; pastIndex: number; repsDone: number; reps: number }
  | { type: 'past'; pastIndex: number; value: number }

const post = self.postMessage as (message: EntropyUpdate) => void

self.onmessage = (e: MessageEvent<EntropyRequest>) => {
  const { kernel, sigma, past, seed, startPast } = e.data
  for (let i = startPast; ; i++) {
    // The rep count is only known once the past's mixing probe resolves the
    // thinning; until the first onRep, report the thin=1 budget.
    post({ type: 'progress', pastIndex: i, repsDone: 0, reps: REPS_PER_PAST })
    const value = estimateOnePast(kernel, sigma, past, pastSeed(seed, i), (repsDone, reps) =>
      post({ type: 'progress', pastIndex: i, repsDone, reps }),
    )
    post({ type: 'past', pastIndex: i, value })
  }
}
