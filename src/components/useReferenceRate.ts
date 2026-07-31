import { useEffect, useRef, useState } from 'react'
import { defaultPast } from '../entropy'
import type { EntropyRequest, EntropyUpdate } from '../worker/entropyWorker'

/** Fixed base seed: resumed runs continue the same per-past seed sequence,
 * so a stop/start pair reproduces an uninterrupted run exactly. */
const BASE_SEED = 20260731

export interface ReferenceRate {
  /** One unbiased estimate per independent past, in completion order. */
  perPast: number[]
  mean: number | null
  se: number | null
  running: boolean
  /** "past 3 · rep 5/8" while computing. */
  progress: string | null
  /** Conditioning window M. */
  past: number
  start: () => void
  stop: () => void
}

/**
 * The in-browser reference-rate estimate: a worker refines it (one
 * independent past at a time) until stopped, and any change to the model
 * invalidates both the values and a run in flight.
 */
export function useReferenceRate(kernel: Float64Array, sigma: number): ReferenceRate {
  const [perPast, setPerPast] = useState<number[]>([])
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<string | null>(null)
  const workerRef = useRef<Worker | null>(null)
  const perPastRef = useRef<number[]>([])
  const past = defaultPast(kernel.length)

  useEffect(() => {
    workerRef.current?.terminate()
    workerRef.current = null
    perPastRef.current = []
    setPerPast([])
    setRunning(false)
    setProgress(null)
  }, [kernel, sigma])

  useEffect(() => () => workerRef.current?.terminate(), [])

  const start = () => {
    if (workerRef.current) return
    const worker = new Worker(new URL('../worker/entropyWorker.ts', import.meta.url), {
      type: 'module',
    })
    worker.onmessage = (e: MessageEvent<EntropyUpdate>) => {
      const u = e.data
      if (u.type === 'past') {
        perPastRef.current = [...perPastRef.current, u.value]
        setPerPast(perPastRef.current)
      } else {
        setProgress(`past ${u.pastIndex + 1} · rep ${u.repsDone}/${u.reps}`)
      }
    }
    workerRef.current = worker
    const request: EntropyRequest = {
      kernel,
      sigma,
      past,
      seed: BASE_SEED,
      startPast: perPastRef.current.length,
    }
    worker.postMessage(request)
    setRunning(true)
    setProgress(null)
  }

  const stop = () => {
    workerRef.current?.terminate()
    workerRef.current = null
    setRunning(false)
    setProgress(null)
  }

  const n = perPast.length
  const mean = n > 0 ? perPast.reduce((a, b) => a + b, 0) / n : null
  let se: number | null = null
  if (mean !== null && n > 1) {
    const v = perPast.reduce((a, b) => a + (b - mean) * (b - mean), 0) / (n - 1)
    se = Math.sqrt(v / n)
  }

  return { perPast, mean, se, running, progress, past, start, stop }
}
