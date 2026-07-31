import { useEffect, useMemo, useRef, useState } from 'react'
import CopyableCommand from './components/CopyableCommand'
import Controls from './components/Controls'
import FilterViz from './components/FilterViz'
import ScrollingView from './components/ScrollingView'
import CompressionChart from './components/CompressionChart'
import MethodNote from './components/MethodNote'
import { DEFAULT_SPEC, clampSpec, designKernel, kernelNorm } from './model/filters'
import { LATENT_SEED } from './model/latent'
import { DEFAULT_LPC_ORDER, LPC_ORDERS } from './compress/codecs'
import type { BoundResult, CodecResult } from './compress/codecs'
import type { CompressRequest, CompressResponse } from './worker/compressWorker'

const BLOCK_SIZES = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]
const DEFAULT_BLOCK_SIZE = 100000

interface CompressionState {
  results: CodecResult[]
  bounds: BoundResult[]
  empiricalStd: number
  computing: boolean
  error: string | null
}

/** The nine codec sizes, measured in a worker on a debounced parameter set. */
function useCompression(
  kernel: Float64Array,
  sigma: number,
  lpcOrder: number,
  blockSize: number,
): CompressionState {
  const [state, setState] = useState<CompressionState>({
    results: [],
    bounds: [],
    empiricalStd: 0,
    computing: true,
    error: null,
  })
  const workerRef = useRef<Worker | null>(null)
  const idRef = useRef(0)

  useEffect(() => {
    const worker = new Worker(new URL('./worker/compressWorker.ts', import.meta.url), {
      type: 'module',
    })
    worker.onmessage = (e: MessageEvent<CompressResponse>) => {
      if (e.data.id !== idRef.current) return
      setState({
        results: e.data.error ? [] : e.data.results,
        bounds: e.data.error ? [] : e.data.bounds,
        empiricalStd: e.data.empiricalStd,
        computing: false,
        error: e.data.error ?? null,
      })
    }
    workerRef.current = worker
    return () => {
      worker.terminate()
      workerRef.current = null
    }
  }, [])

  useEffect(() => {
    setState(s => ({ ...s, computing: true }))
    const id = ++idRef.current
    const timer = setTimeout(() => {
      const request: CompressRequest = {
        id,
        kernel,
        sigma,
        blockSize,
        lpcOrder,
        // The same seed the signal view draws from, so the block really is
        // the data on screen.
        seed: LATENT_SEED,
      }
      workerRef.current?.postMessage(request)
    }, 250)
    return () => clearTimeout(timer)
  }, [kernel, sigma, lpcOrder, blockSize])

  return state
}

/**
 * The terminal command that estimates the reference rate R at the current
 * settings, using the unbiased Monte-Carlo estimator from the companion
 * timeseries-entropy package.
 */
function mcCommand(sigma: number, spec: ReturnType<typeof clampSpec>, rate: number): string {
  const parts = [
    'uvx --from git+https://github.com/concept-collection/timeseries-entropy',
    'timeseries-entropy',
    `--sigma ${sigma}`,
  ]
  switch (spec.family) {
    case 'none':
      parts.push('--filter none')
      break
    case 'movingAverage':
      parts.push('--filter moving-average', `--width ${spec.width}`)
      break
    case 'lowpass':
      parts.push('--filter lowpass', `--high ${spec.highHz}`, `--taps ${spec.taps}`, `--rate ${rate}`)
      break
    case 'bandpass':
      parts.push(
        '--filter bandpass',
        `--low ${spec.lowHz}`,
        `--high ${spec.highHz}`,
        `--taps ${spec.taps}`,
        `--rate ${rate}`,
      )
      break
    case 'firstDifference':
      parts.push('--filter first-difference')
      break
  }
  return parts.join(' ')
}

export default function App() {
  const [sigma, setSigma] = useState(5)
  const [sampleRateHz, setSampleRateHz] = useState(30000)
  const [spec, setSpec] = useState(DEFAULT_SPEC)
  const [lpcOrder, setLpcOrder] = useState(DEFAULT_LPC_ORDER)
  const [blockSize, setBlockSize] = useState(DEFAULT_BLOCK_SIZE)

  const kernel = useMemo(() => designKernel(spec, sampleRateHz), [spec, sampleRateHz])
  const sigmaY = useMemo(() => sigma * kernelNorm(kernel), [kernel, sigma])
  const compression = useCompression(kernel, sigma, lpcOrder, blockSize)

  return (
    <div className="app">
      <header className="app-header">
        <h1>Time-series compressibility</h1>
        <p>
          Gaussian noise → FIR filter → round to integers. How well can the integer stream be
          losslessly compressed, and how close do practical codecs get to the entropy rate of
          the process?
        </p>
      </header>

      {/* The controls stay pinned so the parameters and the ratios they move
          are always on screen together, whatever is scrolled to. */}
      <section className="card control-bar">
        <Controls
          sigma={sigma}
          setSigma={setSigma}
          sampleRateHz={sampleRateHz}
          setSampleRateHz={rate => {
            // Band edges are absolute, so a new rate can push them past
            // Nyquist; re-snap the spec so sliders and kernel stay in step.
            setSampleRateHz(rate)
            setSpec(s => clampSpec(s, rate))
          }}
          spec={spec}
          setSpec={setSpec}
        />
      </section>

      <section className="card">
        <h2>Compression</h2>
        <div className="stat-row">
          <div className="stat">
            <span className="label">predicted std of z</span>
            <span className="value">
              {sigmaY.toFixed(2)} <small>steps</small>
            </span>
          </div>
          <div className="stat">
            <span className="label">measured std of z</span>
            <span className="value">
              {compression.results.length > 0 ? compression.empiricalStd.toFixed(2) : '…'}{' '}
              <small>steps</small>
            </span>
          </div>
          {/* Placeholder: R will be estimated in the browser by the unbiased
              estimator; until then the command below produces it locally. */}
          <div className="stat">
            <span className="label">reference rate R</span>
            <span className="value">
              — <small>bits/sample</small>
            </span>
          </div>
        </div>
        {/* Settings of the measurement, not of the model — so they live with
            the chart they change rather than in the model bar. */}
        <div className="measure-row">
          <label>
            LPC order
            <select value={lpcOrder} onChange={e => setLpcOrder(Number(e.target.value))}>
              {LPC_ORDERS.map(o => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
          <label>
            block size
            <select value={blockSize} onChange={e => setBlockSize(Number(e.target.value))}>
              {BLOCK_SIZES.map(n => (
                <option key={n} value={n}>
                  {n.toLocaleString()} samples
                </option>
              ))}
            </select>
          </label>
        </div>
        {compression.error ? (
          <p className="card-note">Compression failed: {compression.error}</p>
        ) : (
          <CompressionChart
            results={compression.results}
            bounds={compression.bounds}
            computing={compression.computing}
          />
        )}
        <p className="card-note">
          Measured on a {blockSize.toLocaleString()}-sample block of the same latent data the
          signal view shows; sizes include everything a decoder needs (ANS symbol table, LPC
          coefficients). Baseline is raw int16 (16 bits/sample). The hollow bar in each group is
          that group's entropy limit — the order-0 entropy of the stream being coded, which no
          per-sample entropy coder can beat and ANS falls short of by its symbol table plus its
          own arithmetic loss. The reference rate R — the entropy rate of the process itself,
          the limit no lossless method can beat — is not yet computed in the browser; the
          command below estimates it locally (see the method section at the bottom).
        </p>
        <CopyableCommand
          label="reference rate R by unbiased Monte-Carlo (runs locally):"
          command={mcCommand(sigma, spec, sampleRateHz)}
        />
      </section>

      <section className="card">
        <h2>Quantized signal z</h2>
        <ScrollingView kernel={kernel} sigma={sigma} sigmaY={sigmaY} />
        <p className="card-note">
          A window of samples from the model, drawn from a fixed latent noise sequence — changing
          σ or the filter transforms the same underlying data, so the trace morphs rather than
          resampling. Press play to advance through the sequence.
        </p>
      </section>

      <section className="card">
        <h2>Filter</h2>
        <FilterViz kernel={kernel} sampleRateHz={sampleRateHz} />
      </section>

      <section className="card">
        <h2>The reference rate</h2>
        <MethodNote />
      </section>
    </div>
  )
}
