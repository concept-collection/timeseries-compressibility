import { useEffect, useMemo, useRef, useState } from 'react'
import Controls from './components/Controls'
import FilterViz from './components/FilterViz'
import ScrollingView from './components/ScrollingView'
import CompressionChart from './components/CompressionChart'
import MathSection from './components/MathSection'
import { DEFAULT_SPEC, designKernel, kernelNorm } from './model/filters'
import { entropyRateBits } from './model/theory'
import type { CodecResult } from './compress/codecs'
import type { CompressRequest, CompressResponse } from './worker/compressWorker'

const BLOCK_SIZE = 120000
const BLOCK_SEED = 20260729

interface CompressionState {
  results: CodecResult[]
  empiricalStd: number
  computing: boolean
  error: string | null
}

/** The nine codec sizes, measured in a worker on a debounced parameter set. */
function useCompression(kernel: Float64Array, sigma: number, dither: boolean): CompressionState {
  const [state, setState] = useState<CompressionState>({
    results: [],
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
        dither,
        blockSize: BLOCK_SIZE,
        seed: BLOCK_SEED,
      }
      workerRef.current?.postMessage(request)
    }, 250)
    return () => clearTimeout(timer)
  }, [kernel, sigma, dither])

  return state
}

export default function App() {
  const [sigma, setSigma] = useState(5)
  const [sampleRateHz, setSampleRateHz] = useState(30000)
  const [spec, setSpec] = useState(DEFAULT_SPEC)
  const [dither, setDither] = useState(false)

  const kernel = useMemo(() => designKernel(spec, sampleRateHz), [spec, sampleRateHz])
  const sigmaY = useMemo(() => {
    const filtered = sigma * kernelNorm(kernel)
    return dither ? Math.sqrt(filtered * filtered + 1 / 12) : filtered
  }, [kernel, sigma, dither])
  const theoryBits = useMemo(() => entropyRateBits(kernel, sigma), [kernel, sigma])
  const compression = useCompression(kernel, sigma, dither)

  return (
    <div className="app">
      <header className="app-header">
        <h1>Time-series compressibility</h1>
        <p>
          Gaussian noise → FIR filter → optional dither → round to integers. How well can the
          integer stream be losslessly compressed, and does the spectral entropy-rate formula
          predict the limit?
        </p>
      </header>

      <section className="card">
        <h2>Model</h2>
        <Controls
          sigma={sigma}
          setSigma={setSigma}
          sampleRateHz={sampleRateHz}
          setSampleRateHz={setSampleRateHz}
          spec={spec}
          setSpec={setSpec}
          dither={dither}
          setDither={setDither}
        />
      </section>

      <section className="card">
        <h2>Filter</h2>
        <FilterViz kernel={kernel} sampleRateHz={sampleRateHz} sigma={sigma} />
      </section>

      <section className="card">
        <h2>Quantized signal z</h2>
        <ScrollingView kernel={kernel} sigma={sigma} dither={dither} sigmaY={sigmaY} />
        <p className="card-note">
          A window of samples from the model, redrawn when parameters change; press play to watch
          it stream. Sample-and-hold rendering, so the integer staircase appears as σ approaches
          the quantization step.
        </p>
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
          <div className="stat">
            <span className="label">entropy rate R (theory)</span>
            <span className="value">
              {theoryBits.toFixed(2)} <small>bits/sample</small>
            </span>
          </div>
          <div className="stat">
            <span className="label">implied best ratio</span>
            <span className="value">{theoryBits > 0 ? `${(16 / theoryBits).toFixed(2)}×` : '—'}</span>
          </div>
        </div>
        {compression.error ? (
          <p className="card-note">Compression failed: {compression.error}</p>
        ) : (
          <CompressionChart
            results={compression.results}
            theoryBits={theoryBits}
            computing={compression.computing}
          />
        )}
        <p className="card-note">
          Measured on a {BLOCK_SIZE.toLocaleString()}-sample block from the same model; sizes
          include everything a decoder needs (ANS symbol table, LPC coefficients). Baseline is
          raw int16 (16 bits/sample). The dashed line is the high-resolution entropy rate R — it
          ignores dither and is unreliable where S(f) falls below one step² (see the response
          plot).
        </p>
      </section>

      <section className="card">
        <h2>The math</h2>
        <MathSection />
      </section>
    </div>
  )
}
