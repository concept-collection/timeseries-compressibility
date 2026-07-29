import { useEffect, useRef } from 'react'
import { Pipeline } from '../model/pipeline'

/** Display samples generated per second; px per sample is fixed below. */
const RATE = 220
const PX_PER_SAMPLE = 2
const RING_SIZE = 8192

/** A nice round gridline step ≤ span/2. */
function niceStep(span: number): number {
  const raw = span / 2
  const mag = 10 ** Math.floor(Math.log10(raw))
  for (const m of [5, 2.5, 2, 1]) if (m * mag <= raw) return m * mag
  return mag
}

/**
 * The endlessly generated quantized signal z, drawn sample-and-hold so the
 * integer staircase is visible once σ is small. Rendering is a canvas ring
 * buffer fed by the same Pipeline the compression worker uses.
 */
export default function ScrollingView(props: {
  kernel: Float64Array
  sigma: number
  dither: boolean
  /** Predicted std of the quantized signal, for a stable y-scale. */
  sigmaY: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const seedRef = useRef(1)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const pipeline = new Pipeline(props.kernel, props.sigma, props.dither, seedRef.current++)
    const ring = new Float32Array(RING_SIZE)
    let head = 0
    let filled = 0
    const push = (samples: Int16Array) => {
      for (let i = 0; i < samples.length; i++) {
        ring[head] = samples[i]
        head = (head + 1) % RING_SIZE
      }
      filled = Math.min(RING_SIZE, filled + samples.length)
    }

    // Start with a full screen of history so the view is never empty.
    push(pipeline.next(2048))

    const scale = Math.max(4 * props.sigmaY, 3.5)
    const gridStep = niceStep(scale)

    let styles = getComputedStyle(canvas)
    const scheme = window.matchMedia('(prefers-color-scheme: dark)')
    const refreshStyles = () => {
      styles = getComputedStyle(canvas)
    }
    scheme.addEventListener('change', refreshStyles)

    let raf = 0
    let last = performance.now()
    let carry = 0

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw)
      const dt = Math.min(0.25, (now - last) / 1000)
      last = now
      carry += dt * RATE
      const n = Math.floor(carry)
      carry -= n
      if (n > 0) push(pipeline.next(n))

      const dpr = window.devicePixelRatio || 1
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      if (w === 0 || h === 0) return
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr)
        canvas.height = Math.round(h * dpr)
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      const surface = styles.getPropertyValue('--surface')
      ctx.fillStyle = surface
      ctx.fillRect(0, 0, w, h)

      const yOf = (v: number) => h / 2 - (v / scale) * (h / 2 - 12)

      ctx.strokeStyle = styles.getPropertyValue('--grid')
      ctx.lineWidth = 1
      ctx.fillStyle = styles.getPropertyValue('--muted')
      ctx.font = '11px system-ui, sans-serif'
      ctx.textAlign = 'left'
      for (let g = -2; g <= 2; g++) {
        const v = g * gridStep
        if (Math.abs(v) > scale) continue
        const y = Math.round(yOf(v)) + 0.5
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(w, y)
        if (g !== 0) ctx.stroke()
        // A surface-colored halo keeps the label readable over the trace.
        ctx.strokeStyle = surface
        ctx.lineWidth = 3
        const label = `${v > 0 ? '+' : ''}${+v.toPrecision(3)}`
        ctx.strokeText(label, 6, y - 4)
        ctx.fillText(label, 6, y - 4)
        ctx.strokeStyle = styles.getPropertyValue('--grid')
        ctx.lineWidth = 1
      }
      const zeroY = Math.round(yOf(0)) + 0.5
      ctx.strokeStyle = styles.getPropertyValue('--baseline')
      ctx.beginPath()
      ctx.moveTo(0, zeroY)
      ctx.lineTo(w, zeroY)
      ctx.stroke()

      const visible = Math.min(filled, Math.floor(w / PX_PER_SAMPLE))
      ctx.strokeStyle = styles.getPropertyValue('--series-1')
      ctx.lineWidth = 2
      ctx.lineJoin = 'round'
      ctx.beginPath()
      for (let i = 0; i < visible; i++) {
        const idx = (head - visible + i + RING_SIZE) % RING_SIZE
        const x = w - (visible - i) * PX_PER_SAMPLE
        const y = yOf(ring[idx])
        // Sample-and-hold: horizontal run at each value, vertical jump between.
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
        ctx.lineTo(x + PX_PER_SAMPLE, y)
      }
      ctx.stroke()
    }
    raf = requestAnimationFrame(draw)

    return () => {
      cancelAnimationFrame(raf)
      scheme.removeEventListener('change', refreshStyles)
    }
  }, [props.kernel, props.sigma, props.dither, props.sigmaY])

  return <canvas ref={canvasRef} className="scroll-canvas" />
}
