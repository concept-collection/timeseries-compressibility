/**
 * The generating model: x ~ N(0, σ²) i.i.d. → FIR filter → optional additive
 * uniform dither on [-1/2, 1/2) → round to integers (the quantization step is
 * the unit, so σ is measured in steps).
 *
 * A single streaming implementation feeds both the scrolling display and the
 * compression block, so what is compressed is exactly what is shown.
 */
import { GaussianStream } from './random'

export class Pipeline {
  private rng: GaussianStream
  /** Ring of the last kernel-length inputs; index 0 is the newest. */
  private history: Float64Array
  private pos = 0

  constructor(
    private kernel: Float64Array,
    private sigma: number,
    private dither: boolean,
    seed: number,
  ) {
    this.rng = new GaussianStream(seed)
    this.history = new Float64Array(kernel.length)
  }

  /** Generate the next n quantized samples, clamped into int16 range. */
  next(n: number): Int16Array {
    const { kernel, history } = this
    const L = kernel.length
    const out = new Int16Array(n)
    for (let j = 0; j < n; j++) {
      this.pos = (this.pos + L - 1) % L
      history[this.pos] = this.sigma * this.rng.normal()
      let y = 0
      for (let k = 0; k < L; k++) y += kernel[k] * history[(this.pos + k) % L]
      if (this.dither) y += this.rng.uniformCentered()
      out[j] = Math.max(-32768, Math.min(32767, Math.round(y)))
    }
    return out
  }
}
