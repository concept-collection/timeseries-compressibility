/**
 * Conditional-Gaussian arithmetic coding: the prefilter-free way past the
 * order-0 entropy limit.
 *
 * LPC + a memoryless coder codes the *integer* residual with one pooled
 * histogram — the mixture over all fractional parts of the prediction. When
 * the prediction error is a fraction of a quantization step, that mixture
 * costs ~0.3-0.4 bits/sample more than the conditional entropy: whether the
 * real-valued prediction μ falls near a bin centre (~0.4 bits) or a bin edge
 * (~1.1 bits) is information an integer residual has already destroyed.
 *
 * This coder keeps it: the same order-p predictor as the LPC codecs, but with
 * float coefficients, and each sample is arithmetic-coded under the
 * discretized Gaussian N(μ_t, s²) — bin k gets Φ((k+½−μ)/s) − Φ((k−½−μ)/s),
 * tails folded into the edge bins, one escape symbol so that any int16 input
 * still round-trips. The decoder rebuilds μ_t from already-decoded samples
 * with the identical arithmetic, so both sides derive bit-identical tables.
 *
 * The header charges everything the decoder needs: the order, the float32
 * coefficients, the residual scale s, and the marginal std for the warm-up
 * samples (coded before p samples of context exist).
 */
import { ndtr } from '../entropy/normal'
import { fitRealCoeffs } from './lpc'
import type { Codec, CodecSize } from './codecs'

const TOTAL = 1 << 16
const FULL = 2 ** 32 - 1
const HALF = 2 ** 31
const QUARTER = 2 ** 30

/** Symbols cover residuals in [-kwin, kwin); beyond that the escape fires. */
const MAX_KWIN = 8192

class BitWriter {
  bytes: number[] = []
  private acc = 0
  private n = 0

  write(bit: number) {
    this.acc = this.acc * 2 + bit
    if (++this.n === 8) {
      this.bytes.push(this.acc)
      this.acc = 0
      this.n = 0
    }
  }

  flush() {
    while (this.n) this.write(0)
  }
}

class BitReader {
  private pos = 0
  constructor(private data: number[]) {}

  read(): number {
    const byte = this.pos >> 3 < this.data.length ? this.data[this.pos >> 3] : 0
    const bit = (byte >> (7 - (this.pos & 7))) & 1
    this.pos++
    return bit
  }
}

/** Witten–Neal–Cleary arithmetic coder, 32-bit width. All products stay
 * below 2^48, exact in a double, so no BigInt is needed. */
class ArithEncoder {
  private low = 0
  private high = FULL
  private pending = 0
  readonly out = new BitWriter()

  private emit(bit: number) {
    this.out.write(bit)
    for (; this.pending > 0; this.pending--) this.out.write(1 - bit)
  }

  encode(cum: number, freq: number, tot: number) {
    const span = this.high - this.low + 1
    this.high = this.low + Math.floor((span * (cum + freq)) / tot) - 1
    this.low = this.low + Math.floor((span * cum) / tot)
    for (;;) {
      if (this.high < HALF) {
        this.emit(0)
      } else if (this.low >= HALF) {
        this.emit(1)
        this.low -= HALF
        this.high -= HALF
      } else if (this.low >= QUARTER && this.high < HALF + QUARTER) {
        this.pending++
        this.low -= QUARTER
        this.high -= QUARTER
      } else {
        break
      }
      this.low *= 2
      this.high = this.high * 2 + 1
    }
  }

  finish(): number[] {
    this.pending++
    this.emit(this.low < QUARTER ? 0 : 1)
    this.out.flush()
    return this.out.bytes
  }
}

class ArithDecoder {
  private low = 0
  private high = FULL
  private value = 0
  private inp: BitReader

  constructor(data: number[]) {
    this.inp = new BitReader(data)
    for (let i = 0; i < 32; i++) this.value = this.value * 2 + this.inp.read()
  }

  target(tot: number): number {
    const span = this.high - this.low + 1
    return Math.floor(((this.value - this.low + 1) * tot - 1) / span)
  }

  consume(cum: number, freq: number, tot: number) {
    const span = this.high - this.low + 1
    this.high = this.low + Math.floor((span * (cum + freq)) / tot) - 1
    this.low = this.low + Math.floor((span * cum) / tot)
    for (;;) {
      if (this.high < HALF) {
        // nothing to subtract
      } else if (this.low >= HALF) {
        this.low -= HALF
        this.high -= HALF
        this.value -= HALF
      } else if (this.low >= QUARTER && this.high < HALF + QUARTER) {
        this.low -= QUARTER
        this.high -= QUARTER
        this.value -= QUARTER
      } else {
        break
      }
      this.low *= 2
      this.high = this.high * 2 + 1
      this.value = this.value * 2 + this.inp.read()
    }
  }
}

interface Table {
  kwin: number
  freqs: Int32Array // 2*kwin residual symbols, then the escape at index 2*kwin
  cum: Int32Array
}

/** Frequencies (sum TOTAL) of the discretized N(d, s²) over residual bins
 * [-kwin, kwin), tails folded into the edge bins, every bin ≥ 1, escape = 1.
 * Deterministic in (d, s): encoder and decoder call it with identical
 * doubles, so the tables agree bit for bit. */
function freqTable(d: number, s: number): Table {
  const kwin = Math.max(8, Math.min(MAX_KWIN, Math.ceil(10 * s) + 4))
  const nsym = 2 * kwin + 1
  const freqs = new Int32Array(nsym).fill(1)
  const spare = TOTAL - nsym
  const m = Math.min(kwin - 1, Math.floor(8 * s) + 2)

  // Bin probabilities for k in [-m, m]; the ends absorb their whole tails.
  const probs: number[] = []
  let loCdf = 0
  for (let k = -m; k <= m; k++) {
    const hiCdf = k === m ? 1 : ndtr((k + 0.5 - d) / s)
    probs.push(hiCdf - loCdf)
    loCdf = hiCdf
  }
  let assigned = 0
  const scaled = probs.map(p => {
    const f = Math.floor(p * spare)
    assigned += f
    return f
  })
  // Give the rounding deficit to the biggest bin.
  let imax = 0
  for (let i = 1; i < scaled.length; i++) if (scaled[i] > scaled[imax]) imax = i
  scaled[imax] += spare - assigned
  for (let i = 0; i < scaled.length; i++) freqs[i - m + kwin] += scaled[i]

  const cum = new Int32Array(nsym + 1)
  for (let i = 0; i < nsym; i++) cum[i + 1] = cum[i] + freqs[i]
  return { kwin, freqs, cum }
}

/**
 * Tables keyed by the quantized fractional phase d = μ − round(μ). The
 * quantization step grows with s (the phase matters less the wider the
 * bump: the KL cost is ~(Δd/s)²), which caps the cache at ~1k tables of
 * O(s)-sized windows. Encoder and decoder quantize identically, so the
 * cache is pure speed, not a protocol difference.
 */
function makeTableCache(s: number): (d: number) => Table {
  const dstep = Math.max(1 / 1024, s / 256)
  const cache = new Map<number, Table>()
  return d => {
    const key = Math.round(d / dstep)
    let table = cache.get(key)
    if (!table) {
      table = freqTable(key * dstep, s)
      cache.set(key, table)
    }
    return table
  }
}

/** μ_t from the p previous samples — the one shared piece of float
 * arithmetic both sides must reproduce exactly, hence one function. */
function predict(coeffs: Float64Array, zf: Float64Array, t: number): number {
  let mu = 0
  for (let k = 0; k < coeffs.length; k++) mu += coeffs[k] * zf[t - 1 - k]
  return Number.isFinite(mu) ? mu : 0
}

interface Model {
  coeffs: Float64Array // float32-rounded values, as the decoder will see them
  s: number
  stdZ: number
}

/** Header: order (2) + float32 coefficients + s (4) + warm-up std (4). */
function headerBytes(model: Model): number {
  return 2 + 4 * model.coeffs.length + 4 + 4
}

function fitModel(samples: Int16Array, order: number): Model {
  const a = fitRealCoeffs(samples, order)
  const coeffs = Float64Array.from(a ?? [], Math.fround)

  let sum = 0
  let sumSq = 0
  for (let i = 0; i < samples.length; i++) {
    sum += samples[i]
    sumSq += samples[i] * samples[i]
  }
  const n = samples.length
  const varZ = Math.max(0, sumSq / n - (sum / n) ** 2)

  // Residual scale: the variance of the residuals, but with outliers gated
  // at 8× a median-based scale first. On clean data nothing reaches 8σ and
  // this is exactly the residual variance; a lone spike would otherwise
  // inflate the variance — and with it every table's width — even though the
  // escape symbol already prices outliers individually.
  const zf = Float64Array.from(samples)
  const first = coeffs.length
  const total = Math.max(n - first, 0)
  const stride = Math.max(1, Math.ceil(total / (1 << 18)))
  const abs: number[] = []
  for (let t = first; t < n; t += stride) {
    abs.push(Math.abs(zf[t] - predict(coeffs, zf, t)))
  }
  abs.sort((x, y) => x - y)
  const medE = abs.length > 0 ? abs[abs.length >> 1] : Math.sqrt(varZ)
  const gate = Math.max(8 * 1.4826 * medE, 1)
  let errSq = 0
  let count = 0
  for (let t = first; t < n; t++) {
    const e = zf[t] - predict(coeffs, zf, t)
    if (Math.abs(e) <= gate) {
      errSq += e * e
      count++
    }
  }
  const varE = count > 0 ? errSq / count : varZ
  // The coding distribution round(N(μ, s²)) has variance ≈ s² + 1/12, so the
  // measured residual variance overshoots s² by the rounding term.
  const s = Math.fround(Math.max(Math.sqrt(Math.max(varE - 1 / 12, 0)), 0.02))
  const stdZ = Math.fround(Math.max(Math.sqrt(varZ), 0.05))
  return { coeffs, s, stdZ }
}

const UNIFORM_TOT = 256

/** The payload, and the ideal cost of the model it was coded against — the
 * sum of −log2 p over the symbols actually emitted. The gap between that and
 * the payload is the arithmetic coder's own rounding loss. */
function encodeAll(samples: Int16Array, model: Model): { payload: number[]; modelBits: number } {
  const { coeffs, s, stdZ } = model
  const order = coeffs.length
  const zf = Float64Array.from(samples)
  const enc = new ArithEncoder()
  const warm = freqTable(0, stdZ)
  const tableFor = makeTableCache(s)
  let modelBits = 0
  for (let t = 0; t < samples.length; t++) {
    const mu = t < order ? 0 : predict(coeffs, zf, t)
    const c = t < order ? 0 : Math.round(mu)
    const table = t < order ? warm : tableFor(mu - c)
    const sym = samples[t] - c + table.kwin
    if (sym >= 0 && sym < 2 * table.kwin) {
      enc.encode(table.cum[sym], table.freqs[sym], TOTAL)
      modelBits -= Math.log2(table.freqs[sym] / TOTAL)
    } else {
      const esc = 2 * table.kwin
      enc.encode(table.cum[esc], table.freqs[esc], TOTAL)
      const raw = samples[t] & 0xffff
      enc.encode(raw >> 8, 1, UNIFORM_TOT)
      enc.encode(raw & 0xff, 1, UNIFORM_TOT)
      modelBits += -Math.log2(table.freqs[esc] / TOTAL) + 16 // escape, then the raw sample
    }
  }
  return { payload: enc.finish(), modelBits }
}

function decodeAll(payload: number[], n: number, model: Model): Int16Array {
  const { coeffs, s, stdZ } = model
  const order = coeffs.length
  const zf = new Float64Array(n)
  const out = new Int16Array(n)
  const dec = new ArithDecoder(payload)
  const warm = freqTable(0, stdZ)
  const tableFor = makeTableCache(s)
  for (let t = 0; t < n; t++) {
    const mu = t < order ? 0 : predict(coeffs, zf, t)
    const c = t < order ? 0 : Math.round(mu)
    const table = t < order ? warm : tableFor(mu - c)
    const tgt = dec.target(TOTAL)
    let lo = 0
    let hi = table.cum.length - 1
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if (table.cum[mid] <= tgt) lo = mid
      else hi = mid
    }
    dec.consume(table.cum[lo], table.freqs[lo], TOTAL)
    let z: number
    if (lo < 2 * table.kwin) {
      z = lo - table.kwin + c
    } else {
      const hiByte = dec.target(UNIFORM_TOT)
      dec.consume(hiByte, 1, UNIFORM_TOT)
      const loByte = dec.target(UNIFORM_TOT)
      dec.consume(loByte, 1, UNIFORM_TOT)
      z = (((hiByte << 8) | loByte) << 16) >> 16
    }
    out[t] = z
    zf[t] = out[t]
  }
  return out
}

/**
 * Compressed size in bytes — header plus arithmetic-coded payload — after
 * decoding the payload and checking it reproduces the samples exactly, so a
 * reported size always belongs to an encoding that round-trips.
 */
export function conditionalGaussianSize(samples: Int16Array, order: number): CodecSize {
  const model = fitModel(samples, order)
  const { payload, modelBits } = encodeAll(samples, model)
  const decoded = decodeAll(payload, samples.length, model)
  if (decoded.length !== samples.length) {
    throw new Error('conditional-Gaussian round-trip length mismatch')
  }
  for (let i = 0; i < samples.length; i++) {
    if (decoded[i] !== samples[i]) {
      throw new Error(`conditional-Gaussian round-trip mismatch at ${i}`)
    }
  }
  return { bytes: payload.length + headerBytes(model), modelBits }
}

/** The codec, shaped like the others so compressAll can report it. */
export function conditionalGaussianCodec(order: number): Codec {
  return {
    name: `LPC(${order}) + cond. Gaussian AC`,
    note:
      `Order-${order} prediction kept at full precision; each sample arithmetic-coded ` +
      'under a discretized Gaussian centred on the real-valued prediction, using the ' +
      'fractional phase that integer residuals discard; the size includes the float32 ' +
      'coefficients',
    size: samples => conditionalGaussianSize(samples, order),
  }
}
