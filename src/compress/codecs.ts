/**
 * Lossless codecs run in the browser, on int16 sample data.
 *
 * The baseline throughout is the uncompressed int16 stream: 2 bytes = 16 bits
 * per sample. A ratio is (baseline bytes) / (compressed bytes).
 */
import { zlibSync } from 'fflate'
import { init as zstdInit, compress as zstdCompress } from '@bokuweb/zstd-wasm'
import { ansEncode, ansDecode, encodedSize } from './ans'
import { fitLpc, lpcResidual, lpcRestore, modelSize } from './lpc'
// Bundled as an asset so the page needs no network at run time. The path is
// relative rather than a bare specifier because the package's `exports` map
// has no entry for the wasm file, so `@bokuweb/zstd-wasm/dist/web/zstd.wasm`
// cannot be resolved.
import zstdWasmUrl from '../../node_modules/@bokuweb/zstd-wasm/dist/web/zstd.wasm?url'

export interface Codec {
  name: string
  /** Longer description, shown on hover. */
  note: string
  /** Compressed size in bytes, including any table the decoder needs. */
  size: (samples: Int16Array, bytes: Uint8Array) => number
}

/**
 * First differences, in int16 with wraparound — exactly invertible, and the
 * standard prefilter for signals whose neighbouring samples are related.
 */
function delta(samples: Int16Array): Int16Array {
  const d = new Int16Array(samples.length)
  d[0] = samples[0]
  for (let i = 1; i < samples.length; i++) {
    d[i] = ((samples[i] - samples[i - 1]) << 16) >> 16
  }
  return d
}

function undelta(d: Int16Array): Int16Array {
  const s = new Int16Array(d.length)
  s[0] = d[0]
  for (let i = 1; i < d.length; i++) {
    s[i] = ((s[i - 1] + d[i]) << 16) >> 16
  }
  return s
}

function asBytes(samples: Int16Array): Uint8Array {
  return new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength)
}

/**
 * Size of an ANS encoding of `coded`, after decoding it, running `inverse` to
 * undo whatever prefilter produced it, and checking the result against the
 * original samples — so a reported size always belongs to an encoding that
 * actually round-trips. `extraBytes` covers anything else the decoder needs,
 * such as predictor coefficients.
 */
function ansSize(
  samples: Int16Array,
  coded: Int16Array,
  inverse: (coded: Int16Array) => Int16Array,
  extraBytes = 0,
): number {
  const encoded = ansEncode(coded)
  const decoded = inverse(ansDecode(encoded))
  if (decoded.length !== samples.length) throw new Error('ANS round-trip length mismatch')
  for (let i = 0; i < samples.length; i++) {
    if (decoded[i] !== samples[i]) throw new Error(`ANS round-trip mismatch at ${i}`)
  }
  return encodedSize(encoded) + extraBytes
}

/** Predictor orders the UI offers; 32 is the FLAC default and ours. */
export const LPC_ORDERS = [1, 2, 4, 8, 16, 32, 64, 128]
export const DEFAULT_LPC_ORDER = 32

/** Fit the predictor and take the residual, with the coefficients' cost. */
function lpcTransform(
  samples: Int16Array,
  order: number,
): {
  residual: Int16Array
  restore: (residual: Int16Array) => Int16Array
  extraBytes: number
} {
  const model = fitLpc(samples, order)
  if (!model) throw new Error('LPC fit failed')
  return {
    residual: lpcResidual(samples, model),
    restore: residual => lpcRestore(residual, model),
    extraBytes: modelSize(model),
  }
}

export const ZLIB: Codec = {
  name: 'zlib -9',
  note: 'DEFLATE at maximum level — the HDF5 gzip filter',
  size: (_s, bytes) => zlibSync(bytes, { level: 9 }).length,
}

export const ZSTD: Codec = {
  name: 'zstd -19',
  note: 'Zstandard at level 19 — the Blosc/Zarr default family',
  size: (_s, bytes) => zstdCompress(bytes, 19).length,
}

export const ANS: Codec = {
  name: 'ANS',
  note: 'rANS entropy coder over the sample histogram, ported from simple_ans; the size includes the symbol table',
  size: samples => ansSize(samples, samples, x => x),
}

export const DELTA_ZLIB: Codec = {
  name: 'delta + zlib -9',
  note: 'First differences, then DEFLATE',
  size: samples => zlibSync(asBytes(delta(samples)), { level: 9 }).length,
}

export const DELTA_ZSTD: Codec = {
  name: 'delta + zstd -19',
  note: 'First differences, then Zstandard 19',
  size: samples => zstdCompress(asBytes(delta(samples)), 19).length,
}

export const DELTA_ANS: Codec = {
  name: 'delta + ANS',
  note: 'First differences, then the rANS entropy coder',
  size: samples => ansSize(samples, delta(samples), undelta),
}

/** The three LPC codecs at a given predictor order. */
export function lpcCodecs(order: number): Codec[] {
  const note = `Order-${order} linear prediction with integer coefficients; the size includes the coefficients`
  return [
    {
      name: `LPC(${order}) + zlib -9`,
      note: `${note}, then DEFLATE`,
      size: samples => {
        const { residual, extraBytes } = lpcTransform(samples, order)
        return zlibSync(asBytes(residual), { level: 9 }).length + extraBytes
      },
    },
    {
      name: `LPC(${order}) + zstd -19`,
      note: `${note}, then Zstandard 19`,
      size: samples => {
        const { residual, extraBytes } = lpcTransform(samples, order)
        return zstdCompress(asBytes(residual), 19).length + extraBytes
      },
    },
    {
      name: `LPC(${order}) + ANS`,
      note: `${note}, then the rANS entropy coder`,
      size: samples => {
        const { residual, restore, extraBytes } = lpcTransform(samples, order)
        return ansSize(samples, residual, restore, extraBytes)
      },
    },
  ]
}

/** The general-purpose compressors, which know nothing about the data. */
export const GENERAL_CODECS: Codec[] = [ZLIB, ZSTD]

/**
 * Order-0 (memoryless) entropy of an int16 stream, in bits per sample: what a
 * perfect entropy coder for the sample histogram would spend, with nothing
 * charged for describing that histogram. The ANS bars sit above this by the
 * symbol table plus the coder's own arithmetic loss.
 */
export function order0Entropy(samples: Int16Array): number {
  const counts = new Int32Array(65536)
  for (let i = 0; i < samples.length; i++) counts[samples[i] + 32768]++
  let bits = 0
  for (const c of counts) {
    if (c > 0) {
      const p = c / samples.length
      bits -= p * Math.log2(p)
    }
  }
  return bits
}

export interface BoundResult {
  /** The prefilter group this bounds, matching the codec groups. */
  group: string
  note: string
  bitsPerSample: number
  ratio: number
  bytes: number
}

/** The order-0 bound for each prefilter: raw samples, delta, LPC residual. */
export function entropyBounds(samples: Int16Array, lpcOrder: number): BoundResult[] {
  const streams: { group: string; what: string; data: Int16Array }[] = [
    { group: 'no prefilter', what: 'the samples themselves', data: samples },
    { group: 'delta', what: 'the first differences', data: delta(samples) },
    {
      group: 'LPC',
      what: `the order-${lpcOrder} prediction residual`,
      data: lpcTransform(samples, lpcOrder).residual,
    },
  ]
  return streams.map(({ group, what, data }) => {
    const bitsPerSample = order0Entropy(data)
    return {
      group,
      note: `order-0 entropy of ${what} — the limit for a per-sample entropy coder, with no symbol table or coefficients charged`,
      bitsPerSample,
      ratio: 16 / bitsPerSample,
      bytes: Math.ceil((bitsPerSample * samples.length) / 8),
    }
  })
}

let ready: Promise<void> | null = null

// zstd-wasm publishes the *node* build's types while the bundler resolves the
// browser build (its `exports` map has a "browser" condition). Only the browser
// build's `init` takes a wasm URL, so the signature has to be restated here.
const initWithUrl = zstdInit as unknown as (path?: string) => Promise<void>

/** Load the zstd wasm module. Safe to call repeatedly. */
export function initCodecs(): Promise<void> {
  if (!ready) {
    ready = initWithUrl(zstdWasmUrl).catch((err: unknown) => {
      ready = null
      throw err
    })
  }
  return ready
}

export interface CodecResult {
  codec: string
  note: string
  bytes: number
  ratio: number
  bitsPerSample: number
}

/** Compress one int16 buffer with each of the given codecs. */
export function compressAll(buffer: Uint8Array, codecs: Codec[]): CodecResult[] {
  // The session's buffer may sit at an odd offset; align before viewing as int16.
  const aligned = buffer.byteOffset % 2 === 0 ? buffer : new Uint8Array(buffer)
  const samples = new Int16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 2)
  return codecs.map(codec => {
    const bytes = codec.size(samples, aligned)
    return {
      codec: codec.name,
      note: codec.note,
      bytes,
      ratio: aligned.byteLength / bytes,
      bitsPerSample: (8 * bytes) / samples.length,
    }
  })
}
