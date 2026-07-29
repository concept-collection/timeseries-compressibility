/**
 * Compression measurements off the main thread, so the scrolling view never
 * stutters while zstd -19 or the LPC fit runs. One message in (the model),
 * one message out (the nine codec results).
 */
import { Pipeline } from '../model/pipeline'
import {
  initCodecs,
  compressAll,
  ZLIB,
  ZSTD,
  ANS,
  DELTA_ZLIB,
  DELTA_ZSTD,
  DELTA_ANS,
  LPC_ZLIB,
  LPC_ZSTD,
  LPC_ANS,
  type CodecResult,
} from '../compress/codecs'

export interface CompressRequest {
  id: number
  kernel: Float64Array
  sigma: number
  dither: boolean
  blockSize: number
  seed: number
}

export interface CompressResponse {
  id: number
  results: CodecResult[]
  /** Empirical std of the quantized block, for display sanity. */
  empiricalStd: number
  error?: string
}

const CODECS = [ZLIB, ZSTD, ANS, DELTA_ZLIB, DELTA_ZSTD, DELTA_ANS, LPC_ZLIB, LPC_ZSTD, LPC_ANS]

const post = self.postMessage as (message: CompressResponse) => void

self.onmessage = async (e: MessageEvent<CompressRequest>) => {
  const { id, kernel, sigma, dither, blockSize, seed } = e.data
  try {
    await initCodecs()
    const samples = new Pipeline(kernel, sigma, dither, seed).next(blockSize)
    let sum = 0
    let sumSq = 0
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i]
      sumSq += samples[i] * samples[i]
    }
    const mean = sum / samples.length
    const empiricalStd = Math.sqrt(Math.max(0, sumSq / samples.length - mean * mean))
    const bytes = new Uint8Array(samples.buffer, 0, samples.byteLength)
    post({ id, results: compressAll(bytes, CODECS), empiricalStd })
  } catch (err) {
    post({ id, results: [], empiricalStd: 0, error: String(err) })
  }
}
