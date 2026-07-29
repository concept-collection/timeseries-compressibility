/**
 * Compression measurements off the main thread, so the scrolling view never
 * stutters while zstd -19 or the LPC fit runs. One message in (the model),
 * one message out (the nine codec results).
 */
import { LatentSource } from '../model/latent'
import {
  initCodecs,
  compressAll,
  entropyBounds,
  type BoundResult,
  ZLIB,
  ZSTD,
  ANS,
  DELTA_ZLIB,
  DELTA_ZSTD,
  DELTA_ANS,
  lpcCodecs,
  type CodecResult,
} from '../compress/codecs'

export interface CompressRequest {
  id: number
  kernel: Float64Array
  sigma: number
  dither: boolean
  blockSize: number
  lpcOrder: number
  seed: number
}

export interface CompressResponse {
  id: number
  results: CodecResult[]
  /** Order-0 entropy limit for each prefilter group. */
  bounds: BoundResult[]
  /** Empirical std of the quantized block, for display sanity. */
  empiricalStd: number
  error?: string
}

const PLAIN_CODECS = [ZLIB, ZSTD, ANS, DELTA_ZLIB, DELTA_ZSTD, DELTA_ANS]

const post = self.postMessage as (message: CompressResponse) => void

self.onmessage = async (e: MessageEvent<CompressRequest>) => {
  const { id, kernel, sigma, dither, blockSize, lpcOrder, seed } = e.data
  try {
    await initCodecs()
    const samples = new LatentSource(seed).window(0, blockSize, kernel, sigma, dither)
    let sum = 0
    let sumSq = 0
    for (let i = 0; i < samples.length; i++) {
      sum += samples[i]
      sumSq += samples[i] * samples[i]
    }
    const mean = sum / samples.length
    const empiricalStd = Math.sqrt(Math.max(0, sumSq / samples.length - mean * mean))
    post({
      id,
      results: compressAll(samples, [...PLAIN_CODECS, ...lpcCodecs(lpcOrder)]),
      bounds: entropyBounds(samples, lpcOrder),
      empiricalStd,
    })
  } catch (err) {
    post({ id, results: [], bounds: [], empiricalStd: 0, error: String(err) })
  }
}
