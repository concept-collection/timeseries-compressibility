/**
 * Compression measurements off the main thread, so the scrolling view never
 * stutters while zstd -19 or the LPC fit runs. One message in (the model),
 * one message out (the ten codec results).
 */
import { LatentSource } from '../model/latent'
import { conditionalGaussianCodec } from '../compress/conditionalGaussian'
import {
  initCodecs,
  compressAll,
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
  blockSize: number
  lpcOrder: number
  seed: number
}

export interface CompressResponse {
  id: number
  results: CodecResult[]
  /** Empirical std of the quantized block, for display sanity. */
  empiricalStd: number
  error?: string
}

const PLAIN_CODECS = [ZLIB, ZSTD, ANS, DELTA_ZLIB, DELTA_ZSTD, DELTA_ANS]

const post = self.postMessage as (message: CompressResponse) => void

self.onmessage = async (e: MessageEvent<CompressRequest>) => {
  const { id, kernel, sigma, blockSize, lpcOrder, seed } = e.data
  try {
    await initCodecs()
    const samples = new LatentSource(seed).window(0, blockSize, kernel, sigma)
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
      results: compressAll(samples, [
        ...PLAIN_CODECS,
        ...lpcCodecs(lpcOrder),
        conditionalGaussianCodec(lpcOrder),
      ]),
      empiricalStd,
    })
  } catch (err) {
    post({ id, results: [], empiricalStd: 0, error: String(err) })
  }
}
