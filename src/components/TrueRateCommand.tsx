import { useState } from 'react'
import type { FilterSpec } from '../model/filters'

/** The scripts/true_rate.py invocation matching the current parameters. */
export function trueRateCommand(
  sigma: number,
  spec: FilterSpec,
  sampleRateHz: number,
  dither: boolean,
): string {
  const parts = ['python scripts/true_rate.py', `--sigma ${sigma}`]
  switch (spec.family) {
    case 'none':
      parts.push('--filter none')
      break
    case 'firstDifference':
      parts.push('--filter first-difference')
      break
    case 'movingAverage':
      parts.push('--filter moving-average', `--width ${spec.width}`)
      break
    case 'lowpass':
      parts.push(
        '--filter lowpass',
        `--cutoff-hz ${spec.highHz}`,
        `--taps ${spec.taps}`,
        `--sample-rate ${sampleRateHz}`,
      )
      break
    case 'bandpass':
      parts.push(
        '--filter bandpass',
        `--low-hz ${spec.lowHz}`,
        `--high-hz ${spec.highHz}`,
        `--taps ${spec.taps}`,
        `--sample-rate ${sampleRateHz}`,
      )
      break
  }
  if (dither) parts.push('--dither')
  return parts.join(' ')
}

/**
 * The Monte Carlo cross-check, as a copy-pasteable command. The script
 * estimates the true entropy rate (docs/mc-true-rate.md) so the dashed R can
 * be compared against ground truth for the current parameters.
 */
export default function TrueRateCommand(props: {
  sigma: number
  spec: FilterSpec
  sampleRateHz: number
  dither: boolean
}) {
  const [copied, setCopied] = useState(false)
  const command = trueRateCommand(props.sigma, props.spec, props.sampleRateHz, props.dither)
  return (
    <div className="cli-row">
      <span className="cli-label">check R by Monte Carlo:</span>
      <code>{command}</code>
      <button
        className="copy-btn"
        onClick={() => {
          navigator.clipboard.writeText(command).then(() => {
            setCopied(true)
            setTimeout(() => setCopied(false), 1500)
          })
        }}
      >
        {copied ? 'copied ✓' : 'copy'}
      </button>
    </div>
  )
}
