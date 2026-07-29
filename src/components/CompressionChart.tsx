import { useRef, useState } from 'react'
import type { CodecResult } from '../compress/codecs'
import { useWidth } from './useWidth'

const GROUPS = ['no prefilter', 'delta', `LPC`]
const CODER_NAMES = ['zlib', 'zstd', 'ANS']
const CODER_VARS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)']

const LABEL_W = 96
const RIGHT_PAD = 64
const AXIS_H = 26
const GROUP_H = 22
const ROW_H = 26
const BAR_H = 16

type Metric = 'bits' | 'ratio'

interface Tip {
  x: number
  y: number
  result: CodecResult
}

function axisTicks(max: number): number[] {
  const step = max > 24 ? 8 : max > 12 ? 4 : max > 6 ? 2 : max > 3 ? 1 : 0.5
  const out: number[] = []
  for (let v = 0; v <= max + 1e-9; v += step) out.push(v)
  return out
}

/** A bar whose data-end is rounded (4px) while the baseline end stays square. */
function barPath(x0: number, y: number, len: number, h: number): string {
  const r = Math.min(4, len)
  return `M${x0},${y} h${len - r} a${r},${r} 0 0 1 ${r},${r} v${h - 2 * r} a${r},${r} 0 0 1 ${-r},${r} h${-(len - r)} z`
}

export default function CompressionChart(props: {
  results: CodecResult[]
  theoryBits: number
  computing: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const width = useWidth(ref, 720)
  const [metric, setMetric] = useState<Metric>('bits')
  const [tip, setTip] = useState<Tip | null>(null)
  const [hovered, setHovered] = useState<number | null>(null)

  const { results, theoryBits } = props
  if (results.length === 0) {
    return <p className="card-note">Computing compression on the first block…</p>
  }

  const value = (r: CodecResult) => (metric === 'bits' ? r.bitsPerSample : r.ratio)
  const theoryValue = metric === 'bits' ? theoryBits : theoryBits > 0 ? 16 / theoryBits : NaN
  const theoryVisible = Number.isFinite(theoryValue) && theoryValue > 0
  const xMax =
    metric === 'bits'
      ? Math.max(16, ...results.map(value), theoryVisible ? theoryValue : 0) * 1.02
      : Math.max(...results.map(value), theoryVisible ? theoryValue : 0) * 1.1

  const plotW = width - LABEL_W - RIGHT_PAD
  const height = AXIS_H + GROUPS.length * (GROUP_H + 3 * ROW_H) + 6
  const xOf = (v: number) => LABEL_W + (v / xMax) * plotW
  const rowY = (i: number) => AXIS_H + Math.floor(i / 3) * (GROUP_H + 3 * ROW_H) + GROUP_H + (i % 3) * ROW_H

  const fmt = (r: CodecResult) =>
    metric === 'bits' ? r.bitsPerSample.toFixed(2) : `${r.ratio.toFixed(2)}×`

  const onBarMove = (e: React.PointerEvent, r: CodecResult) => {
    const box = ref.current!.getBoundingClientRect()
    setTip({ x: e.clientX - box.left, y: e.clientY - box.top, result: r })
  }

  const theoryX = theoryVisible ? xOf(theoryValue) : 0
  const theoryLabel =
    metric === 'bits' ? `entropy rate R = ${theoryBits.toFixed(2)}` : `R ⇒ ${(16 / theoryBits).toFixed(2)}×`

  return (
    <div>
      <div className="chart-header">
        <div>
          <div className="segmented" role="group" aria-label="metric">
            <button className={metric === 'bits' ? 'active' : ''} onClick={() => setMetric('bits')}>
              bits / sample
            </button>
            <button className={metric === 'ratio' ? 'active' : ''} onClick={() => setMetric('ratio')}>
              compression ratio
            </button>
          </div>
          <span className="metric-hint">
            {metric === 'bits' ? 'lower is better' : 'vs int16 — higher is better'}
          </span>
        </div>
        <div className="legend">
          {CODER_NAMES.map((name, i) => (
            <span key={name}>
              <span className="swatch" style={{ background: CODER_VARS[i] }} />
              {name}
            </span>
          ))}
        </div>
      </div>
      <div className={`chart-body${props.computing ? ' computing' : ''}`} ref={ref}>
        {props.computing && <span className="computing-badge">computing…</span>}
        <svg width={width} height={height}>
          {axisTicks(xMax).map(v => (
            <g key={v}>
              <line x1={xOf(v)} x2={xOf(v)} y1={AXIS_H - 6} y2={height - 4} stroke="var(--grid)" strokeWidth={1} />
              <text x={xOf(v)} y={AXIS_H - 10} textAnchor="middle" className="axis-tick">
                {+v.toFixed(1)}
              </text>
            </g>
          ))}
          <line x1={xOf(0)} x2={xOf(0)} y1={AXIS_H - 6} y2={height - 4} stroke="var(--baseline)" strokeWidth={1} />
          {GROUPS.map((g, gi) => (
            <text key={g} x={0} y={AXIS_H + gi * (GROUP_H + 3 * ROW_H) + 15} className="bar-group-label">
              {g}
            </text>
          ))}
          {results.map((r, i) => {
            const y = rowY(i)
            const len = Math.max(1, (value(r) / xMax) * plotW)
            const label = fmt(r)
            return (
              <g key={r.codec} opacity={hovered === null || hovered === i ? 1 : 0.45}>
                <text x={8} y={y + BAR_H / 2 + 4} className="bar-row-label">
                  {CODER_NAMES[i % 3]}
                </text>
                <path d={barPath(xOf(0), y, len, BAR_H)} fill={CODER_VARS[i % 3]} />
                <text x={xOf(0) + len + 6} y={y + BAR_H / 2 + 4} className="bar-value">
                  {label}
                </text>
                <rect
                  x={0}
                  y={y - (ROW_H - BAR_H) / 2}
                  width={width}
                  height={ROW_H}
                  fill="transparent"
                  onPointerMove={e => {
                    setHovered(i)
                    onBarMove(e, r)
                  }}
                  onPointerLeave={() => {
                    setHovered(null)
                    setTip(null)
                  }}
                />
              </g>
            )
          })}
          {theoryVisible && (
            <g>
              <line
                x1={theoryX}
                x2={theoryX}
                y1={AXIS_H - 2}
                y2={height - 4}
                stroke="var(--ink-2)"
                strokeWidth={1.5}
                strokeDasharray="5 4"
              />
              <text
                x={theoryX + (theoryX > width - 150 ? -6 : 6)}
                y={AXIS_H + 10}
                textAnchor={theoryX > width - 150 ? 'end' : 'start'}
                className="bar-value"
                fill="var(--ink)"
              >
                {theoryLabel}
              </text>
            </g>
          )}
        </svg>
        {tip && (
          <div className="viz-tooltip" style={{ left: tip.x + 14, top: tip.y - 8 }}>
            <div>
              <span className="tip-value">
                {tip.result.bitsPerSample.toFixed(3)} bits/sample · {tip.result.ratio.toFixed(2)}×
              </span>{' '}
              <span className="tip-label">{tip.result.codec}</span>
            </div>
            <div className="tip-label">
              {tip.result.bytes.toLocaleString()} bytes · {tip.result.note}
            </div>
          </div>
        )}
      </div>
      <details className="chart-table">
        <summary>Table view</summary>
        <table>
          <thead>
            <tr>
              <th>method</th>
              <th>bytes</th>
              <th>bits/sample</th>
              <th>ratio vs int16</th>
            </tr>
          </thead>
          <tbody>
            {results.map(r => (
              <tr key={r.codec}>
                <td>{r.codec}</td>
                <td>{r.bytes.toLocaleString()}</td>
                <td>{r.bitsPerSample.toFixed(3)}</td>
                <td>{r.ratio.toFixed(3)}</td>
              </tr>
            ))}
            {theoryBits > 0 && (
              <tr>
                <td>theory: entropy rate R</td>
                <td>—</td>
                <td>{theoryBits.toFixed(3)}</td>
                <td>{(16 / theoryBits).toFixed(3)}</td>
              </tr>
            )}
          </tbody>
        </table>
      </details>
    </div>
  )
}
