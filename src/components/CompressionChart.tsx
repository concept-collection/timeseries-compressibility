import { useRef, useState } from 'react'
import type { BoundResult, CodecResult } from '../compress/codecs'
import { useWidth } from './useWidth'

const GROUPS = ['no prefilter', 'delta', 'LPC']
const CODER_NAMES = ['zlib', 'zstd', 'ANS']
const CODER_VARS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)']
const BOUND_LABEL = 'entropy limit'

const LABEL_W = 100
const RIGHT_PAD = 64
const AXIS_H = 26
const GROUP_H = 20
const ROW_H = 24
const BAR_H = 16
const ROWS_PER_GROUP = 4

type Metric = 'bits' | 'ratio'

/** One drawn bar: a measured codec size, or the entropy limit for its group. */
interface Row {
  key: string
  label: string
  name: string
  note: string
  bytes: number
  bitsPerSample: number
  ratio: number
  isBound: boolean
  color: string
}

interface Tip {
  x: number
  y: number
  row: Row
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

/** Codec rows then the entropy limit, group by group. */
function buildRows(results: CodecResult[], bounds: BoundResult[]): Row[] {
  const rows: Row[] = []
  GROUPS.forEach((group, g) => {
    CODER_NAMES.forEach((coder, c) => {
      const r = results[g * 3 + c]
      if (!r) return
      rows.push({
        key: r.codec,
        label: coder,
        name: r.codec,
        note: r.note,
        bytes: r.bytes,
        bitsPerSample: r.bitsPerSample,
        ratio: r.ratio,
        isBound: false,
        color: CODER_VARS[c],
      })
    })
    const b = bounds.find(x => x.group === group)
    if (b) {
      rows.push({
        key: `${group}-bound`,
        label: BOUND_LABEL,
        name: `${BOUND_LABEL} (${group})`,
        note: b.note,
        bytes: b.bytes,
        bitsPerSample: b.bitsPerSample,
        ratio: b.ratio,
        isBound: true,
        color: 'var(--muted)',
      })
    }
  })
  return rows
}

export default function CompressionChart(props: {
  results: CodecResult[]
  bounds: BoundResult[]
  /** The browser-estimated entropy rate R, once at least one past is in. */
  rateBits: number | null
  computing: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const width = useWidth(ref, 720)
  const [metric, setMetric] = useState<Metric>('ratio')
  const [tip, setTip] = useState<Tip | null>(null)
  const [hovered, setHovered] = useState<string | null>(null)

  const { results, bounds, rateBits } = props
  if (results.length === 0) {
    return <p className="card-note">Computing compression on the first block…</p>
  }

  const rows = buildRows(results, bounds)
  const value = (r: { bitsPerSample: number; ratio: number }) =>
    metric === 'bits' ? r.bitsPerSample : r.ratio
  const rateValue =
    rateBits !== null && rateBits > 0 ? (metric === 'bits' ? rateBits : 16 / rateBits) : null
  const xMax =
    metric === 'bits'
      ? Math.max(16, ...rows.map(value), rateValue ?? 0) * 1.02
      : Math.max(...rows.map(value), rateValue ?? 0) * 1.1

  const plotW = width - LABEL_W - RIGHT_PAD
  const height = AXIS_H + GROUPS.length * (GROUP_H + ROWS_PER_GROUP * ROW_H) + 6
  const xOf = (v: number) => LABEL_W + (v / xMax) * plotW
  const rowY = (i: number) =>
    AXIS_H +
    Math.floor(i / ROWS_PER_GROUP) * (GROUP_H + ROWS_PER_GROUP * ROW_H) +
    GROUP_H +
    (i % ROWS_PER_GROUP) * ROW_H

  const fmt = (r: Row) => (metric === 'bits' ? r.bitsPerSample.toFixed(2) : `${r.ratio.toFixed(2)}×`)

  const onBarMove = (e: React.PointerEvent, row: Row) => {
    const box = ref.current!.getBoundingClientRect()
    setTip({ x: e.clientX - box.left, y: e.clientY - box.top, row })
  }

  const rateX = rateValue !== null ? xOf(rateValue) : 0
  const rateLabel =
    rateBits !== null && rateBits > 0
      ? metric === 'bits'
        ? `R = ${rateBits.toFixed(2)}`
        : `R ⇒ ${(16 / rateBits).toFixed(2)}×`
      : ''

  return (
    <div>
      <div className="chart-header">
        <div>
          <div className="segmented" role="group" aria-label="metric">
            <button className={metric === 'ratio' ? 'active' : ''} onClick={() => setMetric('ratio')}>
              compression ratio
            </button>
            <button className={metric === 'bits' ? 'active' : ''} onClick={() => setMetric('bits')}>
              bits / sample
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
          <span>
            <span className="swatch hollow" />
            entropy limit (not achieved)
          </span>
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
            <text
              key={g}
              x={0}
              y={AXIS_H + gi * (GROUP_H + ROWS_PER_GROUP * ROW_H) + 15}
              className="bar-group-label"
            >
              {g}
            </text>
          ))}
          {rows.map((r, i) => {
            const y = rowY(i)
            const len = Math.max(1, (value(r) / xMax) * plotW)
            return (
              <g key={r.key} opacity={hovered === null || hovered === r.key ? 1 : 0.45}>
                <text
                  x={8}
                  y={y + BAR_H / 2 + 4}
                  className={r.isBound ? 'bar-row-label bound' : 'bar-row-label'}
                >
                  {r.label}
                </text>
                {r.isBound ? (
                  // Hollow: a limit nobody reached, not a measured size.
                  <path
                    d={barPath(xOf(0), y + 1, len, BAR_H - 2)}
                    fill="var(--muted)"
                    fillOpacity={0.12}
                    stroke="var(--muted)"
                    strokeWidth={1.25}
                  />
                ) : (
                  <path d={barPath(xOf(0), y, len, BAR_H)} fill={r.color} />
                )}
                <text
                  x={xOf(0) + len + 6}
                  y={y + BAR_H / 2 + 4}
                  className={r.isBound ? 'bar-value bound' : 'bar-value'}
                >
                  {fmt(r)}
                </text>
                <rect
                  x={0}
                  y={y - (ROW_H - BAR_H) / 2}
                  width={width}
                  height={ROW_H}
                  fill="transparent"
                  onPointerMove={e => {
                    setHovered(r.key)
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
          {rateValue !== null && (
            <g>
              <line
                x1={rateX}
                x2={rateX}
                y1={AXIS_H - 2}
                y2={height - 4}
                stroke="var(--ink-2)"
                strokeWidth={1.5}
                strokeDasharray="5 4"
              />
              <text
                x={rateX + (rateX > width - 150 ? -6 : 6)}
                y={AXIS_H + 10}
                textAnchor={rateX > width - 150 ? 'end' : 'start'}
                className="bar-value"
                fill="var(--ink)"
              >
                {rateLabel}
              </text>
            </g>
          )}
        </svg>
        {tip && (
          <div className="viz-tooltip" style={{ left: tip.x + 14, top: tip.y - 8 }}>
            <div>
              <span className="tip-value">
                {tip.row.bitsPerSample.toFixed(3)} bits/sample · {tip.row.ratio.toFixed(2)}×
              </span>{' '}
              <span className="tip-label">{tip.row.name}</span>
            </div>
            <div className="tip-label">
              {tip.row.isBound ? 'equivalent to ' : ''}
              {tip.row.bytes.toLocaleString()} bytes · {tip.row.note}
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
            {rows.map(r => (
              <tr key={r.key}>
                <td>{r.name}</td>
                <td>{r.bytes.toLocaleString()}</td>
                <td>{r.bitsPerSample.toFixed(3)}</td>
                <td>{r.ratio.toFixed(3)}</td>
              </tr>
            ))}
            {rateBits !== null && rateBits > 0 && (
              <tr>
                <td>entropy rate R (Monte-Carlo)</td>
                <td>—</td>
                <td>{rateBits.toFixed(3)}</td>
                <td>{(16 / rateBits).toFixed(3)}</td>
              </tr>
            )}
          </tbody>
        </table>
      </details>
    </div>
  )
}
