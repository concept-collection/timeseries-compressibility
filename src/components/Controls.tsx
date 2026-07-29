import { FAMILY_LABELS, type FilterFamily, type FilterSpec } from '../model/filters'

export interface ControlsProps {
  sigma: number
  setSigma: (v: number) => void
  sampleRateHz: number
  setSampleRateHz: (v: number) => void
  spec: FilterSpec
  setSpec: (v: FilterSpec) => void
  dither: boolean
  setDither: (v: boolean) => void
}

function formatHz(hz: number): string {
  return hz >= 1000 ? `${(hz / 1000).toFixed(hz >= 10000 ? 0 : 1)} kHz` : `${Math.round(hz)} Hz`
}

/** A slider that is linear in log10 of its value. */
function LogSlider(props: {
  label: string
  value: number
  min: number
  max: number
  display: string
  onChange: (v: number) => void
}) {
  const lo = Math.log10(props.min)
  const hi = Math.log10(props.max)
  return (
    <div className="control">
      <label>
        {props.label} <span className="value">{props.display}</span>
      </label>
      <input
        type="range"
        min={lo}
        max={hi}
        step={(hi - lo) / 400}
        value={Math.log10(props.value)}
        onChange={e => props.onChange(10 ** Number(e.target.value))}
      />
    </div>
  )
}

export default function Controls(p: ControlsProps) {
  const { spec } = p
  const nyquist = p.sampleRateHz / 2
  const sinc = spec.family === 'lowpass' || spec.family === 'bandpass'
  return (
    <div className="controls">
      <LogSlider
        label="σ (quantization steps)"
        value={p.sigma}
        min={0.1}
        max={100}
        display={p.sigma.toPrecision(3)}
        onChange={v => p.setSigma(Number(v.toPrecision(3)))}
      />
      <div className="control">
        <label>filter</label>
        <select
          value={spec.family}
          onChange={e => p.setSpec({ ...spec, family: e.target.value as FilterFamily })}
        >
          {(Object.keys(FAMILY_LABELS) as FilterFamily[]).map(f => (
            <option key={f} value={f}>
              {FAMILY_LABELS[f]}
            </option>
          ))}
        </select>
      </div>
      {spec.family === 'bandpass' && (
        <LogSlider
          label="low cutoff"
          value={spec.lowHz}
          min={10}
          max={nyquist * 0.9}
          display={formatHz(spec.lowHz)}
          onChange={v => p.setSpec({ ...spec, lowHz: Math.round(v) })}
        />
      )}
      {sinc && (
        <LogSlider
          label={spec.family === 'bandpass' ? 'high cutoff' : 'cutoff'}
          value={spec.highHz}
          min={20}
          max={nyquist * 0.98}
          display={formatHz(spec.highHz)}
          onChange={v => p.setSpec({ ...spec, highHz: Math.round(v) })}
        />
      )}
      {sinc && (
        <div className="control">
          <label>
            kernel length <span className="value">{spec.taps} taps</span>
          </label>
          <input
            type="range"
            min={15}
            max={201}
            step={2}
            value={spec.taps}
            onChange={e => p.setSpec({ ...spec, taps: Number(e.target.value) })}
          />
        </div>
      )}
      {spec.family === 'movingAverage' && (
        <div className="control">
          <label>
            width <span className="value">{spec.width} samples</span>
          </label>
          <input
            type="range"
            min={2}
            max={64}
            value={spec.width}
            onChange={e => p.setSpec({ ...spec, width: Number(e.target.value) })}
          />
        </div>
      )}
      <div className="control">
        <label>sample rate</label>
        <select value={p.sampleRateHz} onChange={e => p.setSampleRateHz(Number(e.target.value))}>
          {[1000, 8000, 16000, 30000, 44100, 96000].map(r => (
            <option key={r} value={r}>
              {formatHz(r)}
            </option>
          ))}
        </select>
      </div>
      <div className="control control-toggle">
        <input
          id="dither"
          type="checkbox"
          checked={p.dither}
          onChange={e => p.setDither(e.target.checked)}
        />
        <label htmlFor="dither">dither (uniform ±½ before rounding)</label>
      </div>
    </div>
  )
}
