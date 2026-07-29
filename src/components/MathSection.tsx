import { useMemo } from 'react'
import katex from 'katex'

function Display({ tex }: { tex: string }) {
  const html = useMemo(() => katex.renderToString(tex, { displayMode: true, throwOnError: false }), [tex])
  return <div dangerouslySetInnerHTML={{ __html: html }} />
}

/**
 * Stub for the full derivation. It states the model and the formula the app
 * plots; the reasoning connecting them is to come.
 */
export default function MathSection() {
  return (
    <div className="math-section">
      <p>
        The generating model: i.i.d. Gaussian noise, a FIR filter h, optional uniform dither, and
        rounding to the integer quantization grid (the step is the unit, so σ is measured in
        steps):
      </p>
      <Display tex="x_n \sim \mathcal{N}(0,\sigma^2)\ \text{i.i.d.}, \qquad y = h * x, \qquad z_n = \operatorname{round}(y_n + d_n), \quad d_n \sim \mathcal{U}[-\tfrac12,\tfrac12)\ \text{or}\ 0" />
      <p>
        The dashed reference line is the entropy rate of the stationary Gaussian process y,
        quantized at unit step, in the fine-quantization (high-resolution) limit — the ideal
        lossless rate in bits per sample:
      </p>
      <Display tex="R \;=\; \tfrac12\log_2(2\pi e)\;+\;\int_0^{1/2} \log_2 S(f)\,df, \qquad S(f) = \sigma^2\,|H(f)|^2" />
      <p>
        with f in cycles per sample. With no filter this reduces to ½ log₂(2πe σ²). The formula
        holds when S(f) is well above one step² across the band; where the response dips toward or
        below the quantization step — deep stopbands, small σ — the true entropy rate is larger
        than R (and R can even go negative), and no fixed-order predictor fully whitens the
        process. Quantifying that gap, the effect of dither, and why LPC + ANS is the right
        yardstick is the subject of the full derivation, still to be written.
      </p>
    </div>
  )
}
