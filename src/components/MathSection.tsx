import { useMemo } from 'react'
import katex from 'katex'

function Tex({ tex, display }: { tex: string; display?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: !!display, throwOnError: false }),
    [tex, display],
  )
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

/** One term of the formula: the symbol, then what it is. */
function Def({ tex, children }: { tex: string; children: React.ReactNode }) {
  return (
    <>
      <dt>
        <Tex tex={tex} />
      </dt>
      <dd>{children}</dd>
    </>
  )
}

/**
 * The theoretical rate R exactly as `model/theory.ts` computes it, with every
 * symbol defined. The derivation that justifies it is still to be written.
 */
export default function MathSection() {
  return (
    <div className="math-section">
      <p>
        The dashed line on the compression chart is R, the predicted bits per sample. It is
        computed in three steps: the spectrum of the stored signal, the residual an ideal
        predictor leaves, and the entropy of that residual on the integer grid.
      </p>

      <Tex display tex="S_z(f) \;=\; \sigma^2\,\big|H(f)\big|^2 \;+\; \sigma_q^2, \qquad H(f) \;=\; \sum_{n=0}^{L-1} h_n\, e^{-2\pi i f n}" />
      <Tex display tex="\sigma_e \;=\; 2^{\,\int_0^{1/2} \log_2 S_z(f)\,df}" />
      <Tex display tex="R \;=\; -\sum_{z \in \mathbb{Z}} p_z \log_2 p_z, \qquad p_z \;=\; \Phi\!\left(\frac{z + \tfrac12}{\sigma_e}\right) - \Phi\!\left(\frac{z - \tfrac12}{\sigma_e}\right)" />

      <dl className="defs">
        <Def tex="\sigma">
          standard deviation of the i.i.d. Gaussian input, in quantization steps (the step is the
          unit, so rounding is to the nearest integer)
        </Def>
        <Def tex="h_0,\dots,h_{L-1}">
          the FIR kernel the input is convolved with — the taps drawn in the kernel plot, L of
          them
        </Def>
        <Def tex="H(f)">
          the kernel's frequency response, the quantity plotted in dB as |H(f)|
        </Def>
        <Def tex="f">
          frequency in cycles per sample, running from 0 to ½ (Nyquist); the plots label the same
          axis in Hz, as f times the sample rate
        </Def>
        <Def tex="\sigma_q^2">
          variance charged to rounding, treated as additive white noise: 1/12 for the roundoff
          alone, 1/6 when dither is on (the dither is stored in the integers, so its 1/12 adds)
        </Def>
        <Def tex="S_z(f)">
          power spectrum of the stored integer signal, in steps² per unit frequency
        </Def>
        <Def tex="\sigma_e">
          standard deviation of the innovation — what an ideal linear predictor still cannot
          predict from all earlier samples. The exponent is the Szegő–Kolmogorov formula for the
          one-step prediction error, the geometric mean of the spectrum.
        </Def>
        <Def tex="\Phi">standard normal cumulative distribution function</Def>
        <Def tex="p_z">
          probability that the innovation, rounded to the integer grid, lands on z
        </Def>
        <Def tex="R">
          bits per sample; the compression ratio the chart marks is 16/R, against 16-bit integer
          storage
        </Def>
      </dl>

      <p className="card-note">
        The integral is evaluated by the midpoint rule on 8192 points and the sum over z is taken
        out to where the remaining mass is negligible. A derivation — and an account of where
        modeling the roundoff as white noise stops being fair — is still to be written.
      </p>
    </div>
  )
}
