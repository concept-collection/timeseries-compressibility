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
        The reference rate R treats the roundoff as an additive white noise floor on the spectrum
        — σ<sub>q</sub>² = 1/12 without dither, 1/6 with it (the dither is stored in the
        integers) — takes the one-step Wiener prediction error of the resulting process
        (Szegő–Kolmogorov), and charges the exact entropy of that innovation quantized at unit
        step:
      </p>
      <Display tex="S_z(f) = \sigma^2\,|H(f)|^2 + \sigma_q^2, \qquad \sigma_e^2 = \exp\!\Big(2\!\int_0^{1/2}\!\ln S_z(f)\,df\Big), \qquad R = H_{\Delta}(\sigma_e)" />
      <Display tex="H_{\Delta}(s) = -\sum_{z\in\mathbb{Z}} p_z \log_2 p_z, \qquad p_z = \Phi\!\Big(\tfrac{z+\frac12}{s}\Big) - \Phi\!\Big(\tfrac{z-\frac12}{s}\Big)" />
      <p>
        with f in cycles per sample. In the fine-quantization regime (S ≫ 1 everywhere) this
        reduces to the classical Gaussian entropy rate ½ log₂(2πe) + ∫ log₂ S(f) df — and with
        no filter, to ½ log₂(2πe σ²). The noise floor keeps R finite and positive where a deep
        stopband pushes S(f) below one step², which is where the classical formula diverges to
        −∞. It is still an approximation: roundoff is not truly white, independent, or Gaussian,
        prediction is from the quantized past, and everything degrades when the whole signal
        hides inside the dead zone (σ_y ≪ 1). Quantifying that gap — and why LPC + ANS is the
        right yardstick — is the subject of the full derivation, still to be written.
      </p>
    </div>
  )
}
