import { useMemo } from 'react'
import katex from 'katex'

function Tex({ tex, display }: { tex: string; display?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: !!display, throwOnError: false }),
    [tex, display],
  )
  return <span dangerouslySetInnerHTML={{ __html: html }} />
}

/** Where the entropy rate R comes from, in brief. */
export default function MethodNote() {
  return (
    <div className="math-section">
      <p>
        R is the entropy rate of the quantized process z — the bits per sample that no lossless
        code can beat. It has no usable closed form here, so it is estimated by an unbiased
        Monte-Carlo method from the companion{' '}
        <a href="https://github.com/concept-collection/timeseries-entropy">timeseries-entropy</a>{' '}
        package. The estimand is the conditional entropy of the next sample given a long past,
      </p>
      <Tex display tex="H\big(z_{M+1} \,\big|\, z_1, \dots, z_M\big) \;\searrow\; R \qquad (M \to \infty)," />
      <p>
        which reaches R once M exceeds the memory of the process. A past is drawn from the
        model, and the latent Gaussian input is Gibbs-sampled under the rounding constraints —
        the latents that generated the past are an exact draw from the conditional, so the chain
        starts in stationarity with no burn-in bias — emitting exact draws of z<sub>M+1</sub>.
        Rhee–Glynn randomized telescoping with antithetic half-block corrections then turns the
        plug-in entropies of that chain into an estimate whose expectation is exactly the
        conditional entropy, despite the finite-sample bias of every plug-in estimate and the
        autocorrelation of the Gibbs draws. Averaging over independent pasts gives R with an
        honest standard error.
      </p>
      <p>
        The dotted line on the chart is the package's analytic approximation, computed
        instantly from the filter and σ: Szegő's formula gives the error of linearly
        predicting the next sample from the past, with roundoff entering as a uniform noise
        floor of variance 1/12,
      </p>
      <Tex
        display
        tex="R \;\approx\; G(s_*), \qquad s_*^2 \;=\; \exp\!\Big(\int_0^1 \ln\big(\sigma^2\,|H(f)|^2 + \tfrac{1}{12}\big)\,df\Big) \;-\; \tfrac{1}{12},"
      />
      <p>
        where G(s) is the differential entropy of N(0, s²) + U(−½, ½) — exactly the entropy
        of a rounded Gaussian averaged over grid offsets. The floor keeps the integral finite
        where H(f) has zeros, and G saturates to zero at coarse quantization instead of
        diverging. Treating roundoff as independent dither and prediction as linear are
        approximations — the Monte-Carlo estimate is the exact check on them.
      </p>
      <p className="card-note">
        The estimate button under the Monte-Carlo readout runs exactly this method in a web worker — a
        TypeScript port of the package (src/entropy, hand-synced), one independent past at a
        time until stopped. The command line runs the Python original at the same settings for
        an independent check.
      </p>
    </div>
  )
}
