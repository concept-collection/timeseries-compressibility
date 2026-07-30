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
 * symbol defined, followed by a sketch of the derivation: the dither identity,
 * the per-mode count with the noise at full variance, and the subadditivity
 * ceiling that takes over below threshold.
 */
export default function MathSection() {
  return (
    <div className="math-section">
      <p>
        The dashed line on the compression chart is R, the predicted bits per sample — the
        smaller of a spectral estimate and a rigorous one-sample ceiling:
      </p>

      <Tex display tex="R \;=\; \min\big(R_{\mathrm{spec}},\, R_{\mathrm{samp}}\big)" />
      <Tex display tex="R_{\mathrm{spec}} \;=\; \int_0^1 \tfrac{1}{2}\log_2\!\big(2\pi e\,(S(f) + \nu)\big)\, df, \qquad S(f) \;=\; \sigma^2\,\big|H(f)\big|^2" />
      <Tex display tex="R_{\mathrm{samp}} \;=\; H\big(\operatorname{round}(\mathcal N(0, v) \,[+\, U(-\tfrac12,\tfrac12)\ \text{with dither}])\big), \qquad v \;=\; \sigma^2 \textstyle\sum_m h_m^2" />

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
          the kernel's frequency response, the quantity plotted in dB as |H(f)|; f is in cycles
          per sample, symmetric about ½ (Nyquist), and the plots label the same axis in Hz
        </Def>
        <Def tex="S(f)">
          power spectrum of the filtered signal alone, in steps² per unit frequency — the dither
          is <em>not</em> folded in here
        </Def>
        <Def tex="\nu">
          variance charged to the rounding: 1/12 for the roundoff, 1/6 with dither (the dither's
          own 1/12 is stored in the integers and adds)
        </Def>
        <Def tex="v">
          variance of a single output sample, the integral of S(f)
        </Def>
        <Def tex="R">
          bits per sample; the compression ratio the chart marks is 16/R, against 16-bit integer
          storage
        </Def>
      </dl>

      <h3>The spectral branch</h3>
      <p>
        The <em>dither identity</em> starts it off: for z = round(y) and u an independent
        uniform on [-½, ½)<sup>N</sup>, the discrete entropy of z equals the differential
        entropy of z + u, exactly. When the process is live on the unit-cell scale (v ≳ ¼),
        z + u has nearly the law of y + u, so R is the entropy rate of the signal plus a white
        unit-cell noise — the Zamir–Feder universal-quantization rate; with physical dither the
        smoothing noise is d + u and ν doubles to 1/6.
      </p>
      <p>
        Counting that entropy per Fourier mode, the signal modes are independent Gaussians of
        variance S(f), and each mode of the i.i.d. cube noise mixes all N samples'
        contributions — so the central limit theorem Gaussianizes it, and it enters at its{' '}
        <em>full variance</em> ν. It does not enter at the entropy power 1/(2πe) that a scalar
        quantizer aligned with the mode would charge: the quantization lattice lives in the
        sample basis, and only for the trivial kernel do modes and quantizers align. (An earlier
        version of this app charged entropy power — additive constant 1 instead of 2πe·ν ≈ 1.42
        — and systematically underestimated the measured rate by up to ~0.23 bits/sample. Its
        "exact per-mode" refinement was worse still: it modeled the wrong physics more
        faithfully.) One consequence worth naming: a dead band inside a live process contributes
        ½log₂(2πe/12) ≈ 0.25 bits per mode, not zero.
      </p>

      <h3>The sub-threshold ceiling</h3>
      <p>
        When v ≪ 1 nearly every sample rounds to zero and the true rate collapses
        exponentially, while R<sub>spec</sub> bottoms out at ½log₂(2πe ν) &gt; 0. Subadditivity
        rescues the estimate rigorously: H(z) ≤ Σ<sub>n</sub> H(z<sub>n</sub>), and each stored
        sample is exactly round(N(0, v)) — plus the uniform dither first when it is on — so
        R<sub>samp</sub> is a true upper bound on the rate with exactly the right collapse. The
        min selects it precisely where the spectral branch fails.
      </p>

      <h3>Checks and accuracy</h3>
      <p>
        For the identity kernel the two branches agree with the exact i.i.d. entropy at every σ
        (both carry the Fisher correction log₂e/(24σ²) at large σ; below one step the min
        switches to the exact R<sub>samp</sub>). At high SNR, R<sub>spec</sub> →
        ½log₂(2πe σ²) + ∫log₂|H| df — the Kolmogorov formula. What the spectral branch ignores
        is the cross-mode dependence of the cube noise, at most ½log₂(2πe/12) ≈ 0.2546
        bits/sample and recoverable only when nearly the whole spectrum is noise-dominated;
        Monte-Carlo puts the estimate within ~0.01–0.02 bits/sample for v ≳ 0.25, with the
        worst observed error ~+0.03 near the crossover between branches, slightly positive
        everywhere — as befits a formula whose sample branch is a genuine bound.
      </p>

      <p className="card-note">
        The integral is evaluated by the midpoint rule on 8192 points over [0, ½] (symmetry
        supplies the other half). R<sub>samp</sub> sums the exact bin probabilities of the
        rounded Gaussian — integrated against the triangular dither-overlap window when dither
        is on. The Monte-Carlo command under the chart estimates the true entropy rate of the
        same process, for checking R where the approximations are in doubt.
      </p>
    </div>
  )
}
