# Inverse-gamma goodness-of-fit criteria

All 19 public classes below test a **fully specified** inverse-gamma distribution.

For $x>0$ and $\alpha,\beta>0$,

$$
f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{-\alpha-1}e^{-\beta/x},
\qquad F(x)=Q(\alpha,\beta/x).
$$

Here $Q$ is the regularized upper incomplete gamma function; equivalently,
$\beta/X$ is Gamma(shape $\alpha$, scale 1). Location is fixed at zero.
Fix both parameters with
`InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 3.0, "beta": 2.0})`.
Omitting either parameter does not request estimation: these criteria reject
unsupported composite hypotheses.

All classes return a scalar and use a right-tail rejection region.
Input observations must be finite, positive and nonempty. Singletons, ties and
constant samples are accepted. See [mathematical verification](mathematical-validation.md).

## Criteria

| Criterion | Public class | Short code |
|---|---|---|
| [Kolmogorov–Smirnov](kolmogorov_smirnov_inverse_gamma_criterion.md) | `KolmogorovSmirnovInverseGammaGofStatistic` | `KS` |
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_inverse_gamma_criterion.md) | `KolmogorovSmirnovDPlusInverseGammaGofStatistic` | `KS_PLUS` |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_inverse_gamma_criterion.md) | `KolmogorovSmirnovDMinusInverseGammaGofStatistic` | `KS_MINUS` |
| [Anderson–Darling](anderson_darling_inverse_gamma_criterion.md) | `AndersonDarlingInverseGammaGofStatistic` | `AD` |
| [Cramér–von Mises](cramer_von_mises_inverse_gamma_criterion.md) | `CramerVonMisesInverseGammaGofStatistic` | `CVM` |
| [Watson](watson_inverse_gamma_criterion.md) | `WatsonInverseGammaGofStatistic` | `WAT` |
| [Kuiper](kuiper_inverse_gamma_criterion.md) | `KuiperInverseGammaGofStatistic` | `KUI_INVGAMMA` |
| [Scaled Kuiper](kuiper_scaled_inverse_gamma_criterion.md) | `KuiperScaledInverseGammaGofStatistic` | `KUIPER_SCALED` |
| [Corrected Kuiper](kuiper_corrected_inverse_gamma_criterion.md) | `KuiperCorrectedInverseGammaGofStatistic` | `KUIPER_CORRECTED` |
| [Liao–Shimokawa](liao_shimokawa_inverse_gamma_criterion.md) | `LiaoShimokawaInverseGammaGofStatistic` | `LIAO_SHIMOKAWA` |
| [Zhang A](zhang_a_inverse_gamma_criterion.md) | `ZhangAInverseGammaGofStatistic` | `ZAA` |
| [Zhang C](zhang_c_inverse_gamma_criterion.md) | `ZhangCInverseGammaGofStatistic` | `ZAC` |
| [Zhang K](zhang_k_inverse_gamma_criterion.md) | `ZhangKInverseGammaGofStatistic` | `ZAK` |
| [Greenwood](greenwood_inverse_gamma_criterion.md) | `GreenwoodInverseGammaGofStatistic` | `GRW` |
| [Pearson chi-squared](chi2_pearson_inverse_gamma_criterion.md) | `Chi2PearsonInverseGammaGofStatistic` | `CHI2_PEARSON` |
| [Freeman–Tukey](freeman_tukey_inverse_gamma_criterion.md) | `FreemanTukeyInverseGammaGofStatistic` | `FREEMAN_TUKEY` |
| [Modified log-likelihood](modified_log_likelihood_inverse_gamma_criterion.md) | `ModifiedLogLikelihoodInverseGammaGofStatistic` | `MODIFIED_LOG_LIKELIHOOD` |
| [Neyman modified chi-squared](neyman_chi2_inverse_gamma_criterion.md) | `NeymanChi2InverseGammaGofStatistic` | `CHI2_NEYMAN` |
| [Cressie–Read](cressie_read_inverse_gamma_criterion.md) | `CressieReadInverseGammaGofStatistic` | `CRESSIE_READ` |

The full code appends `_INV_GAMMA_GOODNESS_OF_FIT` to each short code.

## Calibration and binning

For fixed parameters, the probability integral transform gives iid Uniform(0,1)
observations under the null. Thus EDF and spacing null laws depend on sample size
and the chosen statistic, but not on alpha or beta. This conclusion requires
continuous iid data and parameters specified independently of the sample.
It does not justify using fitted-model tables or treating rounded data as continuous.

The five frequency classes accept keyword-only `binning`. The default is
`EqualProbabilityBins()` with `PowerCount(2.0, 0.4)`, no minimum expected count,
and at most 1000 bins. For a fixed count use `EqualProbabilityBins(8)`;
the former `bins=8` argument is no longer supported.
Equal-probability bin counts are multinomial under the specified null.
Other strategies can have parameter-dependent or data-dependent calibration;
see [binning](../binning/index.md) and [storage identity](../binning/calibration.md).

The built-in Monte Carlo resolver currently has no inverse-gamma generator.
External simulation can use `scipy.stats.invgamma.rvs(a=alpha, scale=beta, size=n)`;
repeat the same statistic and binning procedure for each replicate.
This audit validates calculations, not Monte Carlo p-values or nominal test size.

## Numerical behavior

Logarithmic EDF criteria use log-CDF/log-survival without clipping. A log-tail
underflow raises `FloatingPointError`; subsequent arithmetic can still overflow.
Probability-only formulas may see CDF rounding at zero and one. Frequency
validation converts numeric strings to floats; EDF/Greenwood reject such input.
Modified log-likelihood and Neyman return infinity for empty observed cells;
other frequency statistics implement their finite zero-count limits.

## Sources

The [SciPy inverse-gamma reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html)
fixes the shape/scale convention. Scientific sources for every statistic appear
on its individual page and in the [verification report](mathematical-validation.md).
