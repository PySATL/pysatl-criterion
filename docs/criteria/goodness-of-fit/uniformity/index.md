# Uniformity criteria

The criteria below test a fully specified uniform hypothesis. Fix both bounds
through `UniformDistributionDescriptor.DEFAULT.parse({"a": ..., "b": ...})`.

See [mathematical verification](mathematical-validation.md) for the formula checks,
scientific sources, and calibration qualifications.

## Empirical distribution function criteria

These criteria compare the empirical distribution function with the specified
uniform CDF. Their constructors take `parameters`; they do not estimate bounds.
`execute_statistic` returns a statistic, not a p-value. All use right-tail
rejection regions, including the directional D+ and D− statistics.

| Criterion | Public class | Short code |
|---|---|---|
| [Kolmogorov–Smirnov](kolmogorov_smirnov_uniform_criterion.md) | `KolmogorovSmirnovUniformGofStatistic` | `KS` |
| [Anderson–Darling](anderson_darling_uniform_criterion.md) | `AndersonDarlingUniformGofStatistic` | `AD` |
| [Cramér–von Mises](cvm_uniform_criterion.md) | `CrammerVonMisesUniformGofStatistic` | `CVM` |
| [Watson](watson_uniform_criterion.md) | `WatsonUniformGofStatistic` | `WATSON` |
| [Kuiper](kuiper_uniform_criterion.md) | `KuiperUniformGofStatistic` | `KUIPER` |
| [Kolmogorov–Smirnov D+ test](kolmogorov_smirnov_d_plus_uniform_criterion.md) | `KolmogorovSmirnovDPlusUniformGofStatistic` | `KS_PLUS` |
| [Kolmogorov–Smirnov D− test](kolmogorov_smirnov_d_minus_uniform_criterion.md) | `KolmogorovSmirnovDMinusUniformGofStatistic` | `KS_MINUS` |
| [Liao–Shimokawa test](liao_shimokawa_uniform_criterion.md) | `LiaoShimokawaUniformGofStatistic` | `LIAO_SHIMOKAWA` |
| [Scaled Kuiper test](kuiper_scaled_uniform_criterion.md) | `KuiperScaledUniformGofStatistic` | `KUIPER_SCALED` |
| [Corrected Kuiper test](kuiper_corrected_uniform_criterion.md) | `KuiperCorrectedUniformGofStatistic` | `KUIPER_CORRECTED` |
| [Zhang Z_K test](zhang_k_uniform_criterion.md) | `ZhangKUniformGofStatistic` | `ZHANG_ZK` |
| [Zhang Z_A test](zhang_a_uniform_criterion.md) | `ZhangAUniformGofStatistic` | `ZHANG_ZA` |
| [Zhang Z_C test](zhang_c_uniform_criterion.md) | `ZhangCUniformGofStatistic` | `ZHANG_ZC` |

Kuiper's unscaled, scaled and corrected classes return different normalizations;
use critical values for the selected class.

## Frequency criteria

All accept `parameters` and keyword-only `binning`. The default is
`EqualProbabilityBins(PowerCount(2.0, 0.4), min_expected=None, max_bins=1000)`.
See [binning](../binning/index.md) and [calibration storage](../binning/calibration.md).

| Criterion | Public class | Short code |
|---|---|---|
| [Pearson chi-squared](chi2_pearson_uniform_criterion.md) | `Chi2PearsonUniformGofStatistic` | `CHI2_PEARSON` |
| [Freeman–Tukey test](freeman_tukey_uniform_criterion.md) | `FreemanTukeyUniformGofStatistic` | `FREEMAN_TUKEY` |
| [Modified log-likelihood test](modified_log_likelihood_uniform_criterion.md) | `ModifiedLogLikelihoodUniformGofStatistic` | `MODIFIED_LOG_LIKELIHOOD` |
| [Neyman modified chi-squared test](neyman_chi2_uniform_criterion.md) | `NeymanChi2UniformGofStatistic` | `CHI2_NEYMAN` |
| [Cressie–Read test](cressie_read_uniform_criterion.md) | `CressieReadUniformGofStatistic` | `CRESSIE_READ` |

The full code appends `_UNIFORM_GOODNESS_OF_FIT` to the short code.
Binning settings belong to `configuration()` and are part of the storage identity.

## Spacing criteria

| Criterion | Public class |
|---|---|
| [Greenwood](greenwood_uniform_criterion.md) | `GreenwoodTestUniformGofStatistic` |
| [Sherman](sherman_uniform_criterion.md) | `ShermanUniformGofStatistic` |
| [Quesenberry–Miller](quesenberry_miller_uniform_criterion.md) | `QuesenberryMillerUniformGofStatistic` |

## Density and smooth criteria

| Criterion | Public class |
|---|---|
| [Bickel–Rosenblatt](bickel_rosenblatt_uniform_criterion.md) | `BickelRosenblattUniformGofStatistic` |
| [Neyman smooth](neyman_smooth_uniform_criterion.md) | `NeymanSmoothTestUniformGofStatistic` |

## Stein criteria

| Criterion | Public class |
|---|---|
| [Stein](stein_uniform_criterion.md) | `SteinUniformGofStatistic` |
| [Censored Stein](censored_stein_uniform_criterion.md) | `CensoredSteinUniformGofStatistic` |

The censored Stein statistic requires a censoring mechanism for calibration;
complete-data Monte Carlo is insufficient for censored observations.
