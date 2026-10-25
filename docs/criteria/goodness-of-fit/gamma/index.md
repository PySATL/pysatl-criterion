# Gamma goodness-of-fit statistics

The module exposes **29 concrete classes**: 13 generated EDF criteria, five
generated frequency criteria, and 11 local criteria. Min–Toshiyuki and
Liao–Shimokawa share the same mathematical formula under different public names.

The null is Gamma with fixed shape `alfa > 0`, rate `beta > 0` and origin zero:

$$
f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x},\qquad x>0.
$$

Supply both parameters via `GammaDistributionDescriptor.DEFAULT.parse(...)`.
`beta` is a rate, not a scale. Estimating parameters from the tested sample changes
the null law and is outside these criteria's supported hypotheses.

## Catalog

| Criterion | Public class | Family |
|---|---|---|
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_gamma_criterion.md) | `KolmogorovSmirnovDPlusGammaGofStatistic` | edf |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_gamma_criterion.md) | `KolmogorovSmirnovDMinusGammaGofStatistic` | edf |
| [Kolmogorov–Smirnov](kolmogorov_smirnov_gamma_criterion.md) | `KolmogorovSmirnovGammaGofStatistic` | edf |
| [Anderson–Darling](anderson_darling_gamma_criterion.md) | `AndersonDarlingGammaGofStatistic` | edf |
| [Cramér–von Mises](cvm_gamma_criterion.md) | `CramerVonMisesGammaGofStatistic` | edf |
| [Watson](watson_gamma_criterion.md) | `WatsonGammaGofStatistic` | edf |
| [Kuiper](kuiper_gamma_criterion.md) | `KuiperGammaGofStatistic` | edf |
| [Scaled Kuiper](kuiper_scaled_gamma_criterion.md) | `KuiperScaledGammaGofStatistic` | edf |
| [Corrected Kuiper](kuiper_corrected_gamma_criterion.md) | `KuiperCorrectedGammaGofStatistic` | edf |
| [Liao–Shimokawa](liao_shimokawa_gamma_criterion.md) | `LiaoShimokawaGammaGofStatistic` | edf |
| [Zhang K](zhang_k_gamma_criterion.md) | `ZhangKGammaGofStatistic` | edf |
| [Zhang A](zhang_a_gamma_criterion.md) | `ZhangAGammaGofStatistic` | edf |
| [Zhang C](zhang_c_gamma_criterion.md) | `ZhangCGammaGofStatistic` | edf |
| [Greenwood](greenwood_gamma_criterion.md) | `GreenwoodGammaGofStatistic` | spacing |
| [Moran](moran_gamma_criterion.md) | `MoranGammaGofStatistic` | spacing |
| [Min–Toshiyuki (legacy name)](min_toshiyuki_gamma_criterion.md) | `MinToshiyukiGammaGofStatistic` | edf |
| [Pearson chi-squared](chi2_pearson_gamma_criterion.md) | `Chi2PearsonGammaGofStatistic` | frequency |
| [Freeman–Tukey](freeman_tukey_gamma_criterion.md) | `FreemanTukeyGammaGofStatistic` | frequency |
| [Modified log-likelihood](modified_log_likelihood_gamma_criterion.md) | `ModifiedLogLikelihoodGammaGofStatistic` | frequency |
| [Neyman modified chi-squared](neyman_chi2_gamma_criterion.md) | `NeymanChi2GammaGofStatistic` | frequency |
| [Cressie–Read](cressie_read_gamma_criterion.md) | `CressieReadGammaGofStatistic` | frequency |
| [Likelihood ratio](likelihood_ratio_gamma_criterion.md) | `LikelihoodRatioGammaGofStatistic` | likelihood |
| [Probability plot correlation](probability_plot_correlation_gamma_criterion.md) | `ProbabilityPlotCorrelationGammaGofStatistic` | ppcc |
| [Graph edge count](graph_edges_number_gamma_criterion.md) | `GraphEdgesNumberGammaGofStatistic` | graph |
| [Graph maximum degree](graph_max_degree_gamma_criterion.md) | `GraphMaxDegreeGammaGofStatistic` | graph |
| [Graph average degree](graph_average_degree_gamma_criterion.md) | `GraphAverageDegreeGammaGofStatistic` | graph |
| [Graph connected components](graph_connected_components_gamma_criterion.md) | `GraphConnectedComponentsGammaGofStatistic` | graph |
| [Graph clique number](graph_clique_number_gamma_criterion.md) | `GraphCliqueNumberGammaGofStatistic` | graph |
| [Graph independence number](graph_independence_number_gamma_criterion.md) | `GraphIndependenceNumberGammaGofStatistic` | graph |

## Calibration and limitations

For a specified continuous Gamma CDF, its probability transform is uniform.
EDF, spacing and CDF-graph null distributions therefore do not depend on Gamma
parameters in exact arithmetic. PPCC instead depends on shape and cannot detect
positive affine changes of the sample. Graph criteria use a local range/10 rule;
their formulas are verified, but a Gamma-specific published calibration was not
established. Simulate the actual statistic and sample size when calibrating them.

Generated frequency criteria accept `binning=EqualProbabilityBins(count=k)`;
Cressie–Read has fixed power 2/3. Only the local likelihood-ratio class retains
`bins=k` (default 8). Its storage lookup remains blocked. Generated frequency
classes serialize binning configuration for storage. Watson and Kuiper now use
short codes `WATSON` and `KUIPER`; old calibration identifiers must not be assumed
to refer to the new classes.

All classes accept finite nonnegative observations, including zero. Logarithmic
criteria can return infinity at boundaries or ties; PPCC rejects constant samples.
Log-CDF/log-survival evaluation reduces tail cancellation but does not provide
arbitrary precision: extreme Gamma probabilities may underflow and ordinary CDF
values may saturate at 0 or 1. No epsilon clipping changes the mathematical null.

See [mathematical verification](mathematical-validation.md) for evidence,
independent checks and limitations of this audit.
