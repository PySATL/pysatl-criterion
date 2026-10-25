# Exponentiated Weibull goodness-of-fit criteria

All **18 criteria** below test the same fully specified continuous null.

The null is fully specified:

$$
F_{a,k,s}(x)=\left[1-\exp\left(-(x/s)^k\right)\right]^a,\quad x\ge0,
\qquad a,k,s>0.
$$

Fix `exponent=a`, `shape=k`, `scale=s` in
`ExponentiatedWeibullDistributionDescriptor.DEFAULT.parse(...)`; location is zero.
All three parameters are required and are never estimated from the observations.
`execute_statistic` accepts a nonempty one-dimensional finite nonnegative sample;
ties and constant samples are allowed. It returns a scalar statistic, not a p-value,
and leaves the sample and parameters unchanged.

See [mathematical verification](mathematical-validation.md) for sources, independent numerical checks, corrections and limitations.

## EDF criteria (13)

| Criterion | Public class | Short code |
|---|---|---|
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_exponentiated_weibull_criterion.md) | `KolmogorovSmirnovDPlusExponentiatedWeibullGofStatistic` | `KS_PLUS` |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_exponentiated_weibull_criterion.md) | `KolmogorovSmirnovDMinusExponentiatedWeibullGofStatistic` | `KS_MINUS` |
| [Kolmogorov–Smirnov](kolmogorov_smirnov_exponentiated_weibull_criterion.md) | `KolmogorovSmirnovExponentiatedWeibullGofStatistic` | `KS` |
| [Anderson–Darling](anderson_darling_exponentiated_weibull_criterion.md) | `AndersonDarlingExponentiatedWeibullGofStatistic` | `AD` |
| [Cramér–von Mises](cvm_exponentiated_weibull_criterion.md) | `CrammerVonMisesExponentiatedWeibullGofStatistic` | `CVM` |
| [Liao–Shimokawa (historical Min–Toshiyuki)](min_toshiyuki_exponentiated_weibull_criterion.md) | `MinToshiyukiExponentiatedWeibullGofStatistic` | `MT` |
| [Watson](watson_exponentiated_weibull_criterion.md) | `WatsonExponentiatedWeibullGofStatistic` | `WATSON` |
| [Kuiper](kuiper_exponentiated_weibull_criterion.md) | `KuiperExponentiatedWeibullGofStatistic` | `KUIPER` |
| [Scaled Kuiper](kuiper_scaled_exponentiated_weibull_criterion.md) | `KuiperScaledExponentiatedWeibullGofStatistic` | `KUIPER_SCALED` |
| [Corrected Kuiper](kuiper_corrected_exponentiated_weibull_criterion.md) | `KuiperCorrectedExponentiatedWeibullGofStatistic` | `KUIPER_CORRECTED` |
| [Zhang Z_K](zhang_k_exponentiated_weibull_criterion.md) | `ZhangKExponentiatedWeibullGofStatistic` | `ZHANG_ZK` |
| [Zhang Z_A](zhang_a_exponentiated_weibull_criterion.md) | `ZhangAExponentiatedWeibullGofStatistic` | `ZHANG_ZA` |
| [Zhang Z_C](zhang_c_exponentiated_weibull_criterion.md) | `ZhangCExponentiatedWeibullGofStatistic` | `ZHANG_ZC` |

## Frequency criteria (5)

| Criterion | Public class | Short code |
|---|---|---|
| [Pearson chi-squared](chi2_pearson_exponentiated_weibull_criterion.md) | `Chi2PearsonExponentiatedWeibullGofStatistic` | `CHI2_PEARSON` |
| [Freeman–Tukey](freeman_tukey_exponentiated_weibull_criterion.md) | `FreemanTukeyExponentiatedWeibullGofStatistic` | `FREEMAN_TUKEY` |
| [Modified log-likelihood](modified_log_likelihood_exponentiated_weibull_criterion.md) | `ModifiedLogLikelihoodExponentiatedWeibullGofStatistic` | `MODIFIED_LOG_LIKELIHOOD` |
| [Neyman modified chi-squared](neyman_chi2_exponentiated_weibull_criterion.md) | `NeymanChi2ExponentiatedWeibullGofStatistic` | `CHI2_NEYMAN` |
| [Cressie–Read](cressie_read_exponentiated_weibull_criterion.md) | `CressieReadExponentiatedWeibullGofStatistic` | `CRESSIE_READ` |

All rejection regions are right-tailed, including D+ and D−. The three Kuiper normalizations have different critical values. `MinToshiyuki` is the historical API name for the Liao–Shimokawa expression.

Here $O_j$ and $E_j=np_j>0$ are observed and expected counts in exhaustive,
mutually exclusive cells, with $\sum_jO_j=\sum_jE_j=n$.
The constructor accepts keyword-only `binning`. The default is
`EqualProbabilityBins(PowerCount(2.0, 0.4), min_expected=None, max_bins=1000)`:
$B=\min(1000,\max(2,\lceil2n^{0.4}\rceil))$, and $p_j=1/B$.
Thus the default has two cells even for a singleton sample. The old
$\lceil\sqrt n\rceil$ rule and its one-cell singleton result no longer apply.

The example chooses three equal-probability cells explicitly. You can instead use
`ExplicitBins((0, b1, b2, float("inf")))` for fixed full-support edges; then
$p_j=F(b_{j+1})-F(b_j)$. See [binning strategies](../binning/index.md).

The full identifier appends `_EXPONENTIATED_WEIBULL_GOODNESS_OF_FIT` to the short code. Frequency `configuration()` includes the binning strategy. Historical square-root-bin calibrations and the old Watson `W` identifier are incompatible with the current catalog. Storage loading remains explicitly blocked by this family.

For EDF statistics and fixed equal-probability cells the continuous fixed null is distribution-free through the probability integral transform. Fitting any parameter from the same observations changes the null distribution and is outside this API contract.
