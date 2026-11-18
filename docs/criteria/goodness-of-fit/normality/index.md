# Normality statistics

The module `pysatl_criterion.statistics.goodness_of_fit.continuous.normal`
exports **60 concrete statistics**: 16 generated EDF statistics, 5 generated
frequency statistics and 39 explicit statistics, including six graph functionals.
Every entry below links to a separate page with its formula, runnable example,
scientific references and the qualifications on its interpretation.

Read the [mathematical verification report](mathematical-validation.md) before
reusing a published table or interpreting a legacy name as a certified test.

## Hypotheses and calibration

- **Fixed:** KS and CVM require `{'mean': mu, 'var': sigma_squared}`. They accept a
  singleton or a constant sample. Variance is not standard deviation.
- **Family:** 52 statistics take `{}`. For generated EDF/frequency statistics,
  mean and unbiased variance are fitted on every call. Other tests use the
  estimators stated on their individual pages. Calibrate by simulating N(0,1)
  and repeating the complete computation; fitted CDF values are dependent.
- **Graph:** six statistics take `{'var': sigma_squared}` with unknown mean.
  Their radius is translation invariant but not scale invariant. Simulate at
  the specified variance; a standard-normal table cannot be reused arbitrarily.

Lower-tail tests reject for small values, upper-tail tests for large values.
Two-tail statistics require both quantiles; they need not have a symmetric null
law. Directional KS deviations still have an upper-tail rejection rule.
None of these classes returns a p-value. Small samples, ties/rounding and
algorithm options must be reproduced in calibration. Default frequency bins
are equal-probability bins; user-specified boundaries can destroy scale invariance.

## Catalog

| Statistic page | Python class | Full code | Hypothesis | Rejection tail |
| --- | --- | --- | --- | --- |
| [BHS test for normality](bhs_normality_criterion.md) | `BHSNormalityGofStatistic` | `BHS_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Bonett-Seier test for normality](bonett_seier_normality_criterion.md) | `BonettSeierNormalityGofStatistic` | `BS_NORMALITY_GOODNESS_OF_FIT` | Family | Both |
| [Bontemps-Meddahi1 test for normality](bontemps_meddahi1_normality_criterion.md) | `BontempsMeddahi1NormalityGofStatistic` | `BM1_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Bontemps-Meddahi2 test for normality](bontemps_meddahi2_normality_criterion.md) | `BontempsMeddahi2NormalityGofStatistic` | `BM2_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Cabana-Cabana1 test for normality](cabana_cabana1_normality_criterion.md) | `CabanaCabana1NormalityGofStatistic` | `CC1_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Cabana-Cabana2 test for normality](cabana_cabana2_normality_criterion.md) | `CabanaCabana2NormalityGofStatistic` | `CC2_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Chen-Shapiro test for normality](chen_shapiro_normality_criterion.md) | `ChenShapiroNormalityGofStatistic` | `CS_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Coin test for normality](coin_normality_criterion.md) | `CoinNormalityGofStatistic` | `COIN_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [D’Agostino–Pearson omnibus test](dagostino_pearson_normality_criterion.md) | `DAPNormalityGofStatistic` | `DAP_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [D'Agostino test for normality](dagostino_normality_criterion.md) | `DagostinoNormalityGofStatistic` | `D_NORMALITY_GOODNESS_OF_FIT` | Family | Both |
| [Desgagne-Lafaye test for normality](desgagne_lafaye_normality_criterion.md) | `DesgagneLafayeNormalityGofStatistic` | `DLDMZEPD_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Doornik-Hansen test for normality](doornik_hansen_normality_criterion.md) | `DoornikHansenNormalityGofStatistic` | `DH_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Epps-Pulley test for normality](epps_pulley_normality_criterion.md) | `EppsPulleyNormalityGofStatistic` | `EP_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Filliben test for normality](filliben_normality_criterion.md) | `FilliNormalityGofStatistic` | `FILLI_NORMALITY_GOODNESS_OF_FIT` | Family | Lower |
| [GMG test for normality](gmg_normality_criterion.md) | `GMGNormalityGofStatistic` | `GMG_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Glen-Leemis-Barr test for normality](glen_leemis_barr_normality_criterion.md) | `GlenLeemisBarrNormalityGofStatistic` | `GLB_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Graph average degree test for normality](graph_average_degree_normality_criterion.md) | `GraphAverageDegreeNormalityGofStatistic` | `AVGDEGREE_GRAPH_NORMALITY_GOODNESS_OF_FIT` | Graph | Both |
| [Graph clique number test for normality](graph_clique_number_normality_criterion.md) | `GraphCliqueNumberNormalityGofStatistic` | `CLIQUENUMBER_GRAPH_NORMALITY_GOODNESS_OF_FIT` | Graph | Both |
| [Graph connected components test for normality](graph_connected_components_normality_criterion.md) | `GraphConnectedComponentsNormalityGofStatistic` | `CONNECTEDCOMPONENTS_GRAPH_NORMALITY_GOODNESS_OF_FIT` | Graph | Both |
| [Graph edges number test for normality](graph_edges_number_normality_criterion.md) | `GraphEdgesNumberNormalityGofStatistic` | `EDGESNUMBER_GRAPH_NORMALITY_GOODNESS_OF_FIT` | Graph | Both |
| [Graph independence number test for normality](graph_independence_number_normality_criterion.md) | `GraphIndependenceNumberNormalityGofStatistic` | `INDEPENDENCENUMBER_GRAPH_NORMALITY_GOODNESS_OF_FIT` | Graph | Both |
| [Graph max degree test for normality](graph_max_degree_normality_criterion.md) | `GraphMaxDegreeNormalityGofStatistic` | `MAXDEGREE_GRAPH_NORMALITY_GOODNESS_OF_FIT` | Graph | Both |
| [Hosking1 test for normality](hosking1_normality_criterion.md) | `Hosking1NormalityGofStatistic` | `HOSKING1_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Hosking2 test for normality](hosking2_normality_criterion.md) | `Hosking2NormalityGofStatistic` | `HOSKING2_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Hosking3 test for normality](hosking3_normality_criterion.md) | `Hosking3NormalityGofStatistic` | `HOSKING3_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Hosking4 test for normality](hosking4_normality_criterion.md) | `Hosking4NormalityGofStatistic` | `HOSKING4_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Jarque-Bera test for normality](jarque_bera_normality_criterion.md) | `JBNormalityGofStatistic` | `JB_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Kurtosis test for normality](kurtosis_normality_criterion.md) | `KurtosisNormalityGofStatistic` | `KURTOSIS_NORMALITY_GOODNESS_OF_FIT` | Family | Both |
| [Looney-Gulledge test for normality](looney_gulledge_normality_criterion.md) | `LooneyGulledgeNormalityGofStatistic` | `LG_NORMALITY_GOODNESS_OF_FIT` | Family | Lower |
| [Martinez-Iglewicz test for normality](martinez_iglewicz_normality_criterion.md) | `MartinezIglewiczNormalityGofStatistic` | `MI_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Robust Jarque-Bera test for normality](robust_jarque_bera_normality_criterion.md) | `RobustJarqueBeraNormalityGofStatistic` | `RJB_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Ryan-Joiner test for normality](ryan_joiner_normality_criterion.md) | `RyanJoinerNormalityGofStatistic` | `RJ_NORMALITY_GOODNESS_OF_FIT` | Family | Lower |
| [SF test for normality](sf_normality_criterion.md) | `SFNormalityGofStatistic` | `SF_NORMALITY_GOODNESS_OF_FIT` | Family | Lower |
| [SWRG test for normality](swrg_normality_criterion.md) | `SWRGNormalityGofStatistic` | `SWRG_NORMALITY_GOODNESS_OF_FIT` | Family | Lower |
| [Shapiro-Wilk test for normality](shapiro_wilk_normality_criterion.md) | `ShapiroWilkNormalityGofStatistic` | `SW_NORMALITY_GOODNESS_OF_FIT` | Family | Lower |
| [Skewness test for normality](skew_normality_criterion.md) | `SkewNormalityGofStatistic` | `SKEW_NORMALITY_GOODNESS_OF_FIT` | Family | Both |
| [Spiegelhalter test for normality](spiegelhalter_normality_criterion.md) | `SpiegelhalterNormalityGofStatistic` | `SH_NORMALITY_GOODNESS_OF_FIT` | Family | Both |
| [Zhang Q test for normality](zhang_q_normality_criterion.md) | `ZhangQNormalityGofStatistic` | `ZQ_NORMALITY_GOODNESS_OF_FIT` | Family | Both |
| [Zhang Q-star test for normality](zhang_q_star_normality_criterion.md) | `ZhangQStarNormalityGofStatistic` | `ZQS_NORMALITY_GOODNESS_OF_FIT` | Family | Both |
| [Zhang-Wu A test for normality](zhang_wu_a_normality_criterion.md) | `ZhangWuANormalityGofStatistic` | `ZWA_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Zhang-Wu C test for normality](zhang_wu_c_normality_criterion.md) | `ZhangWuCNormalityGofStatistic` | `ZWC_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Kolmogorov-Smirnov test for normality](kolmogorov_smirnov_normality_criterion.md) | `KolmogorovSmirnovNormalityGofStatistic` | `KS_NORMALITY_GOODNESS_OF_FIT` | Fixed | Upper |
| [Anderson-Darling test for normality](anderson_darling_normality_criterion.md) | `AndersonDarlingNormalityGofStatistic` | `AD_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Cramer-von Mises test for normality](cvm_normality_criterion.md) | `CramerVonMiseNormalityGofStatistic` | `CVM_GOODNESS_OF_FIT` | Fixed | Upper |
| [Lilliefors test for normality](lilliefors_normality_criterion.md) | `LillieforsNormalityGofStatistic` | `LILLIE_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Kolmogorov–Smirnov D+ for normality](kolmogorov_smirnov_d_plus_normality_criterion.md) | `KolmogorovSmirnovDPlusNormalityGofStatistic` | `KS_PLUS_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Kolmogorov–Smirnov D− for normality](kolmogorov_smirnov_d_minus_normality_criterion.md) | `KolmogorovSmirnovDMinusNormalityGofStatistic` | `KS_MINUS_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Liao–Shimokawa for normality](liao_shimokawa_normality_criterion.md) | `LiaoShimokawaNormalityGofStatistic` | `LIAO_SHIMOKAWA_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Watson for normality](watson_normality_criterion.md) | `WatsonNormalityGofStatistic` | `WATSON_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Kuiper for normality](kuiper_normality_criterion.md) | `KuiperNormalityGofStatistic` | `KUIPER_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Scaled Kuiper for normality](kuiper_scaled_normality_criterion.md) | `KuiperScaledNormalityGofStatistic` | `KUIPER_SCALED_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Corrected Kuiper for normality](kuiper_corrected_normality_criterion.md) | `KuiperCorrectedNormalityGofStatistic` | `KUIPER_CORRECTED_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Zhang K for normality](zhang_k_normality_criterion.md) | `ZhangKNormalityGofStatistic` | `ZHANG_ZK_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Zhang A for normality](zhang_a_normality_criterion.md) | `ZhangANormalityGofStatistic` | `ZHANG_ZA_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Zhang C for normality](zhang_c_normality_criterion.md) | `ZhangCNormalityGofStatistic` | `ZHANG_ZC_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Pearson chi-squared for normality](chi2_pearson_normality_criterion.md) | `Chi2PearsonNormalityGofStatistic` | `CHI2_PEARSON_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Freeman–Tukey for normality](freeman_tukey_normality_criterion.md) | `FreemanTukeyNormalityGofStatistic` | `FREEMAN_TUKEY_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Modified log likelihood for normality](modified_log_likelihood_normality_criterion.md) | `ModifiedLogLikelihoodNormalityGofStatistic` | `MODIFIED_LOG_LIKELIHOOD_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Neyman chi-squared for normality](neyman_chi2_normality_criterion.md) | `NeymanChi2NormalityGofStatistic` | `CHI2_NEYMAN_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |
| [Cressie–Read for normality](cressie_read_normality_criterion.md) | `CressieReadNormalityGofStatistic` | `CRESSIE_READ_NORMALITY_GOODNESS_OF_FIT` | Family | Upper |

## Example

```python
import numpy as np
from pysatl_criterion.distribution import NormalDistributionDescriptor as Normal
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import (
    KolmogorovSmirnovNormalityGofStatistic,
    ShapiroWilkNormalityGofStatistic,
)

sample = np.random.default_rng(42).normal(size=24)
fixed = KolmogorovSmirnovNormalityGofStatistic(Normal.DEFAULT.parse({"mean": 0, "var": 1}))
family = ShapiroWilkNormalityGofStatistic(Normal.DEFAULT.parse({}))
assert np.isfinite(fixed.execute_statistic(sample))
assert 0 <= family.execute_statistic(sample) <= 1
```

## Scientific provenance

The report distinguishes published formulas, finite-sample approximations and
local definitions. Hosking's L-moment theory does not establish the local
piecewise calibration constants; graph literature does not establish the exact
six graph normality tests; the exact fitted GLB attribution remains unconfirmed.
The pages link the relevant literature with these limits stated explicitly.
