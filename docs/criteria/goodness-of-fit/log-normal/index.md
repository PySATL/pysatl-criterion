# Log-normal goodness-of-fit statistics

All **55 concrete statistics** exported by `continuous.log_normal` are listed
below. There are 15 EDF variants, 5 frequency statistics, one spacing statistic,
32 logarithmic normality wrappers and two KL functionals. The abstract base is
not itself a test. Every row links to a formula, executable example and scientific
sources; the [mathematical verification](mathematical-validation.md) distinguishes
verified definitions from unresolved historical attribution.

## Choosing the hypothesis

- **Fixed:** `LogNormal.DEFAULT.parse({"mu": 0.4, "s": 0.8})` or
  `LogNormal.SHAPE_SCALE.parse({"s": 0.8, "scale": exp(0.4)})`.
  These 21 criteria test a specified CDF; `s` is log standard deviation,
  `mu` is log mean and `scale` is the median. The mean of X is exp(mu+s²/2).
- **Family:** `LogNormal.DEFAULT.parse({})`. The 34 shape criteria remove
  log-location and log-scale. Fully fixed schemas remain accepted for API
  compatibility, but their values are ignored by the shape statistic. These
  criteria cannot distinguish log-normal laws with different parameters.
- Omitted parameters are never silently filled into a hypothesis. Both coordinate
  schemas are supported, partial schemas are rejected, and `hypothesis()`
  preserves the input. Family calibration explicitly chooses mu=0,s=1 for
  sampling; it does not change the hypothesis.

All inputs must be finite, real, one-dimensional and unmasked. Logarithmic
statistics additionally require strictly positive observations and sufficient
nonconstant log data; individual pages state their minimum sizes. EDF statistics
may assign a finite discrepancy to out-of-support data. Frequency zero-count
limits may legitimately be infinite. Undefined input raises `ValueError`.

```python
import numpy as np
from pysatl_criterion.distribution import LogNormalDistributionDescriptor as LogNormal
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_normal import (
    KolmogorovSmirnovLogNormalGofStatistic, ShapiroWilkLogNormalGofStatistic,
)
from pysatl_criterion.utils.generator import get_hypothesis_generator

sample = np.exp([-1.3, -0.8, -0.4, -0.1, 0.1, 0.3, 0.7, 0.9, 1.4, 2.0])
fixed = KolmogorovSmirnovLogNormalGofStatistic(LogNormal.DEFAULT.parse({"mu": 0, "s": 1}))
family = ShapiroWilkLogNormalGofStatistic(LogNormal.DEFAULT.parse({}))
assert np.isfinite(fixed.execute_statistic(sample))
assert np.isfinite(family.execute_statistic(sample))
assert family.hypothesis().parameters() == {}
assert get_hypothesis_generator(family).parameters() == {"mu": 0, "s": 1}
```

## Complete catalog

| Class | Full code | Hypothesis | Rejection tail | Page |
| --- | --- | --- | --- | --- |
| `AndersonDarlingLogNormalGofStatistic` | `AD_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Anderson-Darling test for log-normal data](anderson_darling_log_normal_criterion.md) |
| `BonettSeierLogNormalGofStatistic` | `BS_LOGNORMAL_GOODNESS_OF_FIT` | Family | two_tailed | [Bonett-Seier test for log-normal data](bonett_seier_log_normal_criterion.md) |
| `BontempsMeddahi1LogNormalGofStatistic` | `BM1_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Bontemps-Meddahi1 test for log-normal data](bontemps_meddahi1_log_normal_criterion.md) |
| `BontempsMeddahi2LogNormalGofStatistic` | `BM2_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Bontemps-Meddahi2 test for log-normal data](bontemps_meddahi2_log_normal_criterion.md) |
| `CabanaCabana1LogNormalGofStatistic` | `CC1_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Cabana-Cabana1 test for log-normal data](cabana_cabana1_log_normal_criterion.md) |
| `CabanaCabana2LogNormalGofStatistic` | `CC2_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Cabana-Cabana2 test for log-normal data](cabana_cabana2_log_normal_criterion.md) |
| `ChenShapiroLogNormalGofStatistic` | `CS_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Chen-Shapiro test for log-normal data](chen_shapiro_log_normal_criterion.md) |
| `Chi2PearsonLogNormalGofStatistic` | `CHI2_PEARSON_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Pearson chi-squared for log-normal data](chi2_pearson_log_normal_criterion.md) |
| `CoinLogNormalGofStatistic` | `COIN_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Coin test for log-normal data](coin_log_normal_criterion.md) |
| `CramerVonMiseLogNormalGofStatistic` | `CVM_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Cramer-von Mises test for log-normal data](cvm_log_normal_criterion.md) |
| `CressieReadLogNormalGofStatistic` | `CRESSIE_READ_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Cressie–Read for log-normal data](cressie_read_log_normal_criterion.md) |
| `DAPLogNormalGofStatistic` | `DAP_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [D’Agostino–Pearson omnibus test](dagostino_pearson_log_normal_criterion.md) |
| `DagostinoLogNormalGofStatistic` | `D_LOGNORMAL_GOODNESS_OF_FIT` | Family | two_tailed | [D'Agostino test for log-normal data](dagostino_log_normal_criterion.md) |
| `DesgagneLafayeLogNormalGofStatistic` | `DLDMZEPD_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Desgagne-Lafaye test for log-normal data](desgagne_lafaye_log_normal_criterion.md) |
| `DoornikHansenLogNormalGofStatistic` | `DH_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Doornik-Hansen test for log-normal data](doornik_hansen_log_normal_criterion.md) |
| `EppsPulleyLogNormalGofStatistic` | `EP_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Epps-Pulley test for log-normal data](epps_pulley_log_normal_criterion.md) |
| `FilliLogNormalGofStatistic` | `FILLI_LOGNORMAL_GOODNESS_OF_FIT` | Family | left | [Filliben test for log-normal data](filliben_log_normal_criterion.md) |
| `FreemanTukeyLogNormalGofStatistic` | `FREEMAN_TUKEY_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Freeman–Tukey for log-normal data](freeman_tukey_log_normal_criterion.md) |
| `GMGLogNormalGofStatistic` | `GMG_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [GMG test for log-normal data](gmg_log_normal_criterion.md) |
| `GlenLeemisBarrLogNormalGofStatistic` | `GLB_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Glen-Leemis-Barr test for log-normal data](glen_leemis_barr_log_normal_criterion.md) |
| `Hosking1LogNormalGofStatistic` | `HOSKING1_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Hosking1 test for log-normal data](hosking1_log_normal_criterion.md) |
| `Hosking2LogNormalGofStatistic` | `HOSKING2_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Hosking2 test for log-normal data](hosking2_log_normal_criterion.md) |
| `Hosking3LogNormalGofStatistic` | `HOSKING3_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Hosking3 test for log-normal data](hosking3_log_normal_criterion.md) |
| `Hosking4LogNormalGofStatistic` | `HOSKING4_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Hosking4 test for log-normal data](hosking4_log_normal_criterion.md) |
| `JBLogNormalGofStatistic` | `JB_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Jarque-Bera test for log-normal data](jarque_bera_log_normal_criterion.md) |
| `KLIntegralLogNormalGoFStatistic` | `KL_INT_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [KL integral test for log-normal distribution](kl_integral_log_normal_criterion.md) |
| `KLSupremumLogNormalGoFStatistic` | `KL_SUP_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [KL supremum test for log-normal distribution](kl_supremum_log_normal_criterion.md) |
| `KolmogorovSmirnovDMinusLogNormalGofStatistic` | `KS_MINUS_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Kolmogorov–Smirnov D− for log-normal data](kolmogorov_smirnov_d_minus_log_normal_criterion.md) |
| `KolmogorovSmirnovDPlusLogNormalGofStatistic` | `KS_PLUS_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Kolmogorov–Smirnov D+ for log-normal data](kolmogorov_smirnov_d_plus_log_normal_criterion.md) |
| `KolmogorovSmirnovLogNormalGofStatistic` | `KS_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Kolmogorov-Smirnov test for log-normal data](kolmogorov_smirnov_log_normal_criterion.md) |
| `KuiperCorrectedLogNormalGofStatistic` | `KUIPER_CORRECTED_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Corrected Kuiper for log-normal data](kuiper_corrected_log_normal_criterion.md) |
| `KuiperLogNormalGofStatistic` | `KUIPER_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Kuiper for log-normal data](kuiper_log_normal_criterion.md) |
| `KuiperScaledLogNormalGofStatistic` | `KUIPER_SCALED_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Scaled Kuiper for log-normal data](kuiper_scaled_log_normal_criterion.md) |
| `KurtosisLogNormalGofStatistic` | `KURTOSIS_LOGNORMAL_GOODNESS_OF_FIT` | Family | two_tailed | [Kurtosis test for log-normal data](kurtosis_log_normal_criterion.md) |
| `LiaoShimokawaLogNormalGofStatistic` | `LIAO_SHIMOKAWA_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Liao–Shimokawa for log-normal data](liao_shimokawa_log_normal_criterion.md) |
| `LooneyGulledgeLogNormalGofStatistic` | `LG_LOGNORMAL_GOODNESS_OF_FIT` | Family | left | [Looney-Gulledge test for log-normal data](looney_gulledge_log_normal_criterion.md) |
| `MartinezIglewiczLogNormalGofStatistic` | `MI_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Martinez-Iglewicz test for log-normal data](martinez_iglewicz_log_normal_criterion.md) |
| `ModifiedLogLikelihoodLogNormalGofStatistic` | `MODIFIED_LOG_LIKELIHOOD_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Modified log likelihood for log-normal data](modified_log_likelihood_log_normal_criterion.md) |
| `NeymanChi2LogNormalGofStatistic` | `CHI2_NEYMAN_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Neyman chi-squared for log-normal data](neyman_chi2_log_normal_criterion.md) |
| `QuesenberryMillerLogNormalGofStatistic` | `QUESENBERRY_MILLER_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Quesenberry-Miller test for log-normal distribution](quesenberry_miller_log_normal_criterion.md) |
| `RobustJarqueBeraLogNormalGofStatistic` | `RJB_LOGNORMAL_GOODNESS_OF_FIT` | Family | right | [Robust Jarque-Bera test for log-normal data](robust_jarque_bera_log_normal_criterion.md) |
| `RyanJoinerLogNormalGofStatistic` | `RJ_LOGNORMAL_GOODNESS_OF_FIT` | Family | left | [Ryan-Joiner test for log-normal data](ryan_joiner_log_normal_criterion.md) |
| `SFLogNormalGofStatistic` | `SF_LOGNORMAL_GOODNESS_OF_FIT` | Family | left | [SF test for log-normal data](sf_log_normal_criterion.md) |
| `SWRGLogNormalGofStatistic` | `SWRG_LOGNORMAL_GOODNESS_OF_FIT` | Family | left | [SWRG test for log-normal data](swrg_log_normal_criterion.md) |
| `ShapiroWilkLogNormalGofStatistic` | `SW_LOGNORMAL_GOODNESS_OF_FIT` | Family | left | [Shapiro-Wilk test for log-normal data](shapiro_wilk_log_normal_criterion.md) |
| `SkewLogNormalGofStatistic` | `SKEW_LOGNORMAL_GOODNESS_OF_FIT` | Family | two_tailed | [Skewness test for log-normal data](skew_log_normal_criterion.md) |
| `SpiegelhalterLogNormalGofStatistic` | `SH_LOGNORMAL_GOODNESS_OF_FIT` | Family | two_tailed | [Spiegelhalter test for log-normal data](spiegelhalter_log_normal_criterion.md) |
| `WatsonLogNormalGofStatistic` | `WATSON_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Watson for log-normal data](watson_log_normal_criterion.md) |
| `ZhangALogNormalGofStatistic` | `ZHANG_ZA_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Zhang A for log-normal data](zhang_a_log_normal_criterion.md) |
| `ZhangCLogNormalGofStatistic` | `ZHANG_ZC_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Zhang C for log-normal data](zhang_c_log_normal_criterion.md) |
| `ZhangKLogNormalGofStatistic` | `ZHANG_ZK_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Zhang K for log-normal data](zhang_k_log_normal_criterion.md) |
| `ZhangQLogNormalGofStatistic` | `ZQ_LOGNORMAL_GOODNESS_OF_FIT` | Family | two_tailed | [Zhang Q test for log-normal data](zhang_q_log_normal_criterion.md) |
| `ZhangQStarLogNormalGofStatistic` | `ZQS_LOGNORMAL_GOODNESS_OF_FIT` | Family | two_tailed | [Zhang Q-star test for log-normal data](zhang_q_star_log_normal_criterion.md) |
| `ZhangWuALogNormalGofStatistic` | `ZWA_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Zhang-Wu A test for log-normal data](zhang_wu_a_log_normal_criterion.md) |
| `ZhangWuCLogNormalGofStatistic` | `ZWC_LOGNORMAL_GOODNESS_OF_FIT` | Fixed | right | [Zhang-Wu C test for log-normal data](zhang_wu_c_log_normal_criterion.md) |

## Calibration and limitations

Fixed-CDF EDF formulas use u=F(X); fitted-normal tables do not apply.
Family statistics reuse the normality functional on log observations. Their
normality approximations and qualifications remain applicable: particularly
Hosking 1–4, GLB, Coin and finite-order moment tests. KL uses a finite r window;
its legacy critical-value regressions are not certified. The report explains
what was independently checked and what remains unverified. Recalibrate changed
statistics; old unversioned stored values are not evidence of correct size.
