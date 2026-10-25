# Laplace goodness-of-fit statistics

The 19 fixed-parameter criteria below test independent observations from Laplace(t, s), with finite
location t and positive finite scale s **fixed before observing the sample**.
Construct parameters with `LaplaceDistributionDescriptor.DEFAULT.parse({'t': 0, 's': 1})`.
Missing parameters are rejected; none of these criteria estimates them.

Import classes from `pysatl_criterion.statistics.goodness_of_fit.continuous.laplace`.

| Criterion | Public class | Code |
|---|---|---|
| [Kolmogorov–Smirnov](kolmogorov_smirnov_laplace_criterion.md) | `KolmogorovSmirnovLaplaceGofStatistic` | `KS_LAPLACE_GOODNESS_OF_FIT` |
| [Cramér–von Mises](cvm_laplace_criterion.md) | `CramerVonMisesLaplaceGofStatistic` | `CVM_LAPLACE_GOODNESS_OF_FIT` |
| [Anderson–Darling](anderson_darling_laplace_criterion.md) | `AndersonDarlingLaplaceGofStatistic` | `AD_LAPLACE_GOODNESS_OF_FIT` |
| [Kuiper](kuiper_laplace_criterion.md) | `KuiperLaplaceGofStatistic` | `KUI_LAPLACE_GOODNESS_OF_FIT` |
| [Watson](watson_laplace_criterion.md) | `WatsonLaplaceGofStatistic` | `WAT_LAPLACE_GOODNESS_OF_FIT` |
| [Greenwood](greenwood_laplace_criterion.md) | `GreenwoodLaplaceGofStatistic` | `GRW_LAPLACE_GOODNESS_OF_FIT` |
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_laplace_criterion.md) | `KolmogorovSmirnovDPlusLaplaceGofStatistic` | `KS_PLUS_LAPLACE_GOODNESS_OF_FIT` |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_laplace_criterion.md) | `KolmogorovSmirnovDMinusLaplaceGofStatistic` | `KS_MINUS_LAPLACE_GOODNESS_OF_FIT` |
| [Liao–Shimokawa](liao_shimokawa_laplace_criterion.md) | `LiaoShimokawaLaplaceGofStatistic` | `LIAO_SHIMOKAWA_LAPLACE_GOODNESS_OF_FIT` |
| [Scaled Kuiper](kuiper_scaled_laplace_criterion.md) | `KuiperScaledLaplaceGofStatistic` | `KUIPER_SCALED_LAPLACE_GOODNESS_OF_FIT` |
| [Corrected Kuiper](kuiper_corrected_laplace_criterion.md) | `KuiperCorrectedLaplaceGofStatistic` | `KUIPER_CORRECTED_LAPLACE_GOODNESS_OF_FIT` |
| [Zhang Z_A](zhang_a_laplace_criterion.md) | `ZhangALaplaceGofStatistic` | `ZHANG_ZA_LAPLACE_GOODNESS_OF_FIT` |
| [Zhang Z_K](zhang_k_laplace_criterion.md) | `ZhangKLaplaceGofStatistic` | `ZHANG_ZK_LAPLACE_GOODNESS_OF_FIT` |
| [Zhang Z_C](zhang_c_laplace_criterion.md) | `ZhangCLaplaceGofStatistic` | `ZHANG_ZC_LAPLACE_GOODNESS_OF_FIT` |
| [Freeman–Tukey](freeman_tukey_laplace_criterion.md) | `FreemanTukeyLaplaceGofStatistic` | `FREEMAN_TUKEY_LAPLACE_GOODNESS_OF_FIT` |
| [Modified log-likelihood](modified_log_likelihood_laplace_criterion.md) | `ModifiedLogLikelihoodLaplaceGofStatistic` | `MODIFIED_LOG_LIKELIHOOD_LAPLACE_GOODNESS_OF_FIT` |
| [Neyman modified chi-squared](neyman_chi2_laplace_criterion.md) | `NeymanChi2LaplaceGofStatistic` | `CHI2_NEYMAN_LAPLACE_GOODNESS_OF_FIT` |
| [Cressie–Read](cressie_read_laplace_criterion.md) | `CressieReadLaplaceGofStatistic` | `CRESSIE_READ_LAPLACE_GOODNESS_OF_FIT` |
| [Pearson chi-squared](chi2_pearson_laplace_criterion.md) | `Chi2PearsonLaplaceGofStatistic` | `CHI2_PEARSON_LAPLACE_GOODNESS_OF_FIT` |

`AbstractLaplaceGofStatistic` is an abstract public base, not a registered criterion.
All fixed-parameter criteria reject for large statistic values. KS additionally accepts a CDF
comparison direction; LEFT and RIGHT still use right-tail rejection.
The catalog includes 13 EDF criteria, five frequency criteria and Greenwood.
Frequency criteria accept `binning=`; calibration must use the same binning policy.

See [mathematical validation](mathematical-validation.md) for independent checks,
implementation changes, numerical limitations and calibration requirements.

The EDF extensions are D+, D−, Liao–Shimokawa, scaled/corrected Kuiper and
Zhang K/A/C. They use the existing shared definitions and analytic Laplace log
probabilities for tail-sensitive algorithms. The frequency extensions are Pearson,
Freeman–Tukey, modified log likelihood, Neyman and Cressie–Read. Their definitions
and binning rules are described in the [universal criteria](../universal.md).

```python
from pysatl_criterion.distribution import LaplaceDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.laplace import (
    Chi2PearsonLaplaceGofStatistic,
    KolmogorovSmirnovDPlusLaplaceGofStatistic,
)

parameters = LaplaceDistributionDescriptor.DEFAULT.parse({'t': 0, 's': 1})
sample = [-2., -1., -.5, 0., .2, .5, 1., 2.]
d_plus = KolmogorovSmirnovDPlusLaplaceGofStatistic(parameters).execute_statistic(sample)
pearson = Chi2PearsonLaplaceGofStatistic(
    parameters, binning=EqualProbabilityBins(count=4)
).execute_statistic(sample)
```

The legacy KS constructor still accepts `alternative_type` and `mode`.
Dedicated D+ and D− classes have distinct registry codes; all reject in the right tail.

## Fitted location-scale family

The package also contains 19 specialist criteria (38 concrete criteria in total).
Both parameters are unknown: pass `Distribution.DEFAULT.parse({})`.
Choi–Kim rejects for small values; V3, V4, GV and DLO_Z are signed two-sided
statistics; the other added criteria reject for large values.
Each replicate must be fitted anew. See [definitions and calibration](fitted-definitions.md).

| Criterion | Public class | Code |
|---|---|---|
| [A_RAT](a_rat_laplace_criterion.md) | `AlizadehNoughabiLikelihoodLaplaceGofStatistic` | `A_RAT_LAPLACE_GOODNESS_OF_FIT` |
| [AJ](aj_laplace_criterion.md) | `AlizadehNoughabiJarrahiferizLaplaceGofStatistic` | `AJ_LAPLACE_GOODNESS_OF_FIT` |
| [ME1](me1_laplace_criterion.md) | `Meintanis1LaplaceGofStatistic` | `ME1_LAPLACE_GOODNESS_OF_FIT` |
| [ME2](me2_laplace_criterion.md) | `Meintanis2LaplaceGofStatistic` | `ME2_LAPLACE_GOODNESS_OF_FIT` |
| [CK_V](ck_v_laplace_criterion.md) | `ChoiKimVLaplaceGofStatistic` | `CK_V_LAPLACE_GOODNESS_OF_FIT` |
| [CK_C](ck_c_laplace_criterion.md) | `ChoiKimCLaplaceGofStatistic` | `CK_C_LAPLACE_GOODNESS_OF_FIT` |
| [CK_E](ck_e_laplace_criterion.md) | `ChoiKimELaplaceGofStatistic` | `CK_E_LAPLACE_GOODNESS_OF_FIT` |
| [AP_V](ap_v_laplace_criterion.md) | `AlizadehNoughabiParkVLaplaceGofStatistic` | `AP_V_LAPLACE_GOODNESS_OF_FIT` |
| [AP_E](ap_e_laplace_criterion.md) | `AlizadehNoughabiParkELaplaceGofStatistic` | `AP_E_LAPLACE_GOODNESS_OF_FIT` |
| [AP_Y](ap_y_laplace_criterion.md) | `AlizadehNoughabiParkYLaplaceGofStatistic` | `AP_Y_LAPLACE_GOODNESS_OF_FIT` |
| [AP_A](ap_a_laplace_criterion.md) | `AlizadehNoughabiParkALaplaceGofStatistic` | `AP_A_LAPLACE_GOODNESS_OF_FIT` |
| [AP_Z](ap_z_laplace_criterion.md) | `AlizadehNoughabiParkZLaplaceGofStatistic` | `AP_Z_LAPLACE_GOODNESS_OF_FIT` |
| [BRT_V3](brt_v3_laplace_criterion.md) | `BestRaynerThasV3LaplaceGofStatistic` | `BRT_V3_LAPLACE_GOODNESS_OF_FIT` |
| [BRT_V4](brt_v4_laplace_criterion.md) | `BestRaynerThasV4LaplaceGofStatistic` | `BRT_V4_LAPLACE_GOODNESS_OF_FIT` |
| [GE](ge_laplace_criterion.md) | `GelLaplaceGofStatistic` | `GE_LAPLACE_GOODNESS_OF_FIT` |
| [GV](gv_laplace_criterion.md) | `GonzalezEstradaVillasenorLaplaceGofStatistic` | `GV_LAPLACE_GOODNESS_OF_FIT` |
| [DLO_X](dlo_x_laplace_criterion.md) | `DesgagneLafayeDeMicheauxOuimetXLaplaceGofStatistic` | `DLO_X_LAPLACE_GOODNESS_OF_FIT` |
| [DLO_Z](dlo_z_laplace_criterion.md) | `DesgagneLafayeDeMicheauxOuimetZLaplaceGofStatistic` | `DLO_Z_LAPLACE_GOODNESS_OF_FIT` |
| [DD_SMOOTH](dd_smooth_laplace_criterion.md) | `DataDrivenSmoothLaplaceGofStatistic` | `DD_SMOOTH_LAPLACE_GOODNESS_OF_FIT` |
