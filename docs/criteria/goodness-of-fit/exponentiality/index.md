# Exponential goodness-of-fit criteria

`continuous/exponent` exports **50 concrete criteria**: 14 generated EDF
classes (including fitted-rate Lilliefors), 5 generated frequency classes and
31 specialized classes. Every criterion is linked below with its class name,
short code, hypothesis and example. See [mathematical verification](mathematical-validation.md)
for formulas, evidence and explicit provenance limitations, including the
[scientific bibliography with article links](mathematical-validation.md#scientific-bibliography).

## Model and use

The origin is zero; `lam` is the positive **rate**, not the scale. The mean is
`1/lam`. Pass `ExponentialDistributionDescriptor.DEFAULT.parse({'lam': 2})`
for a specified rate or `parse({})` for an unknown rate where supported.
The latter is fitted within each generated criterion call. KS D+/D−/D accept
only a fixed rate; Lilliefors accepts only an unknown rate. Other generated
criteria accept both. Specialized criteria accept the unknown-rate model.

Full codes append `_EXPONENTIALITY_GOODNESS_OF_FIT` to the short code.
Class prefixes in the tables append `ExponentialityGofStatistic`.
`fixed` means a specified rate, `unknown` means a composite hypothesis.
Right-tail discrepancies reject large values; both-tail rules use both
extremes. Not all named legacy classes are established omnibus tests:
**Six graph criteria are local constructions**. WE uses the published
normalization $G-1/n$, where $G$ is Greenwood. Several historical
conventions remain qualified in the report. Hollander–Proschan uses the left tail against NBU alternatives.

## Edf criteria

| Criterion | Class prefix | Code | Rate | Rejection |
|---|---|---|---|---|
| [Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_exp_criterion.md) | `KolmogorovSmirnovDPlus` | `KS_PLUS` | fixed | right tail |
| [Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_exp_criterion.md) | `KolmogorovSmirnovDMinus` | `KS_MINUS` | fixed | right tail |
| [Kolmogorov–Smirnov](kolmogorov_smirnov_exp_criterion.md) | `KolmogorovSmirnov` | `KS` | fixed | right tail |
| [Lilliefors exponential](lilliefors_exp_criterion.md) | `Lilliefors` | `LILLIE` | unknown | right tail |
| [Anderson–Darling](anderson_darling_exp_criterion.md) | `AndersonDarling` | `AD` | unknown/fixed | right tail |
| [Cramér–von Mises](cvm_exp_criterion.md) | `CramerVonMises` | `CVM` | fixed/unknown | right tail |
| [Liao–Shimokawa](liao_shimokawa_exp_criterion.md) | `LiaoShimokawa` | `LS` | unknown/fixed | right tail |
| [Watson](watson_exp_criterion.md) | `Watson` | `W` | unknown/fixed | right tail |
| [Kuiper](kuiper_exp_criterion.md) | `Kuiper` | `KUIPER` | unknown/fixed | right tail |
| [Scaled Kuiper](kuiper_scaled_exp_criterion.md) | `KuiperScaled` | `KUIPER_SCALED` | unknown/fixed | right tail |
| [Corrected Kuiper](kuiper_corrected_exp_criterion.md) | `KuiperCorrected` | `KUIPER_CORRECTED` | unknown/fixed | right tail |
| [Zhang Z_K](zhang_k_exp_criterion.md) | `ZhangK` | `ZHANG_ZK` | unknown/fixed | right tail |
| [Zhang Z_A](zhang_a_exp_criterion.md) | `ZhangA` | `ZHANG_ZA` | unknown/fixed | right tail |
| [Zhang Z_C](zhang_c_exp_criterion.md) | `ZhangC` | `ZHANG_ZC` | unknown/fixed | right tail |

## Frequency criteria

| Criterion | Class prefix | Code | Rate | Rejection |
|---|---|---|---|---|
| [Pearson chi-square](pearson_exp_criterion.md) | `Chi2Pearson` | `CHI2_PEARSON` | unknown/fixed | right tail |
| [Freeman–Tukey](freeman_tukey_exp_criterion.md) | `FreemanTukey` | `FREEMAN_TUKEY` | unknown/fixed | right tail |
| [Modified log-likelihood](modified_log_likelihood_exp_criterion.md) | `ModifiedLogLikelihood` | `MODIFIED_LOG_LIKELIHOOD` | unknown/fixed | right tail |
| [Neyman chi-square](neyman_chi2_exp_criterion.md) | `NeymanChi2` | `CHI2_NEYMAN` | unknown/fixed | right tail |
| [Cressie–Read](cressie_read_exp_criterion.md) | `CressieRead` | `CRESSIE_READ` | unknown/fixed | right tail |

## Specialized criteria

| Criterion | Class prefix | Code | Rate | Rejection |
|---|---|---|---|---|
| [Epps–Pulley](epps_pulley_exp_criterion.md) | `EppsPulley` | `EP` | unknown | both tails |
| [Ahsanullah](ahsanullah_exp_criterion.md) | `Ahsanullah` | `AHS` | unknown | right tail |
| [Atkinson](atkinson_exp_criterion.md) | `Atkinson` | `ATK` | unknown | right tail |
| [Cox–Oakes](cox_oakes_exp_criterion.md) | `CoxOakes` | `CO` | unknown | both tails |
| [Deshpande](deshpande_exp_criterion.md) | `Deshpande` | `DSP` | unknown | right tail |
| [Epstein spacing log contrast](epstein_exp_criterion.md) | `Epstein` | `EPS` | unknown | right tail |
| [Frozini EDF distance](frozini_exp_criterion.md) | `Frozini` | `FZ` | unknown | right tail |
| [Gini](gini_exp_criterion.md) | `Gini` | `GINI` | unknown | both tails |
| [Gnedenko F ratio](gnedenko_exp_criterion.md) | `Gnedenko` | `GD` | unknown | both tails |
| [Harris F ratio](harris_exp_criterion.md) | `Harris` | `HM` | unknown | both tails |
| [Hegazy–Green 1](hegazy1_exp_criterion.md) | `HegazyGreen1` | `HG1` | unknown | right tail |
| [Hegazy–Green 2](hegazy2_exp_criterion.md) | `HegazyGreen2` | `HG2` | unknown | right tail |
| [Hollander–Proschan](holander_proshan_exp_criterion.md) | `HollanderProshan` | `HP` | unknown | left tail |
| [Kimber–Michael](kimber_michael_exp_criterion.md) | `KimberMichael` | `KM` | unknown | right tail |
| [Kochar exponential scores](kochar_exp_criterion.md) | `Kochar` | `KC` | unknown | right tail |
| [Lorenz ordinate](lorenz_exp_criterion.md) | `Lorenz` | `LZ` | unknown | right tail |
| [Moran log-mean contrast](moran_exp_criterion.md) | `Moran` | `MN` | unknown | both tails |
| [Pietra](pietra_exp_criterion.md) | `Pietra` | `PT` | unknown | both tails |
| [Shapiro–Wilk exponential W](shapiro_wilk_exp_criterion.md) | `ShapiroWilk` | `SW` | unknown | both tails |
| [Rossberg integral contrast](rossberg_exp_criterion.md) | `Rossberg` | `RS` | unknown | right tail |
| [WE](we_exp_criterion.md) | `We` | `WE` | unknown | right tail |
| [Wong–Wong extremal quotient](wong_wong_exp_criterion.md) | `WongWong` | `WW` | unknown | right tail |
| [Greenwood](greenwood_exp_criterion.md) | `Greenwood` | `GREENWOOD` | unknown | both tails |
| [Baringhaus–Henze empirical Laplace transform](baringhaus_henze_elt_exp_criterion.md) | `BaringhausHenzeELT` | `BH_ELT` | unknown | right tail |
| [Baringhaus–Henze mean residual life](baringhaus_henze_bhc_exp_criterion.md) | `BaringhausHenzeBHC` | `BH_BHC` | unknown | right tail |
| [Graph edge count](graph_edges_number_exp_criterion.md) | `GraphEdgesNumber` | `EDGESNUMBER` | unknown | both tails |
| [Graph maximum degree](graph_max_degree_exp_criterion.md) | `GraphMaxDegree` | `MAXDEGREE` | unknown | both tails |
| [Graph average degree](graph_average_degree_exp_criterion.md) | `GraphAverageDegree` | `AVGDEGREE` | unknown | both tails |
| [Graph connected components](graph_connected_components_exp_criterion.md) | `GraphConnectedComponents` | `CONNECTEDCOMPONENTS` | unknown | both tails |
| [Graph clique number](graph_clique_number_exp_criterion.md) | `GraphCliqueNumber` | `CLIQUENUMBER` | unknown | both tails |
| [Graph independence number](graph_independence_number_exp_criterion.md) | `GraphIndependenceNumber` | `INDEPENDENCENUMBER` | unknown | both tails |

## Calibration example

```python
from pysatl_criterion.distribution import ExponentialDistributionDescriptor as E
from pysatl_criterion.statistics.goodness_of_fit.continuous.exponent import (
    AndersonDarlingExponentialityGofStatistic,
)
from pysatl_criterion.hypothesis_testing.exponential_monte_carlo import monte_carlo_exponential

criterion = AndersonDarlingExponentialityGofStatistic(E.DEFAULT.parse({}))
result = monte_carlo_exponential(
    criterion, [0.12, 0.30, 0.55, 0.90, 1.40, 2.10, 3.20, 4.80],
    n_resamples=199, random_state=20261015,
)
print(result.statistic, result.p_value)
```

Every replicate repeats fitting and preserves settings. Rank p-values retain
infinite boundary penalties and include ties. `binning=EqualProbabilityBins(count=5)`
reproduces the old Pearson bin count; its current code is `CHI2_PEARSON`.
Use the separate D+ and D− classes in place of `alternative_type` on KS.

The examples use ordinary public imports. The audit records a pre-existing
aggregate-import error in this checkout and describes the isolated test run;
that run does not establish package-wide import health.
