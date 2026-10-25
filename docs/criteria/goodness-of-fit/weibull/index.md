# Weibull goodness-of-fit criteria

These criteria test the ordinary two-parameter Weibull family with zero location.
Both shape and scale are unknown: construct a criterion with
`WeibullDistributionDescriptor.DEFAULT.parse({})`.
The EDF and frequency criteria below fit both parameters by maximum likelihood
on every call, including every Monte Carlo replicate.
Samples must be finite, strictly positive, nonconstant and contain at least two observations.

The module exports 33 concrete classes: 13 generated EDF criteria, 5 generated
frequency criteria and 15 specialized classes.

See [mathematical verification](mathematical-validation.md) for formula checks,
scientific sources, source-verification limits and numerical test coverage.

## EDF criteria

All classes are available from
`pysatl_criterion.statistics.goodness_of_fit.continuous.weibull`.
Each class name in the table has the suffix `WeibullGofStatistic`.
All reject for large statistic values.

| Criterion | Class prefix | Short code |
|---|---|---|
| [One-sided Kolmogorov–Smirnov D+](kolmogorov_smirnov_d_plus_weibull_criterion.md) | `KolmogorovSmirnovDPlus` | `KS_PLUS` |
| [One-sided Kolmogorov–Smirnov D−](kolmogorov_smirnov_d_minus_weibull_criterion.md) | `KolmogorovSmirnovDMinus` | `KS_MINUS` |
| [Kolmogorov–Smirnov](kolmogorov_smirnov_weibull_criterion.md) | `KolmogorovSmirnov` | `KS` |
| [Anderson–Darling](anderson_darling_weibull_criterion.md) | `AndersonDarling` | `AD` |
| [Cramér–von Mises](cvm_weibull_criterion.md) | `CrammerVonMises` | `CVM` |
| [Liao–Shimokawa](liao_shimokawa_weibull_criterion.md) | `LiaoShimokawa` | `LS` |
| [Watson](watson_weibull_criterion.md) | `Watson` | `WATSON` |
| [Kuiper](kuiper_weibull_criterion.md) | `Kuiper` | `KUIPER` |
| [Scaled Kuiper](kuiper_scaled_weibull_criterion.md) | `KuiperScaled` | `KUIPER_SCALED` |
| [Corrected Kuiper](kuiper_corrected_weibull_criterion.md) | `KuiperCorrected` | `KUIPER_CORRECTED` |
| [Zhang Z_K](zhang_k_weibull_criterion.md) | `ZhangK` | `ZHANG_ZK` |
| [Zhang Z_A](zhang_a_weibull_criterion.md) | `ZhangA` | `ZHANG_ZA` |
| [Zhang Z_C](zhang_c_weibull_criterion.md) | `ZhangC` | `ZHANG_ZC` |

CDF values are evaluated under the fitted Weibull model.
Anderson–Darling, Liao–Shimokawa and the Zhang statistics use logarithmic
probabilities to retain tail precision. The definitions of the EDF discrepancies
are described in the [uniformity catalog](../uniformity/index.md);
here their null distributions reflect the estimation of Weibull parameters.

## Frequency criteria

These classes also have the suffix `WeibullGofStatistic` and reject for large values.
They accept a keyword-only `binning` argument; the default is `EqualProbabilityBins()`.

| Criterion | Class prefix | Short code |
|---|---|---|
| [Pearson chi-squared](chi2_pearson_weibull_criterion.md) | `Chi2Pearson` | `CHI2_PEARSON` |
| [Freeman–Tukey](freeman_tukey_weibull_criterion.md) | `FreemanTukey` | `FREEMAN_TUKEY` |
| [Modified log-likelihood ratio](modified_log_likelihood_weibull_criterion.md) | `ModifiedLogLikelihood` | `MODIFIED_LOG_LIKELIHOOD` |
| [Neyman chi-squared](neyman_chi2_weibull_criterion.md) | `NeymanChi2` | `CHI2_NEYMAN` |
| [Cressie–Read](cressie_read_weibull_criterion.md) | `CressieRead` | `CRESSIE_READ` |

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    Chi2PearsonWeibullGofStatistic,
)

test = Chi2PearsonWeibullGofStatistic(
    Weibull.DEFAULT.parse({}),
    binning=EqualProbabilityBins(count=4),
)
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
```

For details of the rules and their settings, see [binning](../binning/index.md).
The selected rule is included in `configuration()`.

## Calibration

For the EDF criteria and frequency criteria with `EqualProbabilityBins`,
affine invariance in log observations permits simulation at shape = scale = 1.
Every replicate must re-estimate both parameters. Critical values for a fully
specified Weibull or uniform distribution are not applicable.

Other binning rules, including fixed boundaries in observation units, require
external calibration with explicit simulation parameters. The built-in Monte Carlo
resolver rejects these rules for fitted Weibull, because their null distributions
need not share the same invariance. Existing storage-calibration guards remain in
place for this family.

The 15 additional Weibull-specific criteria retain their individual definitions
and calibration requirements; see their pages in this section.

## Specialized criteria

These explicit implementations remain separate from the generated kernels.

| Class | Documentation |
|---|---|
| `CabanaQuirozWeibullGofStatistic` | [Krit maximum-likelihood Cabana-Quiroz CQ* quadratic statistic](cabana_quiroz_weibull_criterion.md) |
| `KullbackLeiblerWeibullGofStatistic` | [Vasicek-entropy KL test for log-Weibull (MLE)](kullback_leibler_weibull_criterion.md) |
| `LaplaceTransform2WeibullGofStatistic` | [Krit LT2 discrete Laplace statistic with maximum-likelihood fits](laplace_transform2_weibull_criterion.md) |
| `LaplaceTransform3WeibullGofStatistic` | [Krit LT3 discrete Laplace statistic with maximum-likelihood fits](laplace_transform3_weibull_criterion.md) |
| `LockhartOReillyStephensWeibullGofStatistic` | [Lockhart–O’Reilly–Stephens spacing Anderson–Darling statistic](los_weibull_criterion.md) |
| `MahdiDoostparastWeibullGofStatistic` | [Doostparast weighted record EDF distance with record-likelihood fitting](mahdi_doostparast_weibull_criterion.md) |
| `MannScheuerFertigWeibullGofStatistic` | [Mann–Scheuer–Fertig upper log-spacing fraction](msf_weibull_criterion.md) |
| `OzturkKorukogluWeibullGofStatistic` | [Öztürk–Korukoğlu standardized scale-ratio test](ozturk_korukoglu_weibull_criterion.md) |
| `EvansJohnsonGreenWeibullGofStatistic` | [Evans–Johnson–Green squared log-probability-plot correlation](rejg_weibull_criterion.md) |
| `SmithBainWeibullGofStatistic` | [Smith–Bain mean-rank correlation discrepancy](rsb_weibull_criterion.md) |
| `ShapiroBrainWeibullGofStatistic` | [Shapiro–Brain centered log-order linear-estimate ratio](shapiro_brain_weibull_criterion.md) |
| `ColesWeibullGofStatistic` | [Coles stabilized probability-plot distance with Blom estimates](spp_weibull_criterion.md) |
| `BestRaynerThasSkewnessWeibullGofStatistic` (`ST1` alias) | [Best–Rayner–Thas skewness component](st1_weibull_criterion.md) |
| `BestRaynerThasKurtosisWeibullGofStatistic` (`ST2` alias) | [Best–Rayner–Thas kurtosis component](st2_weibull_criterion.md) |
| `TikuSinghWeibullGofStatistic` | [Tiku–Singh normalized log-spacing statistic](tiku_singh_weibull_criterion.md) |

### Shapiro–Brain source and calibration

`ShapiroBrainWeibullGofStatistic` (`SB`) uses a centered version of the
[Shapiro–Brain W statistic (1987)](https://doi.org/10.1080/03610918708812586),
with the formula checked through [Shell (2003), equation (9)](https://ipnpr.jpl.nasa.gov/progress_report/42-155/155B.pdf).
It performs no MLE fit and uses both rejection tails. See its
[criterion page](shapiro_brain_weibull_criterion.md) for the rounded-coefficient
centering adjustment and the published-example comparison.

The Öztürk–Korukoğlu criterion implements the standardized $OK_n^*$ ratio
from [Öztürk and Korukoğlu (1988)](https://doi.org/10.1080/03610918808812730),
as reproduced in [Krit (2014), §3.2.2](https://theses.hal.science/tel-01126901/document).
It uses two linear scale estimates of log observations and both rejection tails;
the centering adjustment is explained on its criterion page.

`SmithBainWeibullGofStatistic` (`RSB`) computes $n(1-R^2)$ using log observations
and scores $\log[-\log(1-i/(n+1))]$. This matches
[Krit (2014), §3.2.1, (3.12)](https://theses.hal.science/tel-01126901/document),
attributed there to [Smith and Bain (1976)](https://doi.org/10.1080/03610927608827337).
It rejects in the right tail, eliminates both parameters without fitting, and
requires sample-size-specific calibration. At $n=2$ it is degenerate; use
$n\ge3$ for testing. See the [mathematical verification](mathematical-validation.md#smithbain-follow-up-verification)
for the source qualification and rejection-rule correction.


`EvansJohnsonGreenWeibullGofStatistic` (`REJG`) returns squared correlation
$R^2$ using log observations and ranks $(i-0.3175)/(n+0.365)$ at every index.
The exact formula is checked through [Krit (2014), (3.13)](https://theses.hal.science/tel-01126901/document),
which cites [Evans, Johnson and Green (1989)](https://www.govinfo.gov/content/pkg/GOVPUB-A13-PURL-gpo123139/pdf/GOVPUB-A13-PURL-gpo123139.pdf).
It rejects in the left tail, needs no parameter fit and is degenerate at $n=2$.
See the [verification](mathematical-validation.md#evansjohnsongreen-follow-up-verification)
for the cancellation of the fitted shape, source limitations and calibration.


`ColesWeibullGofStatistic` (`SPP`) uses Blom log-location/log-scale estimates
and returns the maximum arcsine-square-root transformed probability distance.
The former MLE fit has been replaced. The distance follows
[Krit (2014), (3.14)–(3.15)](https://theses.hal.science/tel-01126901/document),
with Blom estimation motivated by [Coles (1989)](https://doi.org/10.1093/biomet/76.3.593).
It rejects in the right tail and must be recalibrated with Blom on every
replicate. See [the criterion page](spp_weibull_criterion.md) for the explicit
order-statistic approximation, $n=2$ degeneracy and source limitations.


`TikuSinghWeibullGofStatistic` (`TS`) uses the generalized-spacing functional
in [Sürücü (2008), (3.4)–(3.5)](https://doi.org/10.1016/j.camwa.2008.03.010),
applied to the log-Weibull family of
[Tiku and Singh (1981)](https://doi.org/10.1080/03610928108828082).
It requires complete samples with $n\ge3$, uses approximate expected log-order
spacings, performs no parameter fit and rejects in both tails. Calibrate at the
observed sample size using this implementation and shape = scale = 1.
See [verification](mathematical-validation.md#tikusingh-follow-up-verification)
for the indexing qualification in Krit (2014) and approximation details.


`MannScheuerFertigWeibullGofStatistic` (`MSF`) returns the fraction of normalized
log spacings above index $\lfloor n/2\rfloor$. It matches
[Krit (2014), (29)](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf),
attributed to [Mann, Scheuer and Fertig (1973)](https://doi.org/10.1080/03610927308827085),
with explicitly approximate expected order statistics. It uses the right tail,
requires complete samples with $n\ge3$, and performs no parameter fit.
The former abbreviated class name has been removed without an alias;
imports and serialized class references must use the full name.
See [verification](mathematical-validation.md#mannscheuerfertig-follow-up-verification)
for index checks, range and calibration qualifications.


`LockhartOReillyStephensWeibullGofStatistic` (`LOS`) applies the Anderson–Darling
functional to $n-2$ cumulative normalized log spacings, following
[Lockhart, O’Reilly and Stephens (1986), equations (1)–(3)](https://doi.org/10.1111/j.2517-6161.1986.tb01418.x),
with approximate expected order statistics. It requires complete samples with
$n\ge3$, eliminates both parameters without fitting, and rejects in the right
tail. Endpoint ties give infinity. Ordinary uniform AD tables do not apply.
The abbreviated class name has been removed without an alias. See
[verification](mathematical-validation.md#lockhartoreillystephens-follow-up-verification).


`KullbackLeiblerWeibullGofStatistic` (`KL`) implements the Gumbel entropy test
of [Pérez-Rodríguez et al. (2009)](https://doi.org/10.1080/03610920802316658)
on log-Weibull data, with the functional checked against EWGoF's `WEDF.test`
KL branch. Both parameters are fitted by MLE; large values reject. The
validated local default window differs from EWGoF's piecewise rule, so
calibration must match $n$, $m$ and the estimator. The finite-sample estimate
is not clipped at zero. See [verification](mathematical-validation.md#kullbackleibler-follow-up-verification)
for source limitations, Gumbel signs and window conventions.
