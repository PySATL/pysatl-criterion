# Log-logistic goodness-of-fit statistics

The family has **20 concrete criteria**: 13 EDF, 5 frequency and 2 fitted-model
statistics. [Mathematical verification](mathematical-validation.md) records the
formula audit, numerical checks, corrections and calibration limitations.

For $x>0$ the distribution is

$$
F_{\alpha,\beta}(x)=\frac1{1+(\alpha/x)^\beta},\qquad
\alpha>0,\quad\beta>0.
$$

Equivalently, $\log X$ is logistic with location $\log\alpha$ and scale $1/\beta$.
CDF and logarithmic tails are evaluated through logarithms to avoid overflow of
$x/\alpha$. All criteria reject for large values. Their output is a statistic,
not a calibrated p-value.

## Catalog

Each class name below has the suffix `LogLogisticGofStatistic`.

| Criterion | Class prefix | Hypothesis | Input |
|---|---|---|---|
| [Kolmogorov–Smirnov D+](ks-plus.md) | `KolmogorovSmirnovDPlus` | Both parameters fixed | Complete positive sample |
| [Kolmogorov–Smirnov D−](ks-minus.md) | `KolmogorovSmirnovDMinus` | Both parameters fixed | Complete positive sample |
| [Kolmogorov–Smirnov](ks.md) | `KolmogorovSmirnov` | Both parameters fixed | Complete positive sample |
| [Anderson–Darling](ad.md) | `AndersonDarling` | Both parameters fixed | Complete positive sample |
| [Cramér–von Mises](cvm.md) | `CramerVonMises` | Both parameters fixed | Complete positive sample |
| [Liao–Shimokawa](liao-shimokawa.md) | `LiaoShimokawa` | Both parameters fixed | Complete positive sample |
| [Watson](watson.md) | `Watson` | Both parameters fixed | Complete positive sample |
| [Kuiper](kuiper.md) | `Kuiper` | Both parameters fixed | Complete positive sample |
| [Scaled Kuiper](kuiper-scaled.md) | `KuiperScaled` | Both parameters fixed | Complete positive sample |
| [Corrected Kuiper](kuiper-corrected.md) | `KuiperCorrected` | Both parameters fixed | Complete positive sample |
| [Zhang K](zhang-k.md) | `ZhangK` | Both parameters fixed | Complete positive sample |
| [Zhang A](zhang-a.md) | `ZhangA` | Both parameters fixed | Complete positive sample |
| [Zhang C](zhang-c.md) | `ZhangC` | Both parameters fixed | Complete positive sample |
| [Pearson chi-squared](pearson.md) | `Chi2Pearson` | Both parameters fixed | Complete positive sample |
| [Freeman–Tukey](freeman-tukey.md) | `FreemanTukey` | Both parameters fixed | Complete positive sample |
| [Modified log-likelihood](modified-log-likelihood.md) | `ModifiedLogLikelihood` | Both parameters fixed | Complete positive sample |
| [Neyman chi-squared](neyman-chi2.md) | `NeymanChi2` | Both parameters fixed | Complete positive sample |
| [Cressie–Read](cressie-read.md) | `CressieRead` | Both parameters fixed | Complete positive sample |
| [Bagdonavicius–Nikulin](nikulin.md) | `Nikulin` | Both parameters unknown | Positive times; optional event flags |
| [Mirvaliev](mirvaliev.md) | `Mirvaliev` | Both parameters unknown | Complete positive sample |

## Parameters and examples

There are no implicit unit-scale/unit-shape defaults in the criterion constructor.
The 18 generated criteria require both parameters; Nikulin and Mirvaliev require
an empty parameter set. Each individual page contains a runnable example and
scientific references. For example:

```python
from pysatl_criterion.distribution import LogLogisticDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_logistic import (
    AndersonDarlingLogLogisticGofStatistic,
)

parameters = Distribution.DEFAULT.parse({"alpha": 1.0, "beta": 2.0})
statistic = AndersonDarlingLogLogisticGofStatistic(parameters)
value = statistic.execute_statistic([0.2, 0.5, 0.8, 1.4, 2.5])
assert value >= 0
```

## Calibration

For fixed parameters, the probability integral transform gives a uniform null
for EDF statistics. Equal-probability frequency cells give multinomial counts
with equal probabilities. Preserve sample size, the exact statistic normalization
and the binning configuration in simulation. General binning strategies need
calibration under the actual distribution. The default frequency interface is
`binning=EqualProbabilityBins()`; the former `bins` keyword is not supported.

For Mirvaliev, refit log-moments in every replicate. For Nikulin, reproduce the
censoring and observation scheme, refit the likelihood and recompute exposure
intervals. Event flag 1 means failure. No built-in log-logistic generator is
registered, so simulation is external. Critical values for fitted Weibull tests
or fixed-parameter KS tests cannot be reused for these fitted criteria.

See [binning](../binning/index.md) and the [verification report](mathematical-validation.md)
for sources and limitations.
