# KS distance to a fixed Gamma CDF.

D+ = max(i/n-u_i), D- = max(u_i-(i-1)/n), i=1,...,n.
Return max(D+,D-), D+, or D- for TWO_TAILED, RIGHT, or LEFT.
All three reject for large values. No parameters are fitted.
Stored calibration supports only the default two-sided CDF distance.

For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
where Gamma has shape alpha, rate beta and known origin zero.
Parameters are fixed, not fitted. Samples must be finite, real,
one-dimensional and nonempty, in the closed support [0, infinity).
Zero is accepted as a boundary value. Ties and constant samples are
allowed except where stated. Inputs and instance state are unchanged.
Continuous-null calibration assumes iid observations; rounded data
require calibration of the observation process.

## Parameters

```text
parameters : ParameterValues
    Values with alfa, beta fixed; omitted parameters are unknown.
alternative_type : AlternativeType, optional
    TWO_TAILED (default), RIGHT or LEFT selects D, D+ or D-.
mode : {"auto", "exact", "asymp", "approx"}, optional
    Compatibility setting; no effect on the scalar statistic.
```

## Source

N. Smirnov (1948), "Table for Estimating the Goodness of Fit of Empirical
Distributions", Ann. Math. Statist. 19, 279-281.
https://doi.org/10.1214/aoms/1177730256

The general statistic is applied to Gamma probabilities.

## Usage

```python
from pysatl_criterion.distribution.distributions import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.gamma import KolmogorovSmirnovGammaGofStatistic

statistic = KolmogorovSmirnovGammaGofStatistic(GammaDistributionDescriptor.DEFAULT.parse({'alfa': 1, 'beta': 1}))
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; large values oppose the null. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
