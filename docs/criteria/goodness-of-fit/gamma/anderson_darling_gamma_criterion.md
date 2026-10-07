# Anderson-Darling statistic for a fixed Gamma CDF.

A2 = -n - sum((2*i-1)*(log(u_i)+log(1-u_(n+1-i))))/n.
Direct log-CDF and log-survival evaluations retain upper-tail precision.
Zero observations give positive infinity. Extreme tails may underflow
in SciPy, also producing infinity; no epsilon clipping is applied.

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
```

## Source

T. W. Anderson and D. A. Darling (1952), "Asymptotic Theory of Certain
Goodness of Fit Criteria Based on Stochastic Processes", Ann. Math.
Statist. 23, 193-212. https://doi.org/10.1214/aoms/1177729437

The general statistic is applied to Gamma probabilities.

## Usage

```python
from pysatl_criterion.distribution.distributions import GammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.gamma import AndersonDarlingGammaGofStatistic

statistic = AndersonDarlingGammaGofStatistic(GammaDistributionDescriptor.DEFAULT.parse({'alfa': 1, 'beta': 1}))
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; large values oppose the null. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
