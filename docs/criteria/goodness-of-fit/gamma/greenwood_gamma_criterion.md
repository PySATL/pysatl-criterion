# Sum of squared Gamma probability spacings.

G = sum(D_j**2), j=1,...,n+1, where D_j=u_j-u_(j-1),
u_0=0 and u_(n+1)=1. Both endpoint spacings are included.
Repeated observations give zero spacings and are permitted.
The selected right tail detects unusually large squared spacings;
it does not test against exceptionally regular spacing.

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
alpha, beta : float, optional
    Fixed finite positive shape and rate, respectively. Both default to 1.
```

## Source

M. Greenwood (1946), "The Statistical Study of Infectious Diseases",
J. R. Statist. Soc. 109, 85-103.
https://doi.org/10.1111/j.2397-2335.1946.tb04649.x

The general statistic is applied to Gamma probabilities.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit.gamma import GreenwoodGammaGofStatistic

statistic = GreenwoodGammaGofStatistic()
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; large values oppose the null. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
