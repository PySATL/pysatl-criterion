# Shapiro-Wilk test for normality

Shapiro-Wilk ``W`` statistic for the normal location-scale family.

## Methods

execute_statistic(rvs, **kwargs)
    Return one scalar statistic.
hypothesis()
    Report fixed null parameters only.
alternative()
    Report the critical tail.

## Notes

H0 is N(mu,sigma**2), with unknown real mu and sigma>0; hypothesis() reports {}.
Location and positive scale cancel, allowing N(0,1) null simulation.
Every simulated sample must go through execute_statistic again.
The lower tail of the statistic defines rejection. Use at least 3
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

W = (sum(a_i*x_(i)))**2 / sum((x-mean(x))**2).
The normalized symmetric weights use the Royston polynomial
approximation, with the exact three-observation weights. This is a
weight approximation to Shapiro-Wilk, not exact covariance inversion.

## References

[1] Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance
   test for normality (complete samples). Biometrika, 52(3-4), 591-611.
   https://doi.org/10.1093/biomet/52.3-4.591
[2] Royston, P. (1995). Remark AS R94: A Remark on Algorithm AS 181:
   The W-test for Normality. Applied Statistics, 44(4), 547-551.
   https://doi.org/10.2307/2986146

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import ShapiroWilkNormalityGofStatistic

statistic = ShapiroWilkNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
