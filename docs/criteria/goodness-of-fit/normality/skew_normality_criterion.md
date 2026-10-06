# Skewness test for normality

D'Agostino transformed-skewness statistic for the normal family.

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
Both tails of the statistic defines rejection. Use at least 8
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

Return the signed D'Agostino transform delta*asinh(y/alpha) of
g1=m3/m2**1.5, with the sample-size coefficients in skew_test.
In particular, exactly zero sample skewness maps to zero. This tests
skewness departures in either direction; it is not an omnibus test.

## References

[1] D'Agostino, R. B. (1970). Transformation to normality of the null
   distribution of g1. Biometrika, 57(3), 679-681.
   https://doi.org/10.1093/biomet/57.3.679

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import SkewNormalityGofStatistic

statistic = SkewNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
