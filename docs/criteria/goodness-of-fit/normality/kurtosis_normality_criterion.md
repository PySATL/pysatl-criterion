# Kurtosis test for normality

Anscombe-Glynn transformed-kurtosis statistic for the normal family.

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
Both tails of the statistic defines rejection. Use at least 5
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

Return the signed Anscombe-Glynn transform of g2=m4/m2**2,
using the finite-n expectation, variance and skewness of g2. A real
cube root is used. A zero transformation denominator is undefined.
Both light and heavy tails matter; the normal approximation is
particularly inaccurate for small samples.

## References

[1] Anscombe, F. J. and Glynn, W. J. (1983). Distribution of the
   kurtosis statistic b2 for normal samples. Biometrika, 70(1), 227-234.
   https://doi.org/10.1093/biomet/70.1.227

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import KurtosisNormalityGofStatistic

statistic = KurtosisNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
