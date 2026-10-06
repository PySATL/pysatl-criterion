# Kolmogorov-Smirnov test for normality

One-sample Kolmogorov-Smirnov statistic for a specified normal law.

## Parameters

alternative_type : AlternativeType, optional
    ``TWO_TAILED`` computes ``max(D_plus, D_minus)``; ``RIGHT`` computes
    ``D_plus = sup(F_n - F_0)`` and ``LEFT`` computes
    ``D_minus = sup(F_0 - F_n)``. Defaults to ``TWO_TAILED``.
mode : str, optional
    Setting retained by the shared KS implementation. ``"auto"`` is
    stored as ``"exact"``. It does not affect the statistic and this
    class does not calculate a p-value.
mean : float, optional
    Finite mean fixed by the null hypothesis. Default is 0.
var : float, optional
    Positive, finite variance fixed by the null hypothesis. Default is 1.
    The normal CDF uses ``scale=sqrt(var)``.

## Methods

execute_statistic(rvs, **kwargs)
    Return one scalar statistic.
hypothesis()
    Report fixed null parameters only.
alternative()
    Report the critical tail.

## Notes

H0 is N(mean,var) with both parameters fixed; hypothesis() reports both.
Constant samples are valid. Simulate the specified normal law.
The upper tail of the statistic defines rejection. Use at least 1
finite real observations in one dimension.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

D+ = max(i/n - F0(x_(i))) and D- = max(F0(x_(i)) - (i-1)/n).
Return D+, D-, or max(D+, D-) according to alternative_type.
F0 is the normal CDF with the fixed mean and variance. All three
distances use the upper critical tail; mode does not compute a p-value.

## References

[1] Smirnov, N. (1948). Table for Estimating the Goodness of Fit of
   Empirical Distributions. The Annals of Mathematical Statistics,
   19(2), 279-281. https://doi.org/10.1214/aoms/1177730256

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import KolmogorovSmirnovNormalityGofStatistic

statistic = KolmogorovSmirnovNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
