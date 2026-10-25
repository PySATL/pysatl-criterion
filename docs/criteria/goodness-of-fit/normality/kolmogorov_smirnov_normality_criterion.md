# Kolmogorov-Smirnov test for normality

**Class:** `KolmogorovSmirnovNormalityGofStatistic`

**Code:** `KS_NORMALITY_GOODNESS_OF_FIT`

One-sample Kolmogorov-Smirnov statistic for a specified normal law.

## Parameters

```text
parameters : ParameterValues
    Values with mean, var fixed; omitted parameters are unknown.
alternative_type : AlternativeType, optional
    ``TWO_TAILED`` computes ``max(D_plus, D_minus)``; ``RIGHT`` computes
    ``D_plus = sup(F_n - F_0)`` and ``LEFT`` computes
    ``D_minus = sup(F_0 - F_n)``. Defaults to ``TWO_TAILED``.
mode : str, optional
    Setting retained by the shared KS implementation. ``"auto"`` is
    stored as ``"exact"``. It does not affect the statistic and this
    class does not calculate a p-value.
```

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
Large values of the statistic define rejection. Use at least 1
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
from pysatl_criterion.distribution import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import KolmogorovSmirnovNormalityGofStatistic

statistic = KolmogorovSmirnovNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({'mean': 0, 'var': 1}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
