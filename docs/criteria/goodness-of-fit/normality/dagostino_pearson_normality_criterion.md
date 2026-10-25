# D’Agostino–Pearson omnibus test

**Class:** `DAPNormalityGofStatistic`

**Code:** `DAP_NORMALITY_GOODNESS_OF_FIT`

D'Agostino-Pearson omnibus ``K_squared`` statistic for normality.

## Parameters

parameters : ParameterValues
    An empty distribution schema; all distribution parameters are unknown.


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
The upper tail of the statistic defines rejection. Use at least 8
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

K2 = Z_skew**2 + Z_kurtosis**2 using the signed transforms
implemented by SkewNormalityGofStatistic and KurtosisNormalityGofStatistic.
The resulting nonnegative discrepancy uses the upper tail.

## References

.. [1] D'Agostino, R. B. and Pearson, E. S. (1973). Tests for departure
   from normality. Empirical results for the distributions of b2 and
   sqrt(b1). Biometrika, 60(3), 613-622.
   https://doi.org/10.1093/biomet/60.3.613

## Examples

```python
import numpy as np
from pysatl_criterion.distribution import NormalDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import DAPNormalityGofStatistic

statistic = DAPNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = np.random.default_rng(42).normal(size=24)
value = statistic.execute_statistic(sample)
assert np.isfinite(value) and value >= 0
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
