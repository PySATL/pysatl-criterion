# Cramer-von Mises test for normality

**Class:** `CramerVonMiseNormalityGofStatistic`

**Code:** `CVM_GOODNESS_OF_FIT`

Cramer-von Mises statistic for a specified normal distribution.

## Parameters

```text
parameters : ParameterValues
    Values with mean, var fixed; omitted parameters are unknown.
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

W2 = 1/(12*n) + sum((F0(x_(i)) - (2*i-1)/(2*n))**2).
F0 is the normal CDF with the fixed mean and variance.

## References

[1] Cramer, H. (1928). On the composition of elementary errors.
   Second paper: Statistical applications. Scandinavian Actuarial
   Journal, 1928(1), 141-180.
   https://doi.org/10.1080/03461238.1928.10416872

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import CramerVonMiseNormalityGofStatistic

statistic = CramerVonMiseNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({'mean': 0, 'var': 1}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
