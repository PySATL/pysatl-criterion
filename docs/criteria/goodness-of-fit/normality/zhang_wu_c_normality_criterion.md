# Zhang-Wu C test for normality

Zhang-Wu likelihood-ratio normality statistic ``Z_C``.

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
The upper tail of the statistic defines rejection. Use at least 4
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

For z=(x-mean(x))/s1, return sum_i((log((1-Phi(z_(i)))/
Phi(z_(i))) - log((n-i+1/4)/(i-3/4)))**2).
Normal log-CDF/log-survival evaluations retain extreme tail information.

## References

[1] Zhang, J. and Wu, Y. (2005).
   Likelihood-ratio tests for normality.
   Computational Statistics & Data Analysis, 49(3), 709-721.
   https://doi.org/10.1016/j.csda.2004.05.034

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import ZhangWuCNormalityGofStatistic

statistic = ZhangWuCNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
