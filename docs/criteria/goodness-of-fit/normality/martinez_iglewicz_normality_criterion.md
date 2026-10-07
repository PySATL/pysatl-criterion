# Martinez-Iglewicz test for normality

Martinez-Iglewicz statistic comparing two estimates of dispersion.

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

Let M=median(x), A=median(abs(x-M)), u=(x-M)/(9*A),
and I={i: abs(u_i)<1}. Define B=n*sum_I((x_i-M)**2*
(1-u_i**2)**4)/(sum_I((1-u_i**2)*(1-5*u_i**2)))**2.
Return sum((x-M)**2)/((n-1)*B). Contributions with abs(u)>=1
are zero in both biweight sums. Zero MAD or a zero biweight
denominator makes the estimator undefined and raises ValueError.

## References

[1] Martinez, J. and Iglewicz, B. (1981).
   A test for departure from normality based on a biweight estimator of scale.
   Biometrika, 68(1), 331-333.
   https://doi.org/10.1093/biomet/68.1.331

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import MartinezIglewiczNormalityGofStatistic

statistic = MartinezIglewiczNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
