# Epps-Pulley test for normality

Epps-Pulley empirical-characteristic-function statistic for normality.

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
The upper tail of the statistic defines rejection. Use at least 2
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

For z=(x-mean(x))/s0, return
n/sqrt(3) + sum_ij(exp(-(z_i-z_j)**2/2))/n
- sqrt(2)*sum_i(exp(-z_i**2/4)).
This equals n times a Gaussian-weighted squared characteristic-function
distance. The double sum includes its diagonal; computation is O(n**2).

## References

[1] Epps, T. W. and Pulley, L. B. (1983). A test for normality based
   on the empirical characteristic function. Biometrika, 70(3), 723-726.
   https://doi.org/10.1093/biomet/70.3.723

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import EppsPulleyNormalityGofStatistic

statistic = EppsPulleyNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
