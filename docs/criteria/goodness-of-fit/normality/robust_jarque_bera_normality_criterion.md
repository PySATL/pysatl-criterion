# Robust Jarque-Bera test for normality

Gel-Gastwirth robust Jarque-Bera normality statistic.

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

Let J=sqrt(pi/2)*mean(abs(x-median(x))). Return
n*(m3/J**3)**2/6 + n*(m4/J**4-3)**2/64.
Moments are centered at the sample mean, whereas J uses the median.
The reference identifies this robust-scale modification; small-sample
chi-square calibration is not justified.

## References

[1] Gel, Y. R. and Gastwirth, J. L. (2008).
   A robust modification of the Jarque-Bera test of normality.
   Economics Letters, 99(1), 30-32.
   https://doi.org/10.1016/j.econlet.2007.05.022

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import RobustJarqueBeraNormalityGofStatistic

statistic = RobustJarqueBeraNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
