# Doornik-Hansen test for normality

Univariate Doornik-Hansen normality statistic.

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

Return Z1**2+Z2**2, with the Doornik-Hansen transformations of
g1=m3/m2**1.5 and g2=m4/m2**2 (Appendix A of the author preprint).
Z1 uses asinh; Z2 uses the real cube root and includes the g1**2
adjustment to kurtosis. The chi-square law is asymptotic.

## References

[1] Doornik, J. A. and Hansen, H. (2008).
   An Omnibus Test for Univariate and Multivariate Normality.
   Oxford Bulletin of Economics and Statistics, 70(s1), 927-939.
   https://doi.org/10.1111/j.1468-0084.2008.00537.x

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import DoornikHansenNormalityGofStatistic

statistic = DoornikHansenNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
