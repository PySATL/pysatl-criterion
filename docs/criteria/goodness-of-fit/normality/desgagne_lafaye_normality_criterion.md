# Desgagne-Lafaye test for normality

Desgagne-Lafaye de Micheaux-Leblanc R_n normality statistic.

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

For z=(x-mean(x))/s0, let r=(0.18240929-mean(z**2*log(abs(z)))/2,
0.5348223-mean(log(1+abs(z))),
0.20981558-mean(log(log(e+abs(z))))). Return n*r.T*A*r, where
A is the rounded inverse fitted-score covariance in stat35.
The first integrand is zero at z=0 by continuity. Theorem 1 of the
author preprint accounts for fitting mean and variance. A is ill
conditioned; rounding its constants limits precision. Small-sample
chi-square calibration is inappropriate.

## References

[1] Desgagne, A., Lafaye de Micheaux, P., and Leblanc, A. (2013).
   "Test of Normality Against Generalized Exponential Power
   Alternatives." Communications in Statistics - Theory and Methods,
   42(1), 164-190. https://doi.org/10.1080/03610926.2011.577548

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import DesgagneLafayeNormalityGofStatistic

statistic = DesgagneLafayeNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
