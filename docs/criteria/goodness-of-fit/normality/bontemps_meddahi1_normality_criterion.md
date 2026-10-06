# Bontemps-Meddahi1 test for normality

Bontemps-Meddahi normality statistic using Hermite orders 3 and 4.

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

Let z=(x-mean(x))/s1 and h_j=He_j/sqrt(j!), where He_j
are probabilists' Hermite polynomials. Return
(sum(h_3(z))**2+sum(h_4(z))**2)/n.
The sample-variance convention ddof=1 is part of this implementation.

## References

[1] Bontemps, C. and Meddahi, N. (2005).
   Testing normality: a GMM approach.
   Journal of Econometrics, 124(1), 149-186.
   https://doi.org/10.1016/j.jeconom.2004.02.014

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import BontempsMeddahi1NormalityGofStatistic

statistic = BontempsMeddahi1NormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
