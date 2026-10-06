# Looney-Gulledge test for normality

Looney-Gulledge probability-plot correlation statistic for normality.

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
The lower tail of the statistic defines rejection. Use at least 3
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

Return corr(x_(i), Phi**-1((i-3/8)/(n+1/4))).
The Blom plotting-position version is used without averaging ties.

## References

[1] Looney, S. W. and Gulledge, T. R., Jr. (1985). Use of the
   Correlation Coefficient with Normal Probability Plots. The American
   Statistician, 39(1), 75-79.
   https://doi.org/10.1080/00031305.1985.10479395

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import LooneyGulledgeNormalityGofStatistic

statistic = LooneyGulledgeNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
