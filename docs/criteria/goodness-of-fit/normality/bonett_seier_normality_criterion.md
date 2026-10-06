# Bonett-Seier test for normality

Bonett-Seier statistic based on a modified Geary kurtosis measure.

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
Both tails of the statistic defines rejection. Use at least 4
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

The publication was identified, but the exact finite-sample
implementation is not fully certified against its primary text.

Let d=mean(abs(x-mean(x))) and omega=13.29*log(s0/d).
Return sqrt(n+2)*(omega-3)/3.54, a signed kurtosis transform.
The constants are rounded approximations; use finite-sample simulation.

## References

[1] Bonett, D. G. and Seier, E. (2002).
   A test of normality with high uniform power.
   Computational Statistics & Data Analysis, 40(3), 435-445.
   https://doi.org/10.1016/S0167-9473(02)00074-9

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import BonettSeierNormalityGofStatistic

statistic = BonettSeierNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
