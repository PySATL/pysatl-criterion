# Chen-Shapiro test for normality

Chen-Shapiro normality statistic based on normalized spacings.

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

For u_i=Phi**-1((i-3/8)/(n+1/4)), return
sqrt(n)*(1-sum_i((x_(i+1)-x_(i))/(u_(i+1)-u_i))/((n-1)*s1)).
This is the upper-tail QH* version, not the lower-tail raw QH.

## References

[1] Chen, L. and Shapiro, S. S. (1995).
   An alternative test for normality based on normalized spacings.
   Journal of Statistical Computation and Simulation, 53(3-4), 269-287.
   https://doi.org/10.1080/00949659508811711

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import ChenShapiroNormalityGofStatistic

statistic = ChenShapiroNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
