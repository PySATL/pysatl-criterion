# BHS test for normality

Brys-Hubert-Struyf MC-LR statistic for the normal family.

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

Return n*v.T*V**-1*v with v=(MC, LMC-0.198828, RMC-0.198828),
V=[[1.24581,0.322918,-0.322918],[0.322918,2.62068,-0.0123455],
[-0.322918,-0.0123455,2.62068]]. LMC=-MC(x<median(x));
RMC=MC(x>median(x)); median observations belong to neither half.
MC is the median of (upper+lower-2*median)/(upper-lower) over
lower<=median<=upper, with antisymmetric rank values for zero ties.
At least two observations must remain in each strict half. Constant
halves have MC=0. The exact kernel median uses O(n**2) time and
memory, unlike the fast selection algorithm in the paper. Constants
agree with the rounded normal row in Table 1; simulate for finite n.

## References

[1] Brys, G., Hubert, M., and Struyf, A. (2008). "Goodness-of-fit tests
   based on a robust measure of skewness." Computational Statistics,
   23, 429-442. https://doi.org/10.1007/s00180-007-0083-7

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import BHSNormalityGofStatistic

statistic = BHSNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
