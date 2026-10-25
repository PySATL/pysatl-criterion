# Zhang-Wu A test for normality

**Class:** `ZhangWuANormalityGofStatistic`

**Code:** `ZWA_NORMALITY_GOODNESS_OF_FIT`

Zhang-Wu likelihood-ratio normality statistic ``Z_A``.

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
Large values of the statistic define rejection. Use at least 4
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

For z=(x-mean(x))/s1, let A=-sum_i(log(Phi(z_(i)))/
(n-i+1/2) + log(1-Phi(z_(i)))/(i-1/2)). Return 10*A-32.
The historical affine scaling is retained and must also be used in
calibration. Logs are evaluated without first rounding normal CDFs.

## References

[1] Zhang, J. and Wu, Y. (2005).
   Likelihood-ratio tests for normality.
   Computational Statistics & Data Analysis, 49(3), 709-721.
   https://doi.org/10.1016/j.csda.2004.05.034

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import ZhangWuANormalityGofStatistic

statistic = ZhangWuANormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
