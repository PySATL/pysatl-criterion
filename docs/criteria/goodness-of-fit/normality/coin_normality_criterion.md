# Coin test for normality

Coin normality statistic based on polynomial regression.

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

The publication was identified, but the exact finite-sample
implementation is not fully certified against its primary text.

Regress z_(i)=(x_(i)-mean(x))/s1 on a_i and a_i**3 without
an intercept, where a_i are approximate expected normal order scores
from nscor2. Return the square of the cubic coefficient. Normal-score
approximations are only supported here for n<=2000. The reference
identifies the regression procedure; its score approximation is
retained and must be reproduced in calibration.

## References

[1] Coin, D. (2008).
   A goodness-of-fit test for normality based on polynomial regression.
   Computational Statistics & Data Analysis, 52(4), 2185-2198.
   https://doi.org/10.1016/j.csda.2007.07.012

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import CoinNormalityGofStatistic

statistic = CoinNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
