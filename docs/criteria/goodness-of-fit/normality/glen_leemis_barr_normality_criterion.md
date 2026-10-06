# Glen-Leemis-Barr test for normality

Glen-Leemis-Barr order-statistic goodness-of-fit statistic.

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

For z=(x-mean(x))/s1, form u_i=Phi(z_(i)), then
v_i=BetaCDF(u_i; i,n+1-i), and sort the v_i again. Return
-n-sum_i((2*n+1-2*i)*log(v_(i))+(2*i-1)*log(1-v_(i)))/n.
These are reversed Anderson-Darling weights, not the ordinary AD
statistic. A primary source confirming this exact fitted formula was
not located in the review; no author-specific calibration is claimed.
The beta transforms are dependent and are not uniform after fitting.
Use simulation of this formula only. Log-binomial sums recover
beta tails that underflow in direct probability arithmetic.

## Examples

```python
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import GlenLeemisBarrNormalityGofStatistic

statistic = GlenLeemisBarrNormalityGofStatistic()
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
