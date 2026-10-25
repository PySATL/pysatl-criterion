# Glen-Leemis-Barr test for normality

**Class:** `GlenLeemisBarrNormalityGofStatistic`

**Code:** `GLB_NORMALITY_GOODNESS_OF_FIT`

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
Large values of the statistic define rejection. Use at least 4
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

## References

Glen, A. G., Leemis, L. M., Barr, D. R. (2001). Order statistics in goodness-of-fit testing. IEEE Transactions on Reliability, 50(2), 209–213.
https://doi.org/10.1109/24.963129

This identifies the order-statistic approach. The primary full text was not accessible during this audit; the reversed-weight, fitted-normal implementation is checked as the formula documented here, not certified as an exact reproduction of that paper.

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import GlenLeemisBarrNormalityGofStatistic

statistic = GlenLeemisBarrNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
