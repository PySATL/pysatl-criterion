# GMG test for normality

Gel-Miao-Gastwirth normality statistic directed at heavy tails.

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

Return s0/J, where J=sqrt(pi/2)*mean(abs(x-median(x))).
This is a directed upper-tail statistic for heavy-tailed departures.
The ddof=0 convention is retained; its finite-sample law differs
from versions using the unbiased sample variance.

## References

[1] Gel, Y. R., Miao, W. and Gastwirth, J. L. (2007).
   Robust directed tests of normality against heavy-tailed alternatives.
   Computational Statistics & Data Analysis, 51(5), 2734-2746.
   https://doi.org/10.1016/j.csda.2006.08.022

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import GMGNormalityGofStatistic

statistic = GMGNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
