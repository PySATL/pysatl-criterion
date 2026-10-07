# Graph clique number test for normality

Clique-number statistic for a normal sample with known variance.

## Parameters

```text
parameters : ParameterValues
    Values with var fixed; omitted parameters are unknown.
```

## Methods

execute_statistic(rvs, **kwargs)
    Return one scalar statistic.
hypothesis()
    Report fixed null parameters only.
alternative()
    Report the critical tail.

## Notes

H0 is N(mu,var), with unknown mu and fixed var; hypothesis() reports var.
Calibrate at this variance; setting the simulation mean to zero is valid.
Both tails of the statistic defines rejection. Use at least 2
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

Connect i and j when abs(x_i-x_j) < (max(x)-min(x))/(10*m2),
where m2=mean((x-mean(x))**2). Return the largest clique size (using sorted intervals).
This local construction is translation invariant but not scale invariant.
var selects the null law for calibration; it is not inserted in the radius.
A primary source for this exact normality test was not located. No
published null law or omnibus power claim is made. Extreme scales that
make the radius or variance unrepresentable raise ValueError.

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import GraphCliqueNumberNormalityGofStatistic

statistic = GraphCliqueNumberNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({'var': 1}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
