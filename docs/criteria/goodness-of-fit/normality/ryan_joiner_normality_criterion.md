# Ryan-Joiner test for normality

Ryan-Joiner probability-plot correlation statistic for normality.

## Parameters

```text
parameters : ParameterValues
    An empty distribution schema; all distribution parameters are unknown.
weighted : bool, optional
    If True, average the plotting probabilities within each group of
    tied observations before applying the normal quantile function.
    Default is False.
cte_alpha : {'3/8', '1/2', '0'}, optional
    Plotting-position constant ``a`` in ``(i-a)/(n-2*a+1)`` for one-based
    ranks. Default is ``'3/8'``. Unrecognized values currently fall back
    to ``3/8``.
```

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

Return corr(x_(i), Phi**-1(p_i)), p_i=(i-alpha)/(n+1-2*alpha).
cte_alpha selects alpha=0, 3/8, or 1/2. If weighted=True, replace
positions of tied observations by their group mean before taking
normal quantiles. This tie convention and nondefault positions need
separate calibration; they are not fixed distribution parameters.

## References

[1] Ryan, T. A., Jr. and Joiner, B. L. (1976). Normal Probability
   Plots and Tests for Normality. Technical report, Statistics
   Department, The Pennsylvania State University. Original report:
   https://www.additive-net.de/en/component/jdownloads/send/70-support/236-normal-probability-plots-and-tests-for-normality-thomas-a-ryan-jr-bryan-l-joiner

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import RyanJoinerNormalityGofStatistic

statistic = RyanJoinerNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
