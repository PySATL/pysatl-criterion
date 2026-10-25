# Cabana-Cabana1 test for normality

**Class:** `CabanaCabana1NormalityGofStatistic`

**Code:** `CC1_NORMALITY_GOODNESS_OF_FIT`

Cabaña-Cabaña normality statistic focused on skewness departures.

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

Use the l=5 process in Cabana and Cabana, equation (7).
For z=(x-mean(x))/s1, h_j=He_j/sqrt(j!), H_j=sum(h_j(z))/sqrt(n),
P(t)=sum_{j=1..5}(h_(j-1)(t)*H_(j+3)/sqrt(j)). Return
sup_t abs(Phi(t)*H_3-phi(t)*P(t)). Evaluate every real stationary
point (roots of H_3+t*P-P') and both limits at infinity.
This is the supremum of the truncated process, not the untruncated
infinite series. Polynomial root solving is subject to float64 accuracy.

## References

[1] Cabaña, A. and Cabaña, E. M. (2003).
   Tests of Normality Based on Transformed Empirical Processes.
   Methodology and Computing in Applied Probability, 5, 309-335.
   https://doi.org/10.1023/A:1026235220018

## Examples

```python
from pysatl_criterion.distribution import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.normal import CabanaCabana1NormalityGofStatistic

statistic = CabanaCabana1NormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
assert np.isfinite(value)
```

See the [complete catalog](index.md) and [mathematical verification](mathematical-validation.md).
