# SWRG test for normality

Rahman-Govindarajulu modification of the Shapiro-Wilk statistic.

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
The lower tail of the statistic defines rejection. Use at least 4
finite real observations in one dimension. Nonzero dispersion is required.
Ties are accepted unless a scale or contrast becomes undefined.
Write x_(i) for ascending observations (i=1,...,n), Phi and phi for
the standard normal CDF and density, s0=std(x,ddof=0), and
s1=std(x,ddof=1). Calculations do not modify the sample or retain fits.

For m_i=Phi**-1(i/(n+1)), f_i=phi(m_i), form weights
a*_i=-(n+1)*(n+2)*f_i*(m_(i-1)*f_(i-1)-2*m_i*f_i
+m_(i+1)*f_(i+1)), taking missing endpoint terms as zero.
Normalize a=a*/sqrt(sum(a* **2)); return
(sum(a_i*x_(i)))**2/sum((x-mean(x))**2). Small W_RG rejects.

## References

[1] Rahman, M. M. and Govindarajulu, Z. (1997).
   A modification of the test of Shapiro and Wilk for normality.
   Journal of Applied Statistics, 24(2), 219-236.
   https://doi.org/10.1080/02664769723828

## Examples

```python
from pysatl_criterion.distribution.distributions import NormalDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.normal import SWRGNormalityGofStatistic

statistic = SWRGNormalityGofStatistic(NormalDistributionDescriptor.DEFAULT.parse({}))
sample = [-1.7, -1.2, -0.9, -0.6, -0.3, -0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.1]
value = statistic.execute_statistic(sample)
bool(np.isfinite(value))
```

See [review and calibration changes](../../../normal-statistics-review.md).
