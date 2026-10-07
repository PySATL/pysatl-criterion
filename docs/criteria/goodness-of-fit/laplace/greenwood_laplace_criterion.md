# Greenwood statistic for the Laplace distribution

`GreenwoodLaplaceGofStatistic` tests independent observations from **Laplace(t, s) with
both parameters fixed in advance**. Its support is the real line. No parameters
are fitted by `execute_statistic`.

## Parameters and usage

`t=0.0` is a finite real scalar location; `s=1.0` is a finite positive real
scalar scale. `rvs` must be a nonempty, one-dimensional sample of finite real
numbers. A single observation, ties, and constant samples are accepted.
Invalid inputs raise `ValueError`. The sample and model are not modified.
`execute_statistic(rvs, **kwargs)` returns a scalar; extra keywords are ignored.

```python
from pysatl_criterion.distribution.distributions import LaplaceDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.laplace import GreenwoodLaplaceGofStatistic

statistic = GreenwoodLaplaceGofStatistic(LaplaceDistributionDescriptor.DEFAULT.parse({'t': 0.0, 's': 1.0}))
value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
```

## Formula and calibration

Let `u_i = F_(t,s)(x_(i))`, for sorted observations, `i=1,...,n`.
The Laplace CDF is `exp((x-t)/s)/2` for `x<t` and
`1-exp(-(x-t)/s)/2` otherwise.

```text
G = sum((u_i-u_(i-1))**2), i = 1, ..., n+1,
with u_0=0 and u_(n+1)=1. Both endpoint spacings are included;
repeated observations yield zero spacings. The result is unscaled.
This right-tail spacing test detects clustering; it is not asserted
to be consistent against every nonuniform alternative.
```

Large values reject the null. These are exact definitions of the statistics,
without asymptotic rescaling. Under the fixed continuous null the transformed
observations are independent uniforms, so the null distribution depends on
sample size, but not on `t` or `s`. `hypothesis()` nevertheless retains both
fixed parameters. Monte Carlo calibration generates Laplace(t,s) samples and
calls `execute_statistic` for each. Fitting parameters to the tested sample is
outside this API and requires calibration accounting for fitting.

CDF rounding can erase very small tail spacings. AD evaluates analytic log
probabilities, avoiding exponential underflow. Standardization avoids an
intermediate overflow in `x-t` when the quotient is representable. Float64
range limitations still apply; a mathematically finite AD value can overflow.
No observations or probabilities are clipped.

## Scientific source

R. J. M. M. Does, R. Helmers and C. A. J. Klaassen (1988),
"Approximating the distribution of Greenwood's statistic",
Statistica Neerlandica 42, 153-162, Section 1, equations (1)-(2).
https://ir.cwi.nl/pub/1694/1694D.pdf

This source describes the general statistic. Here it is applied to the known
Laplace CDF, not to a model fitted to the observations.

## Review status

See the [Laplace audit](../../../laplace-statistics-audit.md) for corrections,
calibration compatibility, and verification limits.
