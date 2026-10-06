# Anderson--Darling statistic for the Laplace distribution

`AndersonDarlingLaplaceGofStatistic` tests independent observations from **Laplace(t, s) with
both parameters fixed in advance**. Its support is the real line. No parameters
are fitted by `execute_statistic`.

## Parameters and usage

`t=0.0` is a finite real scalar location; `s=1.0` is a finite positive real
scalar scale. `rvs` must be a nonempty, one-dimensional sample of finite real
numbers. A single observation, ties, and constant samples are accepted.
Invalid inputs raise `ValueError`. The sample and model are not modified.
`execute_statistic(rvs, **kwargs)` returns a scalar; extra keywords are ignored.

```python
from pysatl_criterion.statistics.goodness_of_fit.laplace import AndersonDarlingLaplaceGofStatistic

statistic = AndersonDarlingLaplaceGofStatistic(t=0.0, s=1.0)
value = statistic.execute_statistic([-1.0, 0.0, 0.5, 2.0])
```

## Formula and calibration

Let `u_i = F_(t,s)(x_(i))`, for sorted observations, `i=1,...,n`.
The Laplace CDF is `exp((x-t)/s)/2` for `x<t` and
`1-exp(-(x-t)/s)/2` otherwise.

```text
A² = -n - sum((2*i-1)/n * (log(u_i) + log(1-u_(n+1-i)))),
i = 1, ..., n. Log probabilities are evaluated analytically:
log(F(x)) = z-log(2) for z < 0 and log1p(-exp(-z)/2) otherwise,
where z = (x-t)/s; log(S(x)) follows by reflection.
This avoids exponential underflow in the logarithmic tails.
No finite-sample or fitted-parameter correction is applied.
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

M. A. Stephens (1974), "EDF Statistics for Goodness of Fit and
Some Comparisons", JASA 69, 730-737, Section 2 (definitions and
computation), Case 0 (fully specified continuous distribution).
https://doi.org/10.1080/01621459.1974.10480196
Full text: https://www.math.utah.edu/~morris/Courses/6010/p1/writeup/ks.pdf

This source describes the general statistic. Here it is applied to the known
Laplace CDF, not to a model fitted to the observations.

## Review status

See the [Laplace audit](../../../laplace-statistics-audit.md) for corrections,
calibration compatibility, and verification limits.
