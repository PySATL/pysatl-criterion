# Krit LT3 discrete Laplace statistic with maximum-likelihood fits.

`LaplaceTransform3WeibullGofStatistic`

## Constructor

```text
parameters : ParameterValues
    An empty distribution schema; all distribution parameters are unknown.
```

## Hypothesis, formula and calibration

```text
F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
and unknown positive scale eta and shape k.
Pass Distribution.DEFAULT.parse({}) to declare both parameters unknown.
Logarithmic location and scale are eliminated within every call.

T=n*sum(exp(a*t-exp(a*t))*(mean(exp(-t*z))-Gamma(1-t))**2).
Here z are MLE-standardized logs and t=j/m, ceil(-2.5*m)<=j<=floor(0.49*m).
This is a discrete sum, with no integration-step multiplier.

Reject for right tail values.
Calibrate with repeated execute_statistic calls on ordinary Weibull
samples. Affine invariance in log(x) permits eta=k=1 for simulation.
Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
Previously stored unversioned Weibull calibrations must be regenerated.
Every simulation must preserve m and a. The generic resolver uses
the default execution settings only. LT3 at m=100 has 300 grid points.
Transform differences are evaluated in log space; statistics beyond the
float64 range return positive infinity.
```

## Calling execute_statistic

```text
Compute the statistic described in the class Notes.

Parameters
----------
rvs : array_like
    One-dimensional finite strictly positive observations, n >= 2.
    Constant samples are invalid; ties are allowed unless stated below.
m : int, optional
    Positive grid resolution, default 100.
a : float, optional
    Finite weight parameter, default -5; not a distribution parameter.
**kwargs : dict
    Unused common-interface keywords.

Returns
-------
statistic : float
    Scalar discrepancy. Infinite boundary penalties are preserved.

Raises
------
ValueError
    Invalid sample, unsupported settings, or unrepresentable fit.

Notes
-----
The input is never changed and estimated parameters are not retained.
```

## Example

```python
from pysatl_criterion.distribution.distributions import WeibullDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.weibull import LaplaceTransform3WeibullGofStatistic

test = LaplaceTransform3WeibullGofStatistic(Distribution.DEFAULT.parse({}))
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```

## References

1. M. Krit (2014), Goodness-of-fit tests for the Weibull distribution based
   on the Laplace transform, sections 2, 4, 5.
   [Source](https://numdam.org/item/JSFS_2014__155_3_135_0.pdf)


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Ordinary and exponentiated Weibull have separate distribution and criterion identities;
old critical values must not be reused after a formula or family change.
