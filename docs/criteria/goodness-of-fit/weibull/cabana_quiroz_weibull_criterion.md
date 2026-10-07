# Krit maximum-likelihood Cabana-Quiroz CQ* quadratic statistic.

`CabanaQuirozWeibullGofStatistic`

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

v=sqrt(n)*(mean(exp(-s*z))-Gamma(1-s)), s=(-0.1,0.02).
Return v.T @ inverse(A) @ v, A=[[1.59,0.91],[0.91,0.53]],
where z are MLE-standardized log observations.

Reject for right tail values.
Calibrate with repeated execute_statistic calls on ordinary Weibull
samples. Affine invariance in log(x) permits eta=k=1 for simulation.
Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
Previously stored unversioned Weibull calibrations must be regenerated.
This uses the specified nonsingular weighting matrix, not an estimated
covariance matrix. A chi-square reference law is not asserted.
Transform values beyond the float64 range give numerical infinity.
```

## Calling execute_statistic

```text
Compute the statistic described in the class Notes.

Parameters
----------
rvs : array_like
    One-dimensional finite strictly positive observations, n >= 2.
    Constant samples are invalid; ties are allowed unless stated below.
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import CabanaQuirozWeibullGofStatistic

test = CabanaQuirozWeibullGofStatistic(Distribution.DEFAULT.parse({}))
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
