# Kolmogorov-Smirnov distance for a specified exponentiated Weibull.

`KolmogorovSmirnovWeibullGofStatistic`

## Constructor

```text
a, k : float, optional
    Fixed positive finite shape parameters, default to 1 and 5 respectively.
    The parameter a is an exponent, not a scale.

alternative_type : AlternativeType, optional
    TWO_TAILED gives D; RIGHT gives D+; LEFT gives D-.
mode : str, optional
    Compatibility setting, default "auto"; no effect on the statistic.
```

## Hypothesis, formula and calibration

```text
F(x)=(1-exp(-x**k))**a on x >= 0, with fixed a, k, scale 1
and location 0. Put u_i=F(x_(i)), i=1,...,n.

D+=max(i/n-u_i); D-=max(u_i-(i-1)/n); D=max(D+,D-).
Every CDF-deviation direction has a right-tail rejection region.

Reject for right tail values.
Calibrate using the specified null and the same sample size.
Previously stored unversioned Weibull calibrations must be regenerated.
The reference concerns general continuous CDFs, applied here via F.
```

## Calling execute_statistic

```text
Compute the statistic described in the class Notes.

Parameters
----------
rvs : array_like
    One-dimensional finite nonnegative observations, n >= 1.
    Ties and constant samples are allowed.
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import KolmogorovSmirnovWeibullGofStatistic

test = KolmogorovSmirnovWeibullGofStatistic()
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```

## References

1. N. Smirnov (1948), Table for estimating the goodness of fit of empirical
   distributions. [Source](https://doi.org/10.1214/aoms/1177730256)


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Historical class names and codes
are retained; old critical values must not be reused after a formula change.
