# Watson centered EDF statistic for a specified exponentiated Weibull.

`WatsonWeibullGofStatistic`

## Constructor

```text
a, k : float, optional
    Fixed positive finite shape parameters, both default to 1.
    The parameter a is an exponent, not a scale.
```

## Hypothesis, formula and calibration

```text
F(x)=(1-exp(-x**k))**a on x >= 0, with fixed a, k, scale 1
and location 0. Put u_i=F(x_(i)), i=1,...,n.

U2=W2-n*(mean(u_i)-1/2)**2; W2 is the Cramer-von Mises statistic.

Reject for right tail values.
Calibrate using the specified null and the same sample size.
Previously stored unversioned Weibull calibrations must be regenerated.
The general centered EDF functional is applied to the known CDF.
Centering residuals before squaring avoids subtracting two large sums.
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import WatsonWeibullGofStatistic

test = WatsonWeibullGofStatistic()
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```

## References

1. G. S. Watson (1961), Goodness-of-fit tests on a circle.
   [Source](https://doi.org/10.1093/biomet/48.1-2.109)


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Historical class names and codes
are retained; old critical values must not be reused after a formula change.
