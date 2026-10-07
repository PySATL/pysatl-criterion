# Cramer-von Mises distance for a specified exponentiated Weibull.

`CrammerVonMisesWeibullGofStatistic`

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

W2=1/(12*n)+sum((u_i-(2*i-1)/(2*n))**2).

Reject for right tail values.
Calibrate using the specified null and the same sample size.
Previously stored unversioned Weibull calibrations must be regenerated.
The reference treats the general EDF functional; here the known CDF
transforms the null to uniformity.
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import CrammerVonMisesWeibullGofStatistic

test = CrammerVonMisesWeibullGofStatistic()
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```

## References

1. T. W. Anderson and D. A. Darling (1952), Asymptotic theory of certain
   "goodness of fit" criteria based on stochastic processes.
   [Source](https://doi.org/10.1214/aoms/1177729437)


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Historical class names and codes
are retained; old critical values must not be reused after a formula change.
