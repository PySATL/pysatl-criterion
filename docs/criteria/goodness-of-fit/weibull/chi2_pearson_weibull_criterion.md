# Pearson counts statistic in equal-probability bins.

`Chi2PearsonWeibullGofStatistic`

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

B=ceil(sqrt(n)); T=sum((O_j-n/B)**2/(n/B)), j=1,...,B.
Bins in CDF space cover [0,1], including both tails.

Reject for right tail values.
Calibrate using the specified null and the same sample size.
Previously stored unversioned Weibull calibrations must be regenerated.
The source gives the general count statistic, applied here to known
cell probabilities. For small expected counts use simulation, not
chi-square tables; the number of bins depends on sample size.
For n=1 there is one bin and the statistic is identically zero.
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import Chi2PearsonWeibullGofStatistic

test = Chi2PearsonWeibullGofStatistic()
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```

## References

1. K. Pearson (1900), On the criterion that a given system of deviations
   from the probable in the case of a correlated system of variables is
   such that it can be reasonably supposed to have arisen from random
   sampling. [Source](https://doi.org/10.1080/14786440009463897)


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Historical class names and codes
are retained; old critical values must not be reused after a formula change.
