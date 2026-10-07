# Doostparast weighted record EDF distance with record-likelihood fitting.

`MahdiDoostparastWeibullGofStatistic`

## Constructor

```text
No constructor arguments. Both ordinary-Weibull scale and shape are unknown.
```

## Hypothesis, formula and calibration

```text
F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
and unknown positive scale eta and shape k. No constructor arguments.
Logarithmic location and scale are eliminated within every call.

The input is a complete sequence in acquisition order. Extract lower
records R_j and the counts K_j of observations until the next record,
including the record itself. Order records increasingly, carrying counts.
Fit eta,k by the record likelihood product f(R_j)*S(R_j)**(K_j-1).
At ordered records, the survival estimate is the product of
1-1/sum(K_l,l>=j). Return n*integral((S_hat-S_fit)**2/F_fit dF_fit)
over the entire support, including the first and last intervals.

Reject for right tail values.
Calibration must reproduce record extraction and condition on at least
two records. The generic unconditional resolver is blocked.
Previously stored unversioned Weibull calibrations must be regenerated.
Order matters for full acquisition sequences. At least two records
are required. Precompressed records require counts; do not sort an
ordinary sample before calling. Inverse-sampling calibration is external.
```

## Calling execute_statistic

```text
Compute the statistic described in the class Notes.

Parameters
----------
rvs : array_like
    One-dimensional finite strictly positive observations, n >= 2.
    Constant samples are invalid; ties are allowed unless stated below.
record_counts : array_like of int, optional
    If given, rvs must be strictly decreasing lower records and these
    positive counts include each record. Otherwise extract from rvs.
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import MahdiDoostparastWeibullGofStatistic

test = MahdiDoostparastWeibullGofStatistic()
value = test.execute_statistic([3.0, 1.2, 2.1, 0.4, 0.8, 0.2])
print(value)
```

## References

1. M. Doostparast (2011), Goodness-of-fit tests for Weibull populations
   on the basis of records, equations (9), (10), (13), (14), (17).
   [Source](https://arxiv.org/abs/1110.5509)


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Historical class names and codes
are retained; old critical values must not be reused after a formula change.
