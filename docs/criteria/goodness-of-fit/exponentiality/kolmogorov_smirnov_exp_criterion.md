# Test for exponentiality of Kolmogorov and Smirnov

KolmogorovSmirnov statistic for zero-origin exponential observations.

## Hypothesis, formula and calibration

H0 is Exp(lam) with known origin zero and fixed rate lam.
Write n for sample size, x_(i) for sorted observations, and
y=x/mean(x). The implemented formula is:

    D+ = max(i/n-F(x_(i))), D- = max(F(x_(i))-(i-1)/n).
    Return max(D+, D-), D+, or D- according to alternative_type.
    Here F(x) = 1-exp(-lam*x); no parameters are estimated.

Reject in the upper tail. Calibrate the exact statistic returned here.
Use independent continuous, uncensored observations. Rounded or tied
data require calibration of the observation process. No parameters
are retained from a previous call. Rescaling uses a power of two
to preserve exact binary ties; unrepresentable positive ratios raise
ValueError. Strict comparisons near floating-point boundaries can
still depend on roundoff.
This is the general EDF statistic applied through the known
exponential CDF (probability integral transform); [1] is a general
source, not a study of a special fitted exponential version.

## Constructor parameters

```text
alternative_type : AlternativeType, optional
    TWO_TAILED (default), RIGHT (D+), or LEFT (D-). All reject
    for large statistic values.
lam : float, optional
    Fixed, finite positive rate; default 1.
```

## Sample and result

```text
Compute the KolmogorovSmirnov scalar statistic.

Parameters
----------
rvs : array_like, shape (n,)
    Finite nonnegative observations; n >= 1. All-zero samples are allowed.
    A copy is used. Ties and positive constants are allowed except
    where they make the formula undefined (see Notes on the class).
**kwargs : dict
    No execution settings are supported; configure the constructor.

Returns
-------
statistic : float
    Value in the convention documented on the class. Infinite
    boundary values are retained where the formula has that limit.

Raises
------
ValueError
    Invalid sample, insufficient observations, invalid sample-dependent
    setting, undefined ratio, or rescaling underflow of positive data.
TypeError
    Unsupported execution keyword arguments.
```

## Example

```python
from pysatl_criterion.statistics.goodness_of_fit.exponent import (
    KolmogorovSmirnovExponentialityGofStatistic,
)

statistic = KolmogorovSmirnovExponentialityGofStatistic()
value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
print(value)
```

## References

1. Durbin, J. (1973). Distribution Theory for Tests Based on the
   Sample Distribution Function, chapter 1. SIAM.
   https://doi.org/10.1137/1.9781611970586.ch1

## Author(s)

Lev Golofastov

See the [current audit and migration notes](../../../exponent-statistics-audit.md).
