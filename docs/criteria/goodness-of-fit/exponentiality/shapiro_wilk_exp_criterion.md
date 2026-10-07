# Shapiro-Wilk test for exponentiality

ShapiroWilk statistic for zero-origin exponential observations.

## Hypothesis, formula and calibration

H0 is Exp(lam), lam > 0 unknown, with known origin zero.
``hypothesis().parameters()`` is empty. Positive rescaling leaves
the statistic unchanged, so Exp(1) simulation is valid.
Write n for sample size, x_(i) for sorted observations, and
y=x/mean(x). The implemented formula is:

    n*(mean(x)-min(x))**2 / ((n-1)*sum((x-mean(x))**2)).
    This W is also translation invariant, so it cannot detect a positive
    shift of an exponential law. Both tails are used. Constant samples
    are undefined; n >= 3 avoids the identically-one n=2 case.

Computation uses `z=(x-min(x))/(max(x)-min(x))` before evaluating W.
Affine invariance preserves the formula while avoiding cancellation
in `mean(x)-min(x)` for nearly constant observations.

Reject in both tails. Calibrate the exact statistic returned here.
Use independent continuous, uncensored observations. Rounded or tied
data require calibration of the observation process. No parameters
are retained from a previous call. Rescaling uses a power of two
to preserve exact binary ties; unrepresentable positive ratios raise
ValueError. Strict comparisons near floating-point boundaries can
still depend on roundoff.

The constructor has no arguments; `lam` is unknown and is not accepted.

## Sample and result

```text
Compute the ShapiroWilk scalar statistic.

Parameters
----------
rvs : array_like, shape (n,)
    Finite nonnegative observations; n >= 3. The total must be positive.
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
from pysatl_criterion.distribution.distributions import ExponentialDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.exponent import (
    ShapiroWilkExponentialityGofStatistic,
)

statistic = ShapiroWilkExponentialityGofStatistic(ExponentialDistributionDescriptor.DEFAULT.parse({}))
value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
print(value)
```

## References

1. Shapiro, S. S. and Wilk, M. B. (1972). An Analysis of Variance
   Test for the Exponential Distribution (Complete Samples).
   Technometrics 14, 355-370.
   https://doi.org/10.1080/00401706.1972.10488921

2. Spinelli, J. J. and Stephens, M. A. (1987). Tests for Exponentiality
   When Origin and Scale Parameters Are Unknown. Technometrics 29, 471–476,
   Section 3, pp. 474–475.
   https://www.stat.cmu.edu/technometrics/80-89/VOL-29-04/v2904471.pdf

## Author(s)

Lev Golofastov

See the [current audit and migration notes](../../../exponent-statistics-audit.md).
