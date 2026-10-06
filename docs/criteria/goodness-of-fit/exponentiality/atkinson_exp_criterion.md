# Atkinson test for exponentiality

Atkinson statistic for zero-origin exponential observations.

## Hypothesis, formula and calibration

H0 is Exp(lam), lam > 0 unknown, with known origin zero.
``hypothesis().parameters()`` is empty. Positive rescaling leaves
the statistic unchanged, so Exp(1) simulation is valid.
Write n for sample size, x_(i) for sorted observations, and
y=x/mean(x). The implemented formula is:

    sqrt(n) * abs(mean(y**p)**(1/p) - Gamma(1+p)**(1/p)).

This is an unstandardized absolute moment contrast. No universal
normal or chi-square calibration is claimed. Negative powers at zero
use the limiting power mean zero. Near p=0 a log-gamma series is used.
For abs(p) < 1e-100 and strictly positive observations, the geometric
mean limit is used: the power correction is below float64 precision.
Near p=1 the contrast can lose relative precision through subtraction.

Reject in the upper tail. Calibrate the exact statistic returned here.
Use independent continuous, uncensored observations. Rounded or tied
data require calibration of the observation process. No parameters
are retained from a previous call. Rescaling uses a power of two
to preserve exact binary ties; unrepresentable positive ratios raise
ValueError. Strict comparisons near floating-point boundaries can
still depend on roundoff.
A primary source establishing this exact implemented convention
was not verified in the review; no published tables or asymptotic
law are endorsed unless derived explicitly above. See the audit.
Settings belong in the constructor. The current storage key omits
these settings; use fresh Monte Carlo calibration for this object,
not StorageLimitDistributionResolver shared across settings.

## Constructor parameters

```text
p : float, optional
    Finite power greater than -1, excluding 0 and 1; default 0.99.
    The reference gamma expression must be representable in float64.
    The exclusions avoid divergent moments and degenerate contrasts.
```

## Sample and result

```text
Compute the Atkinson scalar statistic.

Parameters
----------
rvs : array_like, shape (n,)
    Finite nonnegative observations; n >= 2. The total must be positive.
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
    AtkinsonExponentialityGofStatistic,
)

statistic = AtkinsonExponentialityGofStatistic()
value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
print(value)
```

## Author(s)

Lev Golofastov

See the [current audit and migration notes](../../../exponent-statistics-audit.md).
