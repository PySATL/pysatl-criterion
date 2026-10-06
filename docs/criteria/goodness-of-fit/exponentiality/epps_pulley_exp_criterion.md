# EppsPulley test for exponentiality

EppsPulley statistic for zero-origin exponential observations.

## Hypothesis, formula and calibration

H0 is Exp(lam), lam > 0 unknown, with known origin zero.
``hypothesis().parameters()`` is empty. Positive rescaling leaves
the statistic unchanged, so Exp(1) simulation is valid.
Write n for sample size, x_(i) for sorted observations, and
y=x/mean(x). The implemented formula is:

    sqrt(48*n) * (mean(exp(-y)) - 1/2). This is a signed
    Laplace-transform contrast, not an integrated squared distance.

Reject in the upper tail. This is a directional test; it is not
claimed consistent against every nonexponential alternative.
Use independent continuous, uncensored observations. Rounded or tied
data require calibration of the observation process. No parameters
are retained from a previous call. Rescaling uses a power of two
to preserve exact binary ties; unrepresentable positive ratios raise
ValueError. Strict comparisons near floating-point boundaries can
still depend on roundoff.

The constructor has no arguments; `lam` is unknown and is not accepted.

## Sample and result

```text
Compute the EppsPulley scalar statistic.

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
    EppsPulleyExponentialityGofStatistic,
)

statistic = EppsPulleyExponentialityGofStatistic()
value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
print(value)
```

## References

1. Epps, T. W. and Pulley, L. B. (1986). A Test of Exponentiality
   Vs. Monotone-Hazard Alternatives Derived from the Empirical
   Characteristic Function. JRSS B 48, 206-213.
   https://doi.org/10.1111/j.2517-6161.1986.tb01403.x

## Author(s)

Lev Golofastov

See the [current audit and migration notes](../../../exponent-statistics-audit.md).
