# Harris modification of Gnedenko F-test

Harris statistic for zero-origin exponential observations.

## Hypothesis, formula and calibration

H0 is Exp(lam), lam > 0 unknown, with known origin zero.
``hypothesis().parameters()`` is empty. Positive rescaling leaves
the statistic unchanged, so Exp(1) simulation is valid.
Write n for sample size, x_(i) for sorted observations, and
y=x/mean(x). The implemented formula is:

    mean(concatenate((d[:r], d[-r:])))/mean(d[r:-r]),
    d_i=(n-i+1)*(x_(i)-x_(i-1)), x_(0)=0. Independent
    exponential spacings give the exact F(4*r, 2*(n-2*r)) law.
    A zero denominator with positive numerator gives positive infinity.

Reject in the upper tail. This is a directional test; it is not
claimed consistent against every nonexponential alternative.
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
r : int or None, optional
    Tail count with 1 <= 2*r < n. Default None uses max(1, round(n/4)).
```

## Sample and result

```text
Compute the Harris scalar statistic.

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
    HarrisExponentialityGofStatistic,
)

statistic = HarrisExponentialityGofStatistic(ExponentialDistributionDescriptor.DEFAULT.parse({}))
value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
print(value)
```

## Author(s)

Lev Golofastov

See the [current audit and migration notes](../../../exponent-statistics-audit.md).
