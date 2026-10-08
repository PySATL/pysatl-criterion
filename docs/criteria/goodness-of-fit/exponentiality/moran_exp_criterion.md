# Moran test for exponentiality

Moran statistic for zero-origin exponential observations.

## Hypothesis, formula and calibration

H0 is Exp(lam), lam > 0 unknown, with known origin zero.
``hypothesis().parameters()`` is empty. Positive rescaling leaves
the statistic unchanged, so Exp(1) simulation is valid.
Write n for sample size, x_(i) for sorted observations, and
y=x/mean(x). The implemented formula is:

    -digamma(1) + mean(log(y)), a signed centered log-mean
    statistic. It is not standardized to unit variance. Zero observations
    give negative infinity.

Reject in both tails. This signed contrast is not
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

The constructor has no arguments; `lam` is unknown and is not accepted.

## Sample and result

```text
Compute the Moran scalar statistic.

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
from pysatl_criterion.distribution.distributions import ExponentialDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.exponent import (
    MoranExponentialityGofStatistic,
)

statistic = MoranExponentialityGofStatistic(ExponentialDistributionDescriptor.DEFAULT.parse({}))
value = statistic.execute_statistic([0.2, 0.5, 1.0, 2.0])
print(value)
```

## Author(s)

Lev Golofastov

See the [current audit and migration notes](../../../exponent-statistics-audit.md).

## Calibration correction (2026-10-08)

The signed statistic now uses both tails. Its numerical value is unchanged;
previous one-sided p-values and rejection decisions must be recomputed.
See [independent verification](../../../exponential-verification.md).

## Scope and verification (2026-10-08)

For M=gamma+mean(log(X/mean(X))), the Dirichlet null representation gives
E(M)=gamma+psi(1)-psi(n)+log(n) and
Var(M)=psi1(1)/n-psi1(n). Here psi and psi1 are digamma and trigamma.
sqrt(n)*M/sqrt(pi²/6-1) tends to N(0,1). This is not the absolute
geometric-mean version of Mimoto–Zitikis. Both tails are used. It is not
universally consistent. Finite-sample calibration uses simulated ranks.

[Independent audit and outstanding checks](../../../exponential-verification.md).
