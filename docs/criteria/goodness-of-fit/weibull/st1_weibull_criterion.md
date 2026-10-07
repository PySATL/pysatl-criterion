# Delta-method moment discrepancy for negative log Weibull observations.

`ST1WeibullGofStatistic`

## Constructor

```text
parameters : ParameterValues
    An empty distribution schema; all distribution parameters are unknown.
```

## Hypothesis, formula and calibration

```text
F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
and unknown positive scale eta and shape k.
Pass Distribution.DEFAULT.parse({}) to declare both parameters unknown.
Logarithmic location and scale are eliminated within every call.

Let Z be the standardized maximum Gumbel. Set g=E(Z**3), b=E(Z**4).
For sample standardized -log(x), let g_n and b_n use divisor n.
Influence functions are P3(z)=z**3-3*z-1.5*g*z**2+g/2 and
P4(z)=z**4-4*g*z-2*b*z**2+b. Put V_ij=E(Pi(Z)*Pj(Z)).
Return n*(g_n-g)**2/V_33.

Reject for right tail values.
Calibrate with repeated execute_statistic calls on ordinary Weibull
samples. Affine invariance in log(x) permits eta=k=1 for simulation.
Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
Previously stored unversioned Weibull calibrations must be regenerated.
These are locally defined moment statistics under the historical
ST names. Exact cumulants (j-1)!*zeta(j), j>=2, determine moments
through order eight. The delta method gives an asymptotic chi-square(1)
limit under the null, not an exact finite-sample law or the published
smooth-test normalization. Use simulation for finite samples.
A primary source establishing this exact implemented formula was not
verified in the literature search. Treat it as the specified local
discrepancy; no named-test tables or chi-square limit are asserted.
```

## Calling execute_statistic

```text
Compute the statistic described in the class Notes.

Parameters
----------
rvs : array_like
    One-dimensional finite strictly positive observations, n >= 2.
    Constant samples are invalid; ties are allowed unless stated below.
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
from pysatl_criterion.distribution.distributions import WeibullDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.weibull import ST1WeibullGofStatistic

test = ST1WeibullGofStatistic(Distribution.DEFAULT.parse({}))
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Ordinary and exponentiated Weibull have separate distribution and criterion identities;
old critical values must not be reused after a formula or family change.
