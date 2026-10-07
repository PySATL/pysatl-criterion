# Spacing-entropy discrepancy from a fitted log-Weibull density.

`KullbackLeiblerWeibullGofStatistic`

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

For MLE-standardized ordered logs z_i, define
H=mean(log(n*(z_(min(n,i+m))-z_(max(1,i-m)))/(2*m))).
Return -H-mean(z)+mean(exp(z)), the entropy-based estimate of
KL divergence to g(z)=exp(z-exp(z)). The finite-sample estimate may
be negative. Default m=min(floor(sqrt(n)),floor((n-1)/2)), at least 1.

Reject for right tail values.
Calibrate with repeated execute_statistic calls on ordinary Weibull
samples. Affine invariance in log(x) permits eta=k=1 for simulation.
Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
Previously stored unversioned Weibull calibrations must be regenerated.
Zero entropy-window spacings give positive infinity. Ties with
positive window spacings are allowed. Preserve m in every simulation;
the generic resolver supports only the default execution settings.
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
    One-dimensional finite strictly positive observations, n >= 3.
    Constant samples are invalid; ties are allowed unless stated below.
m : int, optional
    Entropy window, 1 <= m < n/2. Default grows as sqrt(n).
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import KullbackLeiblerWeibullGofStatistic

test = KullbackLeiblerWeibullGofStatistic(Distribution.DEFAULT.parse({}))
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Ordinary and exponentiated Weibull have separate distribution and criterion identities;
old critical values must not be reused after a formula or family change.
