# Centered log-order linear-estimate ratio (historical SB name).

`SBWeibullGofStatistic`

## Constructor

```text
No constructor arguments. Both ordinary-Weibull scale and shape are unknown.
```

## Hypothesis, formula and calibration

```text
F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
and unknown positive scale eta and shape k. No constructor arguments.
Logarithmic location and scale are eliminated within every call.

For centered y_i=log(x_(i))-mean(log(x)), let w_i=log((n+1)/(n-i+1)),
i<n, w_n=n-sum(w_i). Let v_i=w_i*(1+log(w_i))-1 for i<n and
v_n=0.4228*n-sum(v_i). Set b=sum((0.6079*v_i-0.2570*w_i)*y_i)/n.
Return n*b**2/sum(y_i**2). Centering removes rounded-coefficient drift.

Reject for left tail values.
Calibrate with repeated execute_statistic calls on ordinary Weibull
samples. Affine invariance in log(x) permits eta=k=1 for simulation.
Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
Previously stored unversioned Weibull calibrations must be regenerated.
Sb and SB retain the same historical code and formula. The rounded
linear weights are an approximation, not exact Shapiro-Wilk weights.
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import SBWeibullGofStatistic

test = SBWeibullGofStatistic()
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Historical class names and codes
are retained; old critical values must not be reused after a formula change.
