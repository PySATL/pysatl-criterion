# Maximum stabilized probability-plot distance with Weibull MLEs.

`SPPWeibullGofStatistic`

## Constructor

```text
No constructor arguments. Both ordinary-Weibull scale and shape are unknown.
```

## Hypothesis, formula and calibration

```text
F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
and unknown positive scale eta and shape k. No constructor arguments.
Logarithmic location and scale are eliminated within every call.

T=max(abs(2/pi*asin(sqrt((i-0.5)/n))-2/pi*asin(sqrt(G(z_i))))),
where z_i are ascending MLE-standardized logs, G(z)=1-exp(-exp(z)).

Reject for right tail values.
Calibrate with repeated execute_statistic calls on ordinary Weibull
samples. Affine invariance in log(x) permits eta=k=1 for simulation.
Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
Previously stored unversioned Weibull calibrations must be regenerated.
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import SPPWeibullGofStatistic

test = SPPWeibullGofStatistic()
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Historical class names and codes
are retained; old critical values must not be reused after a formula change.
