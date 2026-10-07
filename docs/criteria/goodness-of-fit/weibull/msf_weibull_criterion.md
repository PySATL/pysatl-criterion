# Normalized log-spacing statistic (complete observations).

`MSFWeibullGofStatistic`

## Constructor

```text
No constructor arguments. Both ordinary-Weibull scale and shape are unknown.
```

## Hypothesis, formula and calibration

```text
F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
and unknown positive scale eta and shape k. No constructor arguments.
Logarithmic location and scale are eliminated within every call.

Let y_i=log(x_(i)), mu_i approximate E(log(E_(i))) for unit exponential
order statistics. GoFNS uses a fourth-order Taylor approximation to
Q(U_(i)), Q(p)=log(-log(1-p)), p=i/(n+1).
Let h_i=(y_(i+1)-y_i)/(mu_(i+1)-mu_i), g_i=h_i/sum(h_i).
Return sum(g_i, i=floor(n/2)+1,...,n-1).

Reject for right tail values.
Calibrate with repeated execute_statistic calls on ordinary Weibull
samples. Affine invariance in log(x) permits eta=k=1 for simulation.
Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
Previously stored unversioned Weibull calibrations must be regenerated.
The expectation approximation is retained explicitly; it is not an
exact order-statistic expectation. No censored-data calibration is
supported. LOS is infinite when a terminal normalized spacing is zero.
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import MSFWeibullGofStatistic

test = MSFWeibullGofStatistic()
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Historical class names and codes
are retained; old critical values must not be reused after a formula change.
