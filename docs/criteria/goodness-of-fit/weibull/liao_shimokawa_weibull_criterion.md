# Liao-Shimokawa weighted EDF statistic with Weibull MLEs.

`LiaoShimokawaWeibullGofStatistic`

## Constructor

```text
No constructor arguments. Both ordinary-Weibull scale and shape are unknown.
```

## Hypothesis, formula and calibration

```text
F(x)=1-exp(-(x/eta)**k), x > 0, with location fixed at zero
and unknown positive scale eta and shape k. No constructor arguments.
Logarithmic location and scale are eliminated within every call.

L=sum(max(i/n-u_i,u_i-(i-1)/n)/sqrt(u_i*(1-u_i)))/sqrt(n),
where u_i=G(z_i) and z_i are MLE-standardized ordered logs.

Reject for right tail values.
Calibrate with repeated execute_statistic calls on ordinary Weibull
samples. Affine invariance in log(x) permits eta=k=1 for simulation.
Ordinary fixed-parameter KS tables do not apply to fitted CDFs.
Previously stored unversioned Weibull calibrations must be regenerated.
This selects the maximum-likelihood variant, not graphical estimates.
Log probabilities retain tail precision; there is no epsilon clipping.
Extremely large finite penalties may overflow float64 to infinity.
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
from pysatl_criterion.statistics.goodness_of_fit.weibull import LiaoShimokawaWeibullGofStatistic

test = LiaoShimokawaWeibullGofStatistic()
value = test.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(value)
```

## References

1. M. Liao and T. Shimokawa (1999), A new goodness-of-fit test for type-I
   extreme-value and 2-parameter Weibull distributions with estimated
   parameters. [Source](https://doi.org/10.1080/00949659908811965)


See [the Weibull audit](../../../weibull-statistics-audit.md) for migration details,
source verification limits and calibration changes. Historical class names and codes
are retained; old critical values must not be reused after a formula change.
