# Kuiper EDF range after the Gamma transform.

V = max(i/n-u_i) + max(u_i-(i-1)/n), i=1,...,n.
No finite-sample correction is applied.

For ordered observations x_(i), let u_i=F(x_(i); alpha, beta),
where Gamma has shape alpha, rate beta and known origin zero.
Parameters are fixed, not fitted. Samples must be finite, real,
one-dimensional and nonempty, in the closed support [0, infinity).
Zero is accepted as a boundary value. Ties and constant samples are
allowed except where stated. Inputs and instance state are unchanged.
Continuous-null calibration assumes iid observations; rounded data
require calibration of the observation process.

## Parameters

```text
alpha, beta : float, optional
    Fixed finite positive shape and rate, respectively. Both default to 1.
```

## Source

N. H. Kuiper (1960), "Tests concerning random points on a circle",
Indagationes Mathematicae (Proceedings) 63, 38-47.
https://doi.org/10.1016/S1385-7258(60)50006-0

The general statistic is applied to Gamma probabilities.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit.gamma import KuiperGammaGofStatistic

statistic = KuiperGammaGofStatistic()
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; large values oppose the null. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
