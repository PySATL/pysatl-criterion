# Watson centered EDF statistic after the Gamma transform.

U2 = W2 - n*(mean(u)-1/2)**2, where
W2 = 1/(12*n) + sum((u_i-(2*i-1)/(2*n))**2).
Circular rotation invariance concerns transformed uniform values,
not translation of the original Gamma observations.

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

G. S. Watson (1961), "Goodness-of-fit tests on a circle",
Biometrika 48, 109-114. https://doi.org/10.1093/biomet/48.1-2.109

The general statistic is applied to Gamma probabilities.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit.gamma import WatsonGammaGofStatistic

statistic = WatsonGammaGofStatistic()
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; large values oppose the null. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
