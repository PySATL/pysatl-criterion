# Locally defined tail-weighted Gamma EDF discrepancy.

T = sum(max(i/n-u_i, u_i-(i-1)/n)/sqrt(u_i*(1-u_i)))/sqrt(n).
Weights use log-CDF and log-survival values for tail accuracy.
Zero observations yield positive infinity; extreme weights may overflow.
The public name is retained. No primary source establishing this exact
formula as a Gamma test was found. Liao and Shimokawa (1999) studied
other families and do not establish this Gamma calibration.
Use simulation of this formula under the specified Gamma null.

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

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit.gamma import MinToshiyukiGammaGofStatistic

statistic = MinToshiyukiGammaGofStatistic()
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; large values oppose the null. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
