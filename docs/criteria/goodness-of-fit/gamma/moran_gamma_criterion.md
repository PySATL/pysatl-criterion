# Negative log product of Gamma probability spacings.

M = -sum(log(n*D_j)), j=1,...,n+1, with D_j=u_j-u_(j-1),
u_0=0 and u_(n+1)=1. The historical n scaling is retained: this
is shifted upward by (n+1)*log((n+1)/n) relative to (n+1) scaling.
Zero observations or repeated values give positive infinity.
Log-CDF differences in the lower tail and log-survival differences
in the upper tail avoid subtracting CDF values rounded to one.
Unresolved spacings at float64 precision also produce infinity.
No primary source for this exact normalization was verified in the
source search; do not reuse tables with another normalization.

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
from pysatl_criterion.statistics.goodness_of_fit.gamma import MoranGammaGofStatistic

statistic = MoranGammaGofStatistic()
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; large values oppose the null. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
