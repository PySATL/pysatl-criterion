# Pearson statistic on equiprobable Gamma quantile bins.

X2 = sum((O_j-E_j)**2/E_j), with E_j=n/bins and O_j the bin counts.
Bins are [q_j,q_(j+1)); the last includes its right endpoint.
Here q_j=Gamma.ppf(j/bins, alpha, scale=1/beta).
This is the power-divergence family at power=1.
Reject for large values. Under the fixed null, counts are multinomial
with probabilities 1/bins. Null calibration depends on n, bins and power.
The chi-square(bins-1) approximation requires sufficiently large expected
counts. Storage lookup is blocked because its key omits bins and power;
use MonteCarloLimitDistributionResolver, which uses this instance.
Unrepresentable quantile boundaries raise ValueError.

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
bins : int, optional
    Number of equal-probability bins, at least 2. Default is 8.
```

## Source

N. Cressie and T. R. C. Read (1984), "Multinomial Goodness-Of-Fit
Tests", J. R. Statist. Soc. B 46, 440-464.
https://doi.org/10.1111/j.2517-6161.1984.tb01318.x

The general statistic is applied to Gamma probabilities.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit.gamma import Chi2PearsonGammaGofStatistic

statistic = Chi2PearsonGammaGofStatistic()
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; large values oppose the null. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
