# KS distance to a Gamma CDF fitted by sample moments.

Fit shape = mean(x)**2 / S2 and scale = S2 / mean(x), where
S2 = sum((x-mean(x))**2)/(n-1). Compute max(D+,D-) using the fitted
CDF. Both parameters are unknown; hypothesis().parameters() is empty.
Each call fits independently after rescaling to avoid overflow.
A nonconstant sample of size at least two with positive mean is required.
Zero is accepted as a support boundary by this moment estimator.
This is a Lilliefors-type construction, not the normality test or its
tables. No primary source verifying this exact Gamma moment estimator
and finite-sample calibration was found. The null law depends on shape.
Generic Monte Carlo and storage calibration are blocked: external
calibration must specify the shape or a justified composite-null scheme
and refit each replicate. Ordinary KS tables are inappropriate.
The constructor no longer accepts alpha or beta.

## Usage

```python
from pysatl_criterion.statistics.goodness_of_fit.gamma import LillieforsGammaGofStatistic

statistic = LillieforsGammaGofStatistic()
value = statistic.execute_statistic([0.2, 0.7, 1.3, 2.1, 3.4])
```

Returns a scalar; large values oppose the null. Invalid samples raise `ValueError`.
See the [audit](../../../gamma-statistics-audit.md) for calibration changes.
