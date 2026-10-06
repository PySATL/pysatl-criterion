# Kuiper test for beta distribution

## Description
Performs Kuiper goodness-of-fit test for the hypothesis that the sample comes from a beta distribution.
The statistic combines the largest positive and negative empirical distribution deviations.

Hypothesis of Beta Distribution
The null hypothesis is that the data comes from a beta distribution with positive shape parameters `alpha` and `beta` on the interval $[0, 1]$.

## Usage
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperBetaGofStatistic,
)


test_statistic = KuiperBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```

## Arguments
`alpha` - first positive shape parameter of the beta distribution. Default value is `1`.

`beta` - second positive shape parameter of the beta distribution. Default value is `1`.

`rvs` - a nonempty one-dimensional sample of finite values in `[0, 1]`, passed to `execute_statistic`. Both shape parameters must be finite. All these statistics reject for large values.

## Details
The implementation computes

$$ V = D^+ + D^- $$

where $D^+$ and $D^-$ are the one-sided deviations between empirical plotting positions and beta CDF values.

## Author(s)
Dmitry Deruzhinsky, Aleksei Tokarev, Vladimir Zakharov, Alexey Mironov

## References

The cited paper studies circular uniformity. Here its unscaled V = D+ + D- statistic is applied after the specified Beta CDF transform. No estimated-shape calibration or finite-sample scaling correction is implied.

N. H. Kuiper (1960), "Tests concerning random points on a
   circle", Proc. K. Ned. Akad. Wet. A 63, 38-47.
   https://doi.org/10.1016/S1385-7258(60)50006-0

## Examples
```python
from pysatl_criterion.statistics.goodness_of_fit import (
    KuiperBetaGofStatistic,
)


test_statistic = KuiperBetaGofStatistic(alpha=2, beta=5)
statistic_result = test_statistic.execute_statistic([0.08, 0.14, 0.22, 0.31, 0.38, 0.46, 0.57])
print(statistic_result)
```
