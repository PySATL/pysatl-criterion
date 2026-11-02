# Chi2PearsonLogLogisticGofStatistic

Pearson chi-squared for a fully specified log-logistic distribution.

[All log-logistic criteria](index.md) · [Mathematical verification](mathematical-validation.md)

## Hypothesis and input

Pass `Distribution.DEFAULT.parse({"alpha": alpha, "beta": beta})` as `parameters`.
Both scale `alpha` and shape `beta` must be fixed, finite and strictly positive.
Neither parameter is estimated. The sample must be a nonempty, finite, strictly
positive one-dimensional array; ties and a single observation are accepted.
Censored samples are not supported. The returned value is a scalar statistic;
large values reject the null (right tail).

## Definition

For ordered observations, $u_i=F_{\alpha,\beta}(x_{(i)})$,
$F_{\alpha,\beta}(x)=1/[1+(\alpha/x)^\beta]$ for $x>0$.
Write $r_i=i-1/2$, $D^+=\max_i(i/n-u_i)$ and
$D^-=\max_i(u_i-(i-1)/n)$.

$$
X^2=\sum_j\frac{(O_j-E_j)^2}{E_j}
$$

Here $O_j$ are observed cell counts and $E_j=np_j>0$ are null expected
counts, with equal totals. The keyword `binning` accepts a `BinningStrategy`.
The default `EqualProbabilityBins()` uses `PowerCount()` (at least two cells,
approximately $2n^{0.4}$, rounded up, subject to `max_bins`). To fix the number,
use `EqualProbabilityBins(5)`. Internal boundaries belong to the cell on the
right. The default does not merge small expected counts.
See [binning strategies](../binning/index.md).

An empty observed cell contributes $E_j$.

## Calibration

For a fixed continuous null, EDF statistics and equal-probability cell counts
can be calibrated through independent Uniform(0,1) samples. Other binning rules
require the actual null and the same rule in every replicate. A rounded-data
null must reproduce rounding. Passing parameters estimated from this sample
as fixed values does not account for fitting.

The library currently has no built-in log-logistic random generator: use external
simulation (for example `scipy.stats.fisk`). This class computes a statistic,
not a p-value. For frequency statistics, the chi-square law with $k-1$ degrees
of freedom is an asymptotic approximation for fixed cells with adequate expected
counts, not an exact small-sample calibration. Binning is included in the
frequency criterion's storage configuration.

## Example

```python
from pysatl_criterion.distribution import LogLogisticDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_logistic import (
    Chi2PearsonLogLogisticGofStatistic,
)
from pysatl_criterion.statistics.goodness_of_fit.binning import EqualProbabilityBins

parameters = Distribution.DEFAULT.parse({"alpha": 1.0, "beta": 1.0})
statistic = Chi2PearsonLogLogisticGofStatistic(parameters, binning=EqualProbabilityBins(4))
value = statistic.execute_statistic([0.1, 0.2, 0.5, 0.8, 1.5, 2.0, 4.0, 8.0])
assert value >= 0
```

## Scientific references

[Cressie and Read (1984), Multinomial Goodness-of-Fit Tests](https://doi.org/10.1111/j.2517-6161.1984.tb01318.x); [Pearson (1900)](https://doi.org/10.1080/14786440009463897)
