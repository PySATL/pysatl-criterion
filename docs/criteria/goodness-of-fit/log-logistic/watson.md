# WatsonLogLogisticGofStatistic

Watson for a fully specified log-logistic distribution.

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
U^2=W^2-n(\bar u-1/2)^2
$$

The implementation centers the deviations before squaring to avoid cancellation. No finite-sample correction is applied.

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
    WatsonLogLogisticGofStatistic,
)

parameters = Distribution.DEFAULT.parse({"alpha": 1.0, "beta": 1.0})
statistic = WatsonLogLogisticGofStatistic(parameters)
value = statistic.execute_statistic([0.1, 0.2, 0.5, 0.8, 1.5, 2.0, 4.0, 8.0])
assert value >= 0
```

## Scientific references

[Lemeshko et al. (2014), equations (1)–(6)](https://doi.org/10.3103/S875669901401004X) ([full text](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf)); [Watson (1961)](https://doi.org/10.1093/biomet/48.1-2.109)
