# Kuiper test for the Weibull family

## Description

`KuiperWeibullGofStatistic` has algorithm code `KUIPER_WEIBULL_GOODNESS_OF_FIT`.
It is generated from the shared EDF kernel.

The null is the ordinary Weibull family with zero location:

$$
F(x;k,\eta)=1-\exp\left[-(x/\eta)^k\right],\qquad x>0,\quad k,\eta>0.
$$

Both parameters are fitted by maximum likelihood for every sample.

## Usage

```python
from pysatl_criterion.distribution import WeibullDistributionDescriptor as Weibull
from pysatl_criterion.statistics.goodness_of_fit.continuous.weibull import (
    KuiperWeibullGofStatistic,
)

criterion = KuiperWeibullGofStatistic(Weibull.DEFAULT.parse({}))
statistic = criterion.execute_statistic([0.3, 0.6, 1.0, 1.4, 2.2])
print(statistic)
```

## Arguments

`parameters` — `Weibull.DEFAULT.parse({})`, declaring both shape and scale
unknown. Fixed or partially fixed parameter sets are unsupported.

`rvs` — a one-dimensional, finite, strictly positive, nonconstant sample with
at least two observations. Ties are allowed. Invalid samples raise `ValueError`.
The input is not modified and fitted parameters are not retained on the instance.

## Details

For ordered observations, set $u_i=F(X_{(i)};\widehat k,\widehat\eta)$.
The implemented statistic is

$$
V_n=D_n^++D_n^-.
$$

This is the unscaled Kuiper discrepancy.

Here $D_n^+=\max_i(i/n-u_i)$ and $D_n^-=\max_i(u_i-(i-1)/n)$.

## Calibration

Large values form the right-tail rejection region. `execute_statistic` returns
a statistic, not a p-value. Calibrate at the same sample size by simulating
ordinary Weibull samples and re-estimating both parameters for every replicate.
Log-location-scale invariance permits shape = scale = 1 for this simulation.

Critical values for a fully specified uniform or Weibull CDF do not apply to the
fitted model. The built-in Monte Carlo resolver repeats the fit through
`execute_statistic`; stored unversioned Weibull critical values are rejected.

## References

Kuiper, N. H. (1960). [Tests concerning random points on a circle](https://doi.org/10.1016/S1385-7258(60)50006-0). *Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A*, 63, 38–47.

The references identify the EDF discrepancy. Its application here uses estimated Weibull parameters and therefore requires the fitted-model calibration described above.

See the [Weibull criterion catalog](index.md) for all implementations and shared
calibration requirements.
