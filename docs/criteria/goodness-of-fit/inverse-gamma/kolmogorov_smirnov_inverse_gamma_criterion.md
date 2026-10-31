# Kolmogorov–Smirnov for inverse gamma

## API and hypothesis

`KolmogorovSmirnovInverseGammaGofStatistic` has code `KS_INV_GAMMA_GOODNESS_OF_FIT`.

Both positive parameters must be fixed before observing the sample; location is zero. No parameter fitting is performed. The input must be a nonempty one-dimensional sample of finite, strictly positive observations. Ties and constant samples are accepted; rounded observations require appropriate calibration. The result is one scalar statistic, not a p-value. Large values reject the null.

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$, with sorted observations; $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$. Ordinary CDF values can round to zero or one in extreme tails.

Optional `alternative_type=AlternativeType.RIGHT` returns $D^+$, `LEFT` returns $D^-$, and `TWO_TAILED` returns $D$. All three reject in the right tail of their statistic. `mode` is a compatibility option and does not change the scalar. Legacy directional storage calibration is blocked; prefer the dedicated D+ and D− classes.

## Formula

$$
D=\max(D^+,D^-).
$$

## Example

```python
import numpy as np
from pysatl_criterion.distribution import InverseGammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.inverse_gamma import (
    KolmogorovSmirnovInverseGammaGofStatistic,
)

parameters = InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 3.0, "beta": 2.0})
statistic = KolmogorovSmirnovInverseGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.35, 0.5, 0.7, 1.0, 1.5, 3.0])
assert not np.isnan(value)
```

## Scientific references

[Smirnov (1948), Table for Estimating the Goodness of Fit of Empirical Distributions](https://doi.org/10.1214/aoms/1177730256).

These references support the statistic or its general construction, not a test
with estimated inverse-gamma parameters. See the [family overview](index.md)
and [mathematical verification](mathematical-validation.md) for calibration
and numerical qualifications.
