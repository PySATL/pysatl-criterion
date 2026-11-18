# Zhang C for inverse gamma

## API and hypothesis

`ZhangCInverseGammaGofStatistic` has code `ZAC_INV_GAMMA_GOODNESS_OF_FIT`.

Both positive parameters must be fixed before observing the sample; location is zero. No parameter fitting is performed. The input must be a nonempty one-dimensional sample of finite, strictly positive observations. Ties and constant samples are accepted; rounded observations require appropriate calibration. The result is one scalar statistic, not a p-value. Large values reject the null.

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$, with sorted observations; $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$. The implementation uses log-CDF and log-survival directly, without epsilon clipping. Unrepresentable log tails raise `FloatingPointError`; arithmetic overflow in the final formula can still produce infinity.

## Formula

$$
Z_C=\sum_{i=1}^n\left[\log\frac{1-u_i}{u_i}-\log\frac{n-i+1/4}{i-3/4}\right]^2.
$$

## Example

```python
import numpy as np
from pysatl_criterion.distribution import InverseGammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.inverse_gamma import (
    ZhangCInverseGammaGofStatistic,
)

parameters = InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 3.0, "beta": 2.0})
statistic = ZhangCInverseGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.35, 0.5, 0.7, 1.0, 1.5, 3.0])
assert not np.isnan(value)
```

## Scientific references

[Zhang (2002), Powerful goodness-of-fit tests based on the likelihood ratio](https://doi.org/10.1111/1467-9868.00337), §§2–3; [full text](https://anakena.dcc.uchile.cl/~mnmonsal/eso.pdf).

These references support the statistic or its general construction, not a test
with estimated inverse-gamma parameters. See the [family overview](index.md)
and [mathematical verification](mathematical-validation.md) for calibration
and numerical qualifications.
