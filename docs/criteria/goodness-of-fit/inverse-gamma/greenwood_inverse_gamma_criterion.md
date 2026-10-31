# Greenwood for inverse gamma

## API and hypothesis

`GreenwoodInverseGammaGofStatistic` has code `GRW_INV_GAMMA_GOODNESS_OF_FIT`.

Both positive parameters must be fixed before observing the sample; location is zero. No parameter fitting is performed. The input must be a nonempty one-dimensional sample of finite, strictly positive observations. Ties and constant samples are accepted; rounded observations require appropriate calibration. The result is one scalar statistic, not a p-value. Large values reject the null.

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$, with sorted observations; $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$. Ordinary CDF values can round to zero or one in extreme tails.

All $n+1$ spacings, including both endpoints, enter the sum. The null mean is $2/(n+2)$ and $1/(n+1)\le G\le1$. This is the raw spacing sum without standardization.

## Formula

$$
G=\sum_{i=0}^n(u_{i+1}-u_i)^2,\qquad u_0=0,\ u_{n+1}=1.
$$

## Example

```python
import numpy as np
from pysatl_criterion.distribution import InverseGammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.inverse_gamma import (
    GreenwoodInverseGammaGofStatistic,
)

parameters = InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 3.0, "beta": 2.0})
statistic = GreenwoodInverseGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.35, 0.5, 0.7, 1.0, 1.5, 3.0])
assert not np.isnan(value)
```

## Scientific references

[Does, Helmers and Klaassen (1988), Approximating the distribution of Greenwood’s statistic](https://ir.cwi.nl/pub/1694/1694D.pdf), eqs. (1)–(2).

These references support the statistic or its general construction, not a test
with estimated inverse-gamma parameters. See the [family overview](index.md)
and [mathematical verification](mathematical-validation.md) for calibration
and numerical qualifications.
