# Liao–Shimokawa for inverse gamma

## API and hypothesis

`LiaoShimokawaInverseGammaGofStatistic` has code `LIAO_SHIMOKAWA_INV_GAMMA_GOODNESS_OF_FIT`.

Both positive parameters must be fixed before observing the sample; location is zero. No parameter fitting is performed. The input must be a nonempty one-dimensional sample of finite, strictly positive observations. Ties and constant samples are accepted; rounded observations require appropriate calibration. The result is one scalar statistic, not a p-value. Large values reject the null.

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$, with sorted observations; $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$. The implementation uses log-CDF and log-survival directly, without epsilon clipping. Unrepresentable log tails raise `FloatingPointError`; arithmetic overflow in the final formula can still produce infinity.

The original paper concerns fitted Weibull/extreme-value hypotheses; its critical values do not apply here.

## Formula

$$
L_n=\frac1{\sqrt n}\sum_{i=1}^n\frac{\max(i/n-u_i,u_i-(i-1)/n)}{\sqrt{u_i(1-u_i)}}.
$$

## Example

```python
import numpy as np
from pysatl_criterion.distribution import InverseGammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.inverse_gamma import (
    LiaoShimokawaInverseGammaGofStatistic,
)

parameters = InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 3.0, "beta": 2.0})
statistic = LiaoShimokawaInverseGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.35, 0.5, 0.7, 1.0, 1.5, 3.0])
assert not np.isnan(value)
```

## Scientific references

[Liao and Shimokawa (1999), A new goodness-of-fit test for type-I extreme-value and 2-parameter Weibull distributions with estimated parameters](https://doi.org/10.1080/00949659908811965). The expression is also reproduced in [Al-Zahrani (2012), eq. (13)](https://www.m-hikari.com/ams/ams-2012/ams-125-128-2012/alzahraniAMS125-128-2012.pdf).

These references support the statistic or its general construction, not a test
with estimated inverse-gamma parameters. See the [family overview](index.md)
and [mathematical verification](mathematical-validation.md) for calibration
and numerical qualifications.
