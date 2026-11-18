# Kuiper for inverse gamma

## API and hypothesis

`KuiperInverseGammaGofStatistic` has code `KUI_INVGAMMA_INV_GAMMA_GOODNESS_OF_FIT`.

Both positive parameters must be fixed before observing the sample; location is zero. No parameter fitting is performed. The input must be a nonempty one-dimensional sample of finite, strictly positive observations. Ties and constant samples are accepted; rounded observations require appropriate calibration. The result is one scalar statistic, not a p-value. Large values reject the null.

Write $u_i=F_{\alpha,\beta}(x_{(i)})$, $i=1,\ldots,n$, with sorted observations; $D^+=\max_i(i/n-u_i)$ and $D^-=\max_i(u_i-(i-1)/n)$. Ordinary CDF values can round to zero or one in extreme tails.

Use calibration for this normalization; unscaled, scaled and corrected Kuiper values are different. The correction does not provide an exact finite-sample p-value.

## Formula

$$
V=D^++D^-.
$$

## Example

```python
import numpy as np
from pysatl_criterion.distribution import InverseGammaDistributionDescriptor
from pysatl_criterion.statistics.goodness_of_fit.continuous.inverse_gamma import (
    KuiperInverseGammaGofStatistic,
)

parameters = InverseGammaDistributionDescriptor.DEFAULT.parse({"alpha": 3.0, "beta": 2.0})
statistic = KuiperInverseGammaGofStatistic(parameters)
value = statistic.execute_statistic([0.35, 0.5, 0.7, 1.0, 1.5, 3.0])
assert not np.isnan(value)
```

## Scientific references

[Lemeshko et al. (2014), Solving Problems of Using Some Nonparametric Goodness-of-Fit Tests](https://doi.org/10.3103/S875669901401004X), eqs. (1)–(6); [author-hosted paper](https://ami.nstu.ru/~headrd/seminar/publik_html/Awt_1_2014_eng.pdf).

These references support the statistic or its general construction, not a test
with estimated inverse-gamma parameters. See the [family overview](index.md)
and [mathematical verification](mathematical-validation.md) for calibration
and numerical qualifications.
