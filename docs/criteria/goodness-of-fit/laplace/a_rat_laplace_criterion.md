# A_RAT: fitted Laplace statistic

Public class: `AlizadehNoughabiLikelihoodLaplaceGofStatistic`. Registry code: `A_RAT_LAPLACE_GOODNESS_OF_FIT`.

## Hypothesis and definition

H0: independent observations follow Laplace(t, s) for some unknown t and s > 0.
Pass `Distribution.DEFAULT.parse({})`. Fixed or partially fixed parameters are rejected.

Density-based empirical likelihood ratio A_rat (Noughabi, 2016).

Minimum over integer 1 <= m < min(n**delta, n/2) of the product
2m/[n*(x_(i+m)-x_(i-m))*f_hat(x_i)]. Endpoint ranks are clamped.
Products are evaluated in log space. ``log_scale=True`` returns log(A_rat)
and requires its own calibration. Zero spacings give +infinity.

The exact mathematical definitions and notation are in [fitted definitions](fitted-definitions.md).

## Options and limitations

`delta=0.5`, with `0 < delta < 1`. The window bound includes n/2 as in the comparison paper. `log_scale=False` returns the original product; overflow produces positive infinity. `log_scale=True` returns its logarithm and avoids product overflow. These settings are part of the calibration configuration.

Input must be a finite, real, one-dimensional nonconstant sample with n >= 3
(n >= 4 for DLO). Sorting does not modify the caller's sample. Extreme floating
point shifts can erase differences before execution; no implementation can recover
those differences. These methods return a statistic, not a p-value.

## Calibration

Use the statistic's declared rejection direction. Simulate independent Laplace(0,1)
samples of the same size and call `execute_statistic` on every replicate, preserving
all constructor settings. Location and scale are re-estimated on each call.
`get_hypothesis_generator` selects the standard representative while the public
hypothesis retains both parameters as unknown. Fixed-parameter EDF critical values
are not applicable.

## Example

```python
import numpy as np
from pysatl_criterion.distribution import LaplaceDistributionDescriptor as Distribution
from pysatl_criterion.statistics.goodness_of_fit.continuous.laplace import (
    AlizadehNoughabiLikelihoodLaplaceGofStatistic,
)

criterion = AlizadehNoughabiLikelihoodLaplaceGofStatistic(Distribution.DEFAULT.parse({}))
value = criterion.execute_statistic([-3.2, -1.4, -.8, -.1, .05, .5, .9, 1.7, 4.1])
assert np.isfinite(value)
```

## Scientific references

- [Scientific source 1](https://doi.org/10.1007/s40304-016-0095-0)
- [Scientific source 2](https://arxiv.org/abs/2007.06154)

[Catalog](index.md) · [Mathematical validation](mathematical-validation.md)
