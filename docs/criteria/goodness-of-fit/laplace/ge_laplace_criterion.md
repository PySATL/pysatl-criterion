# GE: fitted Laplace statistic

Public class: `GelLaplaceGofStatistic`. Registry code: `GE_LAPLACE_GOODNESS_OF_FIT`.

## Hypothesis and definition

H0: independent observations follow Laplace(t, s) for some unknown t and s > 0.
Pass `Distribution.DEFAULT.parse({})`. Fixed or partially fixed parameters are rejected.

Ge = n*q3^2/60 + n*(q4-6)^2/1200; qj use sqrt(2)*MAD scale.

Composite Laplace null: both location and scale are unknown.
Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
Rejection direction: right. Constant samples are invalid.

The exact mathematical definitions and notation are in [fitted definitions](fitted-definitions.md).

## Options and limitations

The constants 60 and 1200 are the small/moderate-sample choice recommended by Gel. A sample-specific/asymptotic variance estimator is not substituted.

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
    GelLaplaceGofStatistic,
)

criterion = GelLaplaceGofStatistic(Distribution.DEFAULT.parse({}))
value = criterion.execute_statistic([-3.2, -1.4, -.8, -.1, .05, .5, .9, 1.7, 4.1])
assert np.isfinite(value)
```

## Scientific references

- [Scientific source 1](https://doi.org/10.1016/j.csda.2009.10.008)
- [Scientific source 2](https://arxiv.org/abs/2007.06154)

[Catalog](index.md) · [Mathematical validation](mathematical-validation.md)
