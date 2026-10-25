# BRT_V3: fitted Laplace statistic

Public class: `BestRaynerThasV3LaplaceGofStatistic`. Registry code: `BRT_V3_LAPLACE_GOODNESS_OF_FIT`.

## Hypothesis and definition

H0: independent observations follow Laplace(t, s) for some unknown t and s > 0.
Pass `Distribution.DEFAULT.parse({})`. Fixed or partially fixed parameters are rejected.

Rescaled smooth component V3: sqrt(n/63) times sample skewness.

Composite Laplace null: both location and scale are unknown.
Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
Rejection direction: two-sided. Constant samples are invalid.

The exact mathematical definitions and notation are in [fitted definitions](fitted-definitions.md).

## Options and limitations

The signed, rescaled components use sample central moments with divisor n. The constants 63 and 1188 normalize the limiting variances to one. They differ from the unrescaled moment components with denominators 54 and 1072.8 and from the MLE-polynomial V3/V4 in PoweR. Calibration must match this normalization. Formula reconstruction and the independent moment check are described in [fitted definitions](fitted-definitions.md).

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
    BestRaynerThasV3LaplaceGofStatistic,
)

criterion = BestRaynerThasV3LaplaceGofStatistic(Distribution.DEFAULT.parse({}))
value = criterion.execute_statistic([-3.2, -1.4, -.8, -.1, .05, .5, .9, 1.7, 4.1])
assert np.isfinite(value)
```

## Scientific references

- [Scientific source 1](https://doi.org/10.1016/j.csda.2008.05.023)
- [Scientific source 2](https://arxiv.org/abs/2007.06154)

[Catalog](index.md) · [Mathematical validation](mathematical-validation.md)
