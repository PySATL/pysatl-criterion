# AP_Z: fitted Laplace statistic

Public class: `AlizadehNoughabiParkZLaplaceGofStatistic`. Registry code: `AP_Z_LAPLACE_GOODNESS_OF_FIT`.

## Hypothesis and definition

H0: independent observations follow Laplace(t, s) for some unknown t and s > 0.
Pass `Distribution.DEFAULT.parse({})`. Fixed or partially fixed parameters are rejected.

AP_z = log(2*theta_z) + 1 - H_z; m=1 by default.

Composite Laplace null: both location and scale are unknown.
Use Distribution.DEFAULT.parse({}); refit on every simulated sample.
Rejection direction: right. Constant samples are invalid.

The exact mathematical definitions and notation are in [fitted definitions](fitted-definitions.md).

## Options and limitations

Window `m` is an integer, defaults to **1**, and must satisfy `1 <= m < n/2`. This default is an explicit implementation setting, not the optimized size-dependent window table in the comparison paper. Choose a window before calibration. See the exact entropy estimates and corrected scales in [fitted definitions](fitted-definitions.md). `scale_estimator="corrected"` is the default; `"mle"` implements the corresponding maximum-likelihood-scale variant. Zero relevant spacings produce positive infinity.

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
    AlizadehNoughabiParkZLaplaceGofStatistic,
)

criterion = AlizadehNoughabiParkZLaplaceGofStatistic(Distribution.DEFAULT.parse({}))
value = criterion.execute_statistic([-3.2, -1.4, -.8, -.1, .05, .5, .9, 1.7, 4.1])
assert np.isfinite(value)
```

## Scientific references

- [Scientific source 1](https://doi.org/10.1080/00949655.2015.1104685)
- [Scientific source 2](https://arxiv.org/abs/2007.06154)

[Catalog](index.md) · [Mathematical validation](mathematical-validation.md)
