# DD_SMOOTH: fitted Laplace statistic

Public class: `DataDrivenSmoothLaplaceGofStatistic`. Registry code: `DD_SMOOTH_LAPLACE_GOODNESS_OF_FIT`.

## Hypothesis and definition

H0: independent observations follow Laplace(t, s) for some unknown t and s > 0.
Pass `Distribution.DEFAULT.parse({})`. Fixed or partially fixed parameters are rejected.

Data-driven efficient Legendre score test for fitted Laplace data.

W_k=n*v_k'*(I-A_k*A_k')^(-1)*v_k, v=mean(phi)-A*mean(score).
Select the smallest k maximizing W_k-k*log(n), 1 <= k <= max_components.
This is an explicit Laplace specialization of the general score selection
construction, not a reproduction of the DD implementation compared in
Best et al. (2008). Calibrate by refitting simulated Laplace(0,1) samples.

The exact mathematical definitions and notation are in [fitted definitions](fitted-definitions.md).

## Options and limitations

`max_components=4` by default, with integer values from 1 to 10. Numerical population integrals form the nuisance-score projection once during construction; the sample calculation uses a Numba kernel. This is a documented Laplace specialization of Janic–Ledwina, **not a verified reproduction of the DD procedure compared by Best–Rayner–Thas**. No omnibus-consistency claim is made for a fixed maximum dimension.

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
    DataDrivenSmoothLaplaceGofStatistic,
)

criterion = DataDrivenSmoothLaplaceGofStatistic(Distribution.DEFAULT.parse({}))
value = criterion.execute_statistic([-3.2, -1.4, -.8, -.1, .05, .5, .9, 1.7, 4.1])
assert np.isfinite(value)
```

## Scientific references

- [Scientific source 1](https://www.impan.pl/~ledwina/Janic-Ledwina-24-09-08.pdf)
- [Scientific source 2](https://doi.org/10.1016/j.csda.2008.05.023)

[Catalog](index.md) · [Mathematical validation](mathematical-validation.md)
