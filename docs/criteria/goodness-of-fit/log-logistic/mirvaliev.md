# MirvalievLogLogisticGofStatistic

Mirvaliev moment-fitted statistic on log-transformed complete data.

[All log-logistic criteria](index.md) · [Mathematical verification](mathematical-validation.md)

## Parameters

`parameters`: pass `LogLogisticDistributionDescriptor.DEFAULT.parse({})`.
Both alpha and beta are unknown and are fitted anew on every execution.
Partially or fully fixed parameters are unsupported.

n_intervals : int, optional
    Number of equiprobable cells, at least 2; default 8.

## Methods
execute_statistic(rvs, **kwargs)
    Return a scalar discrepancy.
hypothesis()
    Return the fixed parameters; omitted parameters are unknown.

## Notes
Apply equations (8)-(10) of [1] to log(X), a logistic location-scale
family. Fit its mean and standard deviation with divisor n, per call.
Both log-logistic parameters are unknown. In unit-variance coordinates,
cell residuals are v_j=(N_j-n/r)/sqrt(n/r), q_j=1/sqrt(r).
B contains fixed-boundary probability derivatives divided by sqrt(p).
C contains cell integrals of (Y, Y²-1) divided by sqrt(p),
V=diag(1,16/5), K=diag(1,2). Set A=I-qq'+C(V-C'C)^-1 C',
D=C-B K^-1 V, L=V+D' A D. Return v' A v-v' A D L^-1 D' A v.
Integrals are evaluated by adaptive quadrature. These are logistic
moments, not normal moments. No MLE or survival-hazard correction is used.
Reject large values. Log-location/scale invariance justifies simulation
at alpha=beta=1 with refitting. Preserve n and r. The asymptotic
chi-square(r-1) statement in [1] is not an exact finite-sample law.
No built-in log-logistic generator exists; Monte Carlo is external.
Storage omits r and is blocked; discard old MIRVALIEV calibrations.
The cited paper treats logistic data; logarithms give the log-logistic
application. A source for the former censored version was not found;
censored observations now raise ValueError. No fitted state is retained.

## References
[1] N. Pya (2004), "Goodness-of-fit tests for the logistic
   distribution", Mathematical Journal 4(2), 68-75, equations (8)-(10).
   [Source](https://math.kz/media/journal/journal2017-06-0725590.pdf)
## Examples
```python
from pysatl_criterion.distribution import LogLogisticDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_logistic import (
    MirvalievLogLogisticGofStatistic,
)

statistic = MirvalievLogLogisticGofStatistic(LogLogisticDistributionDescriptor.DEFAULT.parse({}))
value = statistic.execute_statistic(np.exp(np.linspace(-2, 2, 100)))
bool(np.isfinite(value))
```

## Input and result

Compute the statistic independently for the supplied sample.

## Parameters
rvs : array_like
    Positive finite 1D times (at least three, not constant), or
    (times, delta) with matching binary indicators, 1 for events.
    All indicators must be one; censoring is unsupported.
**kwargs : dict
    Accepted for the common interface; unused.

## Returns
statistic : float or numpy.float64
    Nonnegative discrepancy; larger values reject the null.

## Raises
ValueError
    Invalid sample or singular correction matrix.
RuntimeError
    Numerical failure produces a negative quadratic form.

## Calibration and interpretation

The moment fit is to $\log X$, not to $X$: with $m=\overline{\log X}$ and
$s^2=n^{-1}\sum(\log X_i-m)^2$, the implied parameters are
$\hat\alpha=e^m$ and $\hat\beta=\pi/(\sqrt3s)$. Computation uses standardized
logs and need not form these possibly overflowing parameters.
This applies the logistic statistic in Pya (2004) after the invertible logarithm
transformation. It is not a censored-data extension. At a fixed number of cells,
the source's limiting law is $\chi^2_{r-1}$ under its assumptions; calibrate finite
samples externally with moment fitting repeated in every replicate.
