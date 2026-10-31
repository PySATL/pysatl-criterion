# NikulinLogLogisticGofStatistic

Bagdonavicius-Nikulin statistic with fitted log-logistic survival.

[All log-logistic criteria](index.md) · [Mathematical verification](mathematical-validation.md)

## Parameters

`parameters`: pass `LogLogisticDistributionDescriptor.DEFAULT.parse({})`.
Both alpha and beta are unknown and are fitted anew on every execution.
Partially or fully fixed parameters are unsupported.

n_intervals : int, optional
    Number of equal-exposure intervals, at least 2; default 5.

## Methods
execute_statistic(rvs, **kwargs)
    Return a scalar discrepancy.
hypothesis()
    Return the fixed parameters; omitted parameters are unknown.

## Notes
Both alpha and beta are unknown. Each call fits the right-censored
likelihood in standardized log-time location and log-scale coordinates.
Non-informative right censoring is assumed; delta=1 denotes an event.
H(t)=logaddexp(0, beta*log(t/alpha)); hence h(t)=beta*F(t)/t.
Section 5 of [1] chooses boundaries with sum_i min(H(t_i), H(a_j))
equal to j/k times total exposure, so all e_j equal total/k.
In Sections 3-4, Z=(U-e)/sqrt(n), A=diag(U/n), C contains cell sums
of event hazard scores divided by n, and I their outer-product mean.
Return Z'A^-1 Z + W'G^-1 W, W=C A^-1 Z, G=I-C A^-1 C'.
Scores use equivalent location/log-scale coordinates. G is calculated
by within-cell centering, without a ridge or pseudoinverse.
Empty event cells have positive exposure and return infinity. Singular
or ill-conditioned G is rejected. Small samples need fewer intervals.
Reject large values. The paper gives an asymptotic chi-square result
under regularity conditions; finite-sample calibration must reproduce
censoring, interval count and refitting. Generic Monte Carlo and storage
are blocked: they cannot express the censoring plan. Old BN_GOF
calibration values must be discarded. No estimates are retained.

## References
[1] V. Bagdonavicius and M. Nikulin (2011), "Chi-squared tests for
   general composite hypotheses from censored samples", Sections 3-5.
   [Source](https://doi.org/10.1016/j.crma.2011.01.007)
   [Source](https://www.numdam.org/item/10.1016/j.crma.2011.01.007.pdf)
## Examples
```python
from pysatl_criterion.distribution import LogLogisticDistributionDescriptor
import numpy as np
from pysatl_criterion.statistics.goodness_of_fit.continuous.log_logistic import NikulinLogLogisticGofStatistic

statistic = NikulinLogLogisticGofStatistic(LogLogisticDistributionDescriptor.DEFAULT.parse({}))
value = statistic.execute_statistic(np.exp(np.linspace(-2, 2, 100)))
bool(np.isfinite(value))
```

## Input and result

Compute the statistic independently for the supplied sample.

## Parameters
rvs : array_like
    Positive finite 1D times (at least three, not constant), or
    (times, delta) with matching binary indicators, 1 for events.
    Nikulin requires at least three events at two distinct times.
**kwargs : dict
    Accepted for the common interface; unused.

## Returns
statistic : float or numpy.float64
    Nonnegative discrepancy; larger values reject the null.
    Positive infinity when any event cell is empty.

## Raises
ValueError
    Invalid sample or singular correction matrix.
RuntimeError
    No finite MLE converges.

## Calibration and interpretation

This is the Bagdonavicius–Nikulin survival statistic, not the complete-data
Nikulin–Rao–Robson multinomial statistic. An event flag of **1 means observed
failure**, and 0 means right censoring. Without flags all observations are events.
The risk integral includes censored exposure. Intervals are $(a_{j-1},a_j]$,
so an event on an internal boundary belongs to the preceding interval.
The implementation uses the last observed time as its observation horizon.
The article's asymptotic result requires regularity and an appropriate
observation scheme; this implementation does not establish an exact null law
for a random sample-maximum horizon. Use external calibration reproducing
censoring, stopping, grouping and parameter fitting.
